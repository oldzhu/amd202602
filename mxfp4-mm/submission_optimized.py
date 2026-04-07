"""Experimental MXFP4-MM path for inline HIP compile probing."""

import os
import sys
from typing import Any

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


_LOG_KEYS: set[str] = set()
_HIP_MODULE: Any | None = None
_HIP_MODULE_STATE = "uninitialized"
_HIP_PROBE_ATTEMPTED = False
_CPP_SOURCE = r"""
#include <torch/extension.h>

void launch_probe(torch::Tensor scratch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_probe", &launch_probe, "launch_probe");
}
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cstdint>

#include <hip/hip_runtime.h>

__global__ void probe_kernel(int32_t* scratch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scratch[0] = 7;
    }
}

void launch_probe(torch::Tensor scratch) {
    probe_kernel<<<1, 64, 0, 0>>>(reinterpret_cast<int32_t*>(scratch.data_ptr()));
}
"""


def _log_once(key: str, message: str) -> None:
    if key in _LOG_KEYS:
        return
    _LOG_KEYS.add(key)
    print(f"[mxfp4-mm hip] {message}", file=sys.stderr)


def _quant_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp4, x_scale = dynamic_mxfp4_quant(x)
    x_scale = e8m0_shuffle(x_scale)
    return x_fp4.view(dtypes.fp4x2), x_scale.view(dtypes.fp8_e8m0)


def _baseline_gemm(
    a_q: torch.Tensor,
    b_shuffle: torch.Tensor,
    a_scale_sh: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> torch.Tensor:
    return aiter.gemm_a4w4(
        a_q,
        b_shuffle,
        a_scale_sh,
        b_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )


def _get_hip_module() -> Any:
    global _HIP_MODULE
    global _HIP_MODULE_STATE

    if _HIP_MODULE is not None:
        return _HIP_MODULE
    if _HIP_MODULE_STATE == "failed":
        raise RuntimeError("inline HIP module initialization already failed")

    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
    os.environ.setdefault("CXX", "clang++")

    try:
        _HIP_MODULE = load_inline(
            name="mxfp4_mm_hip_launch_probe_v2",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=None,
            extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
            verbose=False,
            with_cuda=True,
        )
    except Exception:
        _HIP_MODULE_STATE = "failed"
        raise

    _HIP_MODULE_STATE = "ready"
    _log_once("hip-compile", "inline HIP compile-probe module compiled successfully")
    return _HIP_MODULE


def _probe_inline_hip_compile(a_q: torch.Tensor, b_shuffle: torch.Tensor) -> None:
    global _HIP_PROBE_ATTEMPTED
    if _HIP_PROBE_ATTEMPTED:
        return
    _HIP_PROBE_ATTEMPTED = True

    shape_text = f"M={a_q.shape[0]} N={b_shuffle.shape[0]} K={a_q.shape[1] * 2}"

    try:
        module = _get_hip_module()
    except Exception as exc:
        _log_once(
            "hip-compile-failed",
            f"inline HIP compile probe unavailable for {shape_text}: {type(exc).__name__}: {exc}",
        )
        return

    _log_once(
        "hip-launch-skipped",
        f"inline HIP compile probe succeeded for {shape_text}; launch is intentionally skipped because even the scratch-only launch path times out remotely",
    )

    del module


def custom_kernel(data: input_t) -> output_t:
    a, _, _, b_shuffle, b_scale_sh = data

    if not a.is_contiguous():
        a = a.contiguous()

    a_q, a_scale_sh = _quant_mxfp4(a)
    output = _baseline_gemm(a_q, b_shuffle, a_scale_sh, b_scale_sh)
    _probe_inline_hip_compile(a_q, b_shuffle)
    return output