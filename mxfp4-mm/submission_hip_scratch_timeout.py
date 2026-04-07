"""Manual MM test: pure HIP compile plus scratch-only launch."""

import os
import sys
from typing import Any

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


_LOG_KEYS: set[str] = set()
_HIP_MODULE: Any | None = None
_HIP_MODULE_STATE = "uninitialized"
_HIP_PROBE_ATTEMPTED = False
_SCRATCH_CACHE: dict[tuple[str, int], torch.Tensor] = {}

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
    print(f"[mxfp4-mm hip manual] {message}", file=sys.stderr)


def _get_scratch(device: torch.device) -> torch.Tensor:
    device_index = -1 if device.index is None else device.index
    cache_key = (device.type, device_index)
    scratch = _SCRATCH_CACHE.get(cache_key)
    if scratch is None:
        scratch = torch.zeros((1,), dtype=torch.int32, device=device)
        _SCRATCH_CACHE[cache_key] = scratch
    return scratch


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
            name="mxfp4_mm_hip_manual_timeout_v1",
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
    _log_once("hip-compile", "inline HIP scratch-only module compiled successfully")
    return _HIP_MODULE


def _probe_inline_hip_launch(a: torch.Tensor, b_shuffle: torch.Tensor) -> None:
    global _HIP_PROBE_ATTEMPTED
    if _HIP_PROBE_ATTEMPTED:
        return
    _HIP_PROBE_ATTEMPTED = True

    shape_text = f"M={a.shape[0]} N={b_shuffle.shape[0]} K={a.shape[1]}"

    try:
        module = _get_hip_module()
    except Exception as exc:
        _log_once(
            "hip-compile-failed",
            f"inline HIP scratch-only probe unavailable for {shape_text}: {type(exc).__name__}: {exc}",
        )
        return

    scratch = _get_scratch(a.device)
    scratch.zero_()

    _log_once("hip-launch-start", f"launching scratch-only HIP probe for {shape_text}")
    module.launch_probe(scratch)
    _log_once("hip-launch-done", f"scratch-only HIP probe completed for {shape_text}; scratch0={int(scratch[0].item())}")


def custom_kernel(data: input_t) -> output_t:
    a, _, _, b_shuffle, _ = data

    if not a.is_contiguous():
        a = a.contiguous()

    _probe_inline_hip_launch(a, b_shuffle)

    output_shape = (a.shape[0], b_shuffle.shape[0])
    return torch.zeros(output_shape, dtype=torch.bfloat16, device=a.device)