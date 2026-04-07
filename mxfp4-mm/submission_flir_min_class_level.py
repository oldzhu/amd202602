import torch

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

from _mlir import ir
import flydsl
from flydsl.dialects.ext import arith, buffer_ops, flir
from flydsl.lang.ir.types import T, memref
from flydsl.runtime.device import get_rocm_arch


_FLIR_EXECUTABLE = None


def _quant_mxfp4(x, shuffle=True):
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)


def _build_flir_noop():
    gpu_arch = get_rocm_arch()
    dyn = ir.ShapedType.get_dynamic_size()

    class _Noop(flir.MlirModule):
        GPU_MODULE_NAME = "diag_noop_class_level"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        @flir.kernel
        def kernel_noop(
            self: flir.T.i64,
            arg_out: lambda: memref(dyn, T.i32),
        ):
            c0 = arith.constant(0, type=T.i32)
            c7 = arith.constant(7, type=T.i32)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
            buffer_ops.buffer_store(c7, out_rsrc, c0)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: memref(dyn, T.i32),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(32, index=True)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "kernel_noop"],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_out],
            )

    return flydsl.compile(_Noop())


def _get_flir_executable():
    global _FLIR_EXECUTABLE
    if _FLIR_EXECUTABLE is None:
        _FLIR_EXECUTABLE = _build_flir_noop()
    return _FLIR_EXECUTABLE


def custom_kernel(data):
    a, _, _, b_shuffle, b_scale_sh = data

    scratch = torch.zeros((1,), device=a.device, dtype=torch.int32)
    exe = _get_flir_executable()
    exe(scratch.view(-1))

    a_q, a_scale_sh = _quant_mxfp4(a, shuffle=True)
    return aiter.gemm_a4w4(
        a_q,
        b_shuffle,
        a_scale_sh,
        b_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )