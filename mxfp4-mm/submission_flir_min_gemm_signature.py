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


def _tile_m_for_m(m):
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    if m <= 128:
        return 128
    return 256


def _build_flir_probe(tile_m, tile_n):
    gpu_arch = get_rocm_arch()
    dyn = ir.ShapedType.get_dynamic_size()
    module_name = f"diag_gemm_sig_tm{tile_m}_tn{tile_n}"

    class _Probe(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        @flir.kernel
        def kernel_probe(
            self: flir.T.i64,
            arg_c: lambda: memref(dyn, T.bf16),
            arg_a: lambda: memref(dyn, T.ui8),
            arg_b: lambda: memref(dyn, T.ui8),
            arg_scale_a: lambda: memref(dyn, T.i32),
            arg_scale_b: lambda: memref(dyn, T.i32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c0 = arith.constant(0, type=T.i32)
            z = arith.constant(0.0, type=T.bf16)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
            buffer_ops.buffer_store(z, c_rsrc, c0)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(dyn, T.bf16),
            arg_a: lambda: memref(dyn, T.ui8),
            arg_b: lambda: memref(dyn, T.ui8),
            arg_scale_a: lambda: memref(dyn, T.i32),
            arg_scale_b: lambda: memref(dyn, T.i32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            gx = (c_m + tm - one) / tm
            gy = (c_n + tn - one) / tn

            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "kernel_probe"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, c_m, c_n, c_k],
            )

    return flydsl.compile(_Probe())


def _get_flir_executable(tile_m, tile_n):
    global _FLIR_EXECUTABLE
    cache_key = (tile_m, tile_n)
    if _FLIR_EXECUTABLE is None:
        _FLIR_EXECUTABLE = {}
    exe = _FLIR_EXECUTABLE.get(cache_key)
    if exe is None:
        exe = _build_flir_probe(tile_m, tile_n)
        _FLIR_EXECUTABLE[cache_key] = exe
    return exe


def custom_kernel(data):
    a, _, _, b_shuffle, b_scale_sh = data
    m, k = a.shape
    n = b_shuffle.shape[0]

    tm = _tile_m_for_m(m)
    tn = 256 if n >= 256 else 128
    pad_m = (m + tm - 1) // tm * tm

    a_padded = torch.nn.functional.pad(a, (0, 0, 0, pad_m - m)) if m != pad_m else a
    a_q, a_scale_sh = _quant_mxfp4(a_padded, shuffle=True)

    scratch_c = torch.empty((pad_m, n), device=a.device, dtype=torch.bfloat16)
    exe = _get_flir_executable(tm, tn)
    exe(
        scratch_c.view(-1),
        a_q.view(torch.uint8).view(-1),
        b_shuffle.view(torch.uint8).view(-1),
        a_scale_sh.view(torch.int32).view(-1),
        b_scale_sh.view(torch.int32).view(-1),
        pad_m,
        n,
        k,
    )

    output = aiter.gemm_a4w4(
        a_q,
        b_shuffle,
        a_scale_sh,
        b_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return output[:m, :]