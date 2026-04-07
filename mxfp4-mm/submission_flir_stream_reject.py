"""Manual MM test: minimal flir-based FlyDSL launch."""

import re
import sys
from typing import Any

import torch
from task import input_t, output_t

from _mlir import ir
import flydsl
from flydsl.dialects.ext import arith, buffer_ops, flir
from flydsl.runtime.device import get_rocm_arch


_LOG_KEYS: set[str] = set()
_FLIR_EXECUTABLE: Any | None = None


def _log_once(key: str, message: str) -> None:
    if key in _LOG_KEYS:
        return
    _LOG_KEYS.add(key)
    print(f"[mxfp4-mm flir manual] {message}", file=sys.stderr)


def _summarize_text(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + " ..."


def _extract_symbol_names(text: str) -> list[str]:
    symbols = set(re.findall(r"@([A-Za-z0-9_.$]+)", text))
    return sorted(symbols)


def _log_attr(exe: Any, attr_name: str) -> None:
    try:
        value = object.__getattribute__(exe, attr_name)
    except Exception as exc:
        _log_once(f"flir-attr-{attr_name}", f"attr {attr_name} unavailable: {type(exc).__name__}: {exc}")
        return

    if callable(value):
        _log_once(f"flir-attr-{attr_name}", f"attr {attr_name} is callable type={type(value).__name__}")
        return

    if isinstance(value, str):
        symbols = _extract_symbol_names(value)
        _log_once(
            f"flir-attr-{attr_name}",
            f"attr {attr_name} string len={len(value)} symbols={symbols[:24]} text={_summarize_text(value)}",
        )
        return

    _log_once(f"flir-attr-{attr_name}", f"attr {attr_name} type={type(value).__module__}.{type(value).__name__}")


def _log_engine_symbols(engine: Any) -> None:
    try:
        names = sorted(name for name in object.__dir__(engine) if not name.startswith("__") or name == "__call__")
        _log_once("flir-engine-dir", f"engine attrs={names[:60]}")
    except Exception as exc:
        _log_once("flir-engine-dir", f"engine dir failed: {type(exc).__name__}: {exc}")
        return

    candidate_methods = ["raw_lookup", "lookup", "invoke", "dump_to_object_file"]
    candidate_symbols = [
        "__call__",
        "kernel_noop",
        "diag_noop",
        "_mlir_ciface___call__",
        "_mlir_ciface_kernel_noop",
    ]

    for method_name in candidate_methods:
        try:
            method = object.__getattribute__(engine, method_name)
        except Exception as exc:
            _log_once(
                f"flir-engine-method-{method_name}",
                f"engine method {method_name} unavailable: {type(exc).__name__}: {exc}",
            )
            continue

        _log_once(
            f"flir-engine-method-{method_name}",
            f"engine method {method_name} is callable type={type(method).__name__}",
        )

        if method_name in {"raw_lookup", "lookup"}:
            for symbol_name in candidate_symbols:
                try:
                    result = method(symbol_name)
                    _log_once(
                        f"flir-engine-{method_name}-{symbol_name}",
                        f"{method_name}({symbol_name}) -> {result}",
                    )
                except Exception as exc:
                    _log_once(
                        f"flir-engine-{method_name}-{symbol_name}",
                        f"{method_name}({symbol_name}) failed: {type(exc).__name__}: {exc}",
                    )


def _build_flir_noop() -> Any:
    gpu_arch = get_rocm_arch()

    class _Noop(flir.MlirModule):
        GPU_MODULE_NAME = "diag_noop"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(32, index=True)
            dyn = ir.ShapedType.get_dynamic_size()

            @flir.kernel
            def kernel_noop(
                self: flir.T.i64,
                arg_out: lambda: flir.memref(dyn, flir.T.i32),
            ):
                c0 = arith.constant(0, type=flir.T.i32)
                c7 = arith.constant(7, type=flir.T.i32)
                out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
                buffer_ops.buffer_store(c7, out_rsrc, c0)

            @flir.jit
            def __call__(
                self: flir.T.i64,
                arg_out: lambda: flir.memref(dyn, flir.T.i32),
            ):
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel_noop"],
                    grid_size=(c1, c1, c1),
                    block_size=(bdx, c1, c1),
                    kernel_operands=[arg_out],
                )

    module = _Noop()
    return flydsl.compile(module)


def _get_flir_executable() -> Any:
    global _FLIR_EXECUTABLE
    if _FLIR_EXECUTABLE is None:
        _log_once("flir-compile-start", "compiling minimal flir noop module")
        _FLIR_EXECUTABLE = _build_flir_noop()
        _log_once("flir-compile-done", "compiled minimal flir noop module")
    return _FLIR_EXECUTABLE


def _probe_flir_ir() -> None:
    exe = _get_flir_executable()

    exe_type = f"{type(exe).__module__}.{type(exe).__name__}"
    _log_once("flir-exe-type", f"executor type={exe_type}")

    try:
        names = sorted(name for name in object.__dir__(exe) if not name.startswith("__") or name == "__call__")
        _log_once("flir-exe-dir", f"executor attrs={names[:40]}")
    except Exception as exc:
        _log_once("flir-exe-dir", f"object.__dir__ failed: {type(exc).__name__}: {exc}")

    for attr_name in [
        "__call__",
        "_llvm_sigs",
        "dump",
        "ir",
        "source_ir",
        "module",
        "_module",
        "entry",
        "_entry",
        "_ir_text",
        "asm",
        "mlir_module",
        "engine",
        "_engine",
        "raw_lookup",
    ]:
        _log_attr(exe, attr_name)

    try:
        llvm_sigs = object.__getattribute__(exe, "_llvm_sigs")
        _log_once("flir-llvm-sigs", f"_llvm_sigs={llvm_sigs}")
    except Exception as exc:
        _log_once("flir-llvm-sigs", f"_llvm_sigs unavailable: {type(exc).__name__}: {exc}")

    try:
        engine = object.__getattribute__(exe, "engine")
    except Exception:
        engine = None

    if engine is not None:
        _log_engine_symbols(engine)

    try:
        dump_fn = object.__getattribute__(exe, "dump")
    except Exception:
        dump_fn = None

    if callable(dump_fn):
        try:
            dump_result = dump_fn()
            if isinstance(dump_result, str):
                symbols = _extract_symbol_names(dump_result)
                _log_once(
                    "flir-dump-result",
                    f"dump() string len={len(dump_result)} symbols={symbols[:24]} text={_summarize_text(dump_result)}",
                )
            else:
                _log_once("flir-dump-result", f"dump() returned type={type(dump_result).__name__}")
        except Exception as exc:
            _log_once("flir-dump-result", f"dump() failed: {type(exc).__name__}: {exc}")


def custom_kernel(data: input_t) -> output_t:
    a, _, _, b_shuffle, _ = data

    _probe_flir_ir()

    output_shape = (a.shape[0], b_shuffle.shape[0])
    return torch.zeros(output_shape, dtype=torch.bfloat16, device=a.device)