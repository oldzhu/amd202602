"""Manual MM test: pure FlyDSL launch to reproduce stream rejection."""

import sys

import torch
from task import input_t, output_t

import flydsl.compiler as flyc
import flydsl.expr as fx


_LOG_KEYS: set[str] = set()
_FLYDSL_PROBE_ATTEMPTED = False


def _log_once(key: str, message: str) -> None:
    if key in _LOG_KEYS:
        return
    _LOG_KEYS.add(key)
    print(f"[mxfp4-mm flydsl manual] {message}", file=sys.stderr)


@flyc.kernel
def _noop_kernel():
    pass


@flyc.jit
def _noop_launch():
    _noop_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


def _probe_flydsl_launch() -> None:
    global _FLYDSL_PROBE_ATTEMPTED
    if _FLYDSL_PROBE_ATTEMPTED:
        return
    _FLYDSL_PROBE_ATTEMPTED = True

    _log_once("flydsl-launch-start", "launching minimal FlyDSL noop kernel")
    _noop_launch()
    _log_once("flydsl-launch-done", "minimal FlyDSL noop kernel launch returned")


def custom_kernel(data: input_t) -> output_t:
    a, _, _, b_shuffle, _ = data

    _probe_flydsl_launch()

    output_shape = (a.shape[0], b_shuffle.shape[0])
    return torch.zeros(output_shape, dtype=torch.bfloat16, device=a.device)