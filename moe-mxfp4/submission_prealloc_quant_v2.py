"""
MoE-MXFP4: Pre-allocate quant buffers via safe torch.empty interception.

Instead of reproducing AITER's internal Triton kernel call (fragile),
this approach temporarily patches torch.empty to return cached tensors
during the fused_dynamic_mxfp4_quant_moe_sort call. This is AITER-version-
independent and avoids all kernel parameter mismatches.

Key optimization: fused_dynamic_mxfp4_quant_moe_sort allocates 2 tensors
(fp4 output + blockscale) per call. It's called 2x per fused_moe (stage1 +
stage2). Pre-allocating saves 4 torch.empty calls per custom_kernel invocation.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _afm

# Store the original function and torch.empty
_orig_quant_sort = _afm.fused_dynamic_mxfp4_quant_moe_sort
_orig_torch_empty = torch.empty

# Cache for pre-allocated tensors (keyed by shape+dtype+device)
_ALLOC_CACHE = {}
_PREALLOC_ACTIVE = [False]


def _caching_empty(*args, **kwargs):
    """torch.empty replacement that caches allocations when active."""
    if not _PREALLOC_ACTIVE[0]:
        return _orig_torch_empty(*args, **kwargs)

    # Only cache tuple-form calls: torch.empty((shape,), dtype=..., device=...)
    if len(args) >= 1 and isinstance(args[0], tuple):
        shape = args[0]
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', None)
        dev_str = str(device) if device is not None else 'default'
        key = (shape, dtype, dev_str)
        cached = _ALLOC_CACHE.get(key)
        if cached is not None:
            return cached
        tensor = _orig_torch_empty(*args, **kwargs)
        _ALLOC_CACHE[key] = tensor
        return tensor

    return _orig_torch_empty(*args, **kwargs)


def _prealloc_quant_sort_wrapper(
    x, sorted_ids, num_valid_ids, token_num, topk, block_size=32, scaling_mode="even"
):
    """Wrapper that activates allocation caching around the original function."""
    _PREALLOC_ACTIVE[0] = True
    try:
        return _orig_quant_sort(
            x, sorted_ids, num_valid_ids, token_num, topk, block_size, scaling_mode
        )
    finally:
        _PREALLOC_ACTIVE[0] = False


# Apply patches at module load time
torch.empty = _caching_empty
_afm.fused_dynamic_mxfp4_quant_moe_sort = _prealloc_quant_sort_wrapper


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        _,
        _,
        _,
        _,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = topk_ids.shape[0]
    block_size_m = 32 if M <= 256 else None

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        block_size_M=block_size_m,
    )
