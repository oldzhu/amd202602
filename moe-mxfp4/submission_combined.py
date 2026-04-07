"""
MoE-MXFP4: Combined optimization:
1. Pre-alloc sort buffers for M<=256 (saves sort allocation overhead)
2. torch.empty caching for quant buffers (saves quant allocation overhead)
3. block_m=32 for M<=256 (optimal), fused_moe for M>256 (auto block_m)
4. Direct fused_moe_2stages for M<=256, fused_moe wrapper for M>256
"""

import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe, fused_moe_2stages
import aiter.fused_moe as _afm

# Monkey-patch for quant buffer pre-allocation
_orig_quant_sort = _afm.fused_dynamic_mxfp4_quant_moe_sort
_orig_torch_empty = torch.empty
_ALLOC_CACHE = {}
_PREALLOC_ACTIVE = [False]


def _caching_empty(*args, **kwargs):
    if not _PREALLOC_ACTIVE[0]:
        return _orig_torch_empty(*args, **kwargs)
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
    _PREALLOC_ACTIVE[0] = True
    try:
        return _orig_quant_sort(
            x, sorted_ids, num_valid_ids, token_num, topk, block_size, scaling_mode
        )
    finally:
        _PREALLOC_ACTIVE[0] = False


torch.empty = _caching_empty
_afm.fused_dynamic_mxfp4_quant_moe_sort = _prealloc_quant_sort_wrapper

# Sort buffer cache
_SORT_CACHE = {}


def _get_sort_buffers(M, topk, num_experts, model_dim, block_size, device):
    key = (M, topk, num_experts, model_dim, block_size)
    cached = _SORT_CACHE.get(key)
    if cached is not None:
        return cached
    max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)
    cached = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    _SORT_CACHE[key] = cached
    return cached


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
    topk = topk_ids.shape[1]
    num_experts = gate_up_weight_shuffled.shape[0]
    model_dim = hidden_states.shape[1]

    if M <= 256:
        # Optimized path: pre-alloc sort + direct fused_moe_2stages
        block_size_m = 32
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = \
            _get_sort_buffers(M, topk, num_experts, model_dim, block_size_m, hidden_states.device)

        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            block_size_m,
            None, None, 0,
        )

        return fused_moe_2stages(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            True,
            block_size_m,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            q_dtype_a=aiter_dtypes.fp4x2,
            q_dtype_w=gate_up_weight_shuffled.dtype,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None,
            a2_scale=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )
    else:
        # Standard path: fused_moe with auto block_m
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
            block_size_M=None,
        )
