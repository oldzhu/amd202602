"""
MoE-MXFP4: Pre-allocated buffers bypass.
Calls fused_moe_2stages directly with cached sorting/intermediate buffers.
"""

import sys
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import (
    fused_moe,
    fused_moe_2stages,
    moe_sorting,
    get_2stage_cfgs,
    get_block_size_M,
)

_SORT_BUF_CACHE: dict[tuple, tuple] = {}
_INTER_BUF_CACHE: dict[tuple, torch.Tensor] = {}
_OUT_BUF_CACHE: dict[tuple, torch.Tensor] = {}


def _get_sort_buffers(M, topk, n_experts, device):
    key = (M, topk, n_experts)
    cached = _SORT_BUF_CACHE.get(key)
    if cached is None:
        total = M * topk
        sorted_token_ids = torch.empty(
            (total + n_experts * 32,), dtype=torch.int32, device=device
        )
        sorted_weights = torch.empty(
            (total + n_experts * 32,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.empty(
            (n_experts,), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty(
            (n_experts,), dtype=torch.int32, device=device
        )
        moe_buf = torch.empty(
            (n_experts + 1,), dtype=torch.int32, device=device
        )
        cached = (sorted_token_ids, sorted_weights, sorted_expert_ids,
                  num_valid_ids, moe_buf)
        _SORT_BUF_CACHE[key] = cached
    return cached


def _get_inter_buffer(M, topk, inter_size, device):
    key = (M, topk, inter_size)
    cached = _INTER_BUF_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (M * topk, inter_size), dtype=torch.bfloat16, device=device
        )
        _INTER_BUF_CACHE[key] = cached
    return cached


def _get_out_buffer(M, d_hidden, device):
    key = (M, d_hidden)
    cached = _OUT_BUF_CACHE.get(key)
    if cached is None:
        cached = torch.empty(
            (M, d_hidden), dtype=torch.bfloat16, device=device
        )
        _OUT_BUF_CACHE[key] = cached
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
