"""
MoE-MXFP4: Direct CKTile bypass with full buffer pre-allocation.
Calls moe_sorting_fwd + moe_cktile2stages_gemm1 + quant + gemm2 directly,
skipping all fused_moe wrapper overhead.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter
from aiter.ops.moe_sorting import moe_sorting_fwd
from aiter.ops.moe_op import moe_cktile2stages_gemm1, moe_cktile2stages_gemm2
from aiter.triton_impl.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

_CACHE = {}
TOPK = 8


def _get_cached(M, E, model_dim, inter_dim, device):
    key = (M, E, model_dim, inter_dim)
    c = _CACHE.get(key)
    if c is not None:
        return c

    # Determine block_m per AITER heuristic for fp4x2
    token = M * TOPK
    if token < 2048:
        block_m = 16
    elif token < 16384:
        block_m = 32
    else:
        block_m = 64

    # Sorting buffer sizes
    max_num_tokens_padded = M * TOPK + E * block_m - TOPK
    max_num_m_blocks = (max_num_tokens_padded + block_m - 1) // block_m

    sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)

    # Stage1 intermediate: (token_num, topk, inter_dim*2) in bf16
    stage1_out = torch.empty((M, TOPK, inter_dim * 2), dtype=torch.bfloat16, device=device)

    c = {
        "block_m": block_m,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "moe_buf": moe_buf,
        "stage1_out": stage1_out,
    }
    _CACHE[key] = c
    return c


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

    M = topk_ids.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    model_dim = hidden_states.shape[1]
    inter_dim = config["d_expert_pad"]
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    c = _get_cached(M, E, model_dim, inter_dim, hidden_states.device)
    block_m = c["block_m"]

    # Step 1: Sort tokens to experts
    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        c["sorted_token_ids"],
        c["sorted_weights"],
        c["sorted_expert_ids"],
        c["num_valid_ids"],
        c["moe_buf"],
        E,
        block_m,
    )

    # Step 2: Stage1 GEMM (gate_up projection with SiLU activation)
    moe_cktile2stages_gemm1(
        hidden_states,                    # XQ [M, model_dim]
        gate_up_weight_shuffled,          # WQ [E, 2*inter_dim, model_dim//2] fp4x2
        c["stage1_out"],                  # Y [M, topk, 2*inter_dim] pre-allocated
        c["sorted_token_ids"],
        c["sorted_expert_ids"],
        c["num_valid_ids"],
        TOPK,
        n_padded_zeros=0,
        k_padded_zeros=hidden_pad,
        topk_weight=None,
        x_scale=None,
        w_scale=gate_up_weight_scale_shuffled,
        exp_bias=None,
        activation=0,                     # 0=Silu
        block_m=block_m,
        split_k=1,
        kernel_name="",
    )

    # Step 3: Inter-stage quantization (MXFP4)
    a2_fp4, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        c["stage1_out"],
        c["sorted_token_ids"],
        c["num_valid_ids"],
        M,
        TOPK,
        block_size=32,
    )

    # Step 4: Stage2 GEMM (down projection with expert weighting)
    moe_cktile2stages_gemm2(
        a2_fp4,                           # XQ (quantized intermediate)
        down_weight_shuffled,             # WQ [E, model_dim, inter_dim//2] fp4x2
        c["moe_buf"],                     # Y [M, model_dim] output
        c["sorted_token_ids"],
        c["sorted_expert_ids"],
        c["num_valid_ids"],
        TOPK,
        n_padded_zeros=0,
        k_padded_zeros=intermediate_pad,
        topk_weight=c["sorted_weights"],
        x_scale=a2_scale,
        w_scale=down_weight_scale_shuffled,
        exp_bias=None,
        activation=0,
        block_m=block_m,
        split_k=1,
        kernel_name="",
    )

    return c["moe_buf"]
