"""MoE dispatch probe - runs diagnostics on AITER heuristic dispatch"""
import os
import sys
os.environ['AITER_LOG_MORE'] = '1'
import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_block_size_M, get_ksplit, nextPow2, get_padded_M, fused_moe
from aiter.jit.core import AITER_CONFIGS, AITER_CONFIG_FMOE
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
    
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    M = topk_ids.shape[0]
    
    # Print diagnostics
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DISPATCH DIAGNOSTICS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"GFX: {get_gfx()}, CU: {get_cu_num()}", file=sys.stderr)
    print(f"M={M}, topk={topk_ids.shape[1]}", file=sys.stderr)
    print(f"hidden_states: {hidden_states.shape} {hidden_states.dtype}", file=sys.stderr)
    print(f"gate_up_weight: {gate_up_weight_shuffled.shape} {gate_up_weight_shuffled.dtype}", file=sys.stderr)
    print(f"down_weight: {down_weight_shuffled.shape} {down_weight_shuffled.dtype}", file=sys.stderr)
    print(f"config: {config}", file=sys.stderr)
    
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    
    # Get inter_dim from weight shapes
    from aiter.fused_moe import get_inter_dim
    _, model_dim, inter_dim = get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    print(f"E={E}, model_dim={model_dim}, inter_dim={inter_dim}, topk={topk}", file=sys.stderr)
    
    # Check padded M  
    padded_M = get_padded_M(M)
    print(f"padded_M={padded_M}", file=sys.stderr)
    
    # Check block_size_M heuristic
    bsm = get_block_size_M(padded_M, topk, E, inter_dim)
    print(f"get_block_size_M(padded_M={padded_M}, topk={topk}, E={E}, inter_dim={inter_dim}) = {bsm}", file=sys.stderr)
    
    # Check ksplit
    ks = get_ksplit(padded_M, topk, E, inter_dim, model_dim)
    print(f"get_ksplit(padded_M={padded_M}, topk={topk}, E={E}, inter_dim={inter_dim}, model_dim={model_dim}) = {ks}", file=sys.stderr)
    
    # Check tune file
    tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    print(f"tune_file: {tune_file}", file=sys.stderr)
    print(f"tune_file exists: {os.path.exists(tune_file)}", file=sys.stderr)
    
    if os.path.exists(tune_file):
        import pandas as pd
        df = pd.read_csv(tune_file)
        print(f"tune_file rows: {len(df)}", file=sys.stderr)
        
        # Check if our shape is in the CSV
        cu_num = get_cu_num()
        mask = (df['cu_num'] == cu_num) if 'cu_num' in df.columns else pd.Series([False]*len(df))
        mask2 = df['q_type'].astype(str).str.contains('per_1x32') if 'q_type' in df.columns else pd.Series([False]*len(df))
        both = mask & mask2
        print(f"Entries matching cu_num={cu_num} AND per_1x32: {both.sum()}", file=sys.stderr)
        if both.any():
            cols = ['cu_num','token','model_dim','inter_dim','expert','topk','block_m','ksplit','kernelName1','kernelName2']
            available_cols = [c for c in cols if c in df.columns]
            print(df[both][available_cols].to_string(), file=sys.stderr)
    
    # Check 1-stage dict
    from aiter.fused_moe import fused_moe_1stage_dict
    gfx = get_gfx()
    print(f"\n1-stage dict entries for {gfx} + per_1x32:", file=sys.stderr)
    if gfx in fused_moe_1stage_dict:
        for k, v in fused_moe_1stage_dict[gfx].items():
            if 'per_1x32' in str(k):
                print(f"  {k}", file=sys.stderr)
    
    # Compute the actual dispatch
    print(f"\nDoing actual fused_moe call...", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
