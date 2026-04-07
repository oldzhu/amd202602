"""
MoE-MXFP4: Minimal overhead — pure AITER auto with zero Python extras.
"""

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def custom_kernel(data: input_t) -> output_t:
    return fused_moe(
        data[0],
        data[5],
        data[6],
        data[9],
        data[10],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=data[7],
        w2_scale=data[8],
        hidden_pad=data[11]["d_hidden_pad"] - data[11]["d_hidden"],
        intermediate_pad=data[11]["d_expert_pad"] - data[11]["d_expert"],
    )
