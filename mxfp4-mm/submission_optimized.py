"""
===============================================================================
MXFP4-MM: Corrected Implementation for AMD GPU MODE Hackathon
===============================================================================

🐛 BUG FIX: The scales must be shuffled with e8m0_shuffle() before passing
   to gemm_a4w4. This was missing in the original submission!

📋 PROBLEM SUMMARY:
    Compute C = A @ B.T where:
    - A: [M, K] bfloat16 activation matrix
    - B: [N, K] bfloat16 weight matrix (quantized to MXFP4)
    - Output: [M, N] bfloat16

🔧 KEY FIX:
    Before (WRONG):
        A_q, A_scale_sh = dynamic_mxfp4_quant(A)
        A_q = A_q.view(dtypes.fp4x2)
        A_scale_sh = A_scale_sh.view(dtypes.fp8_e8m0)  # Not shuffled!
    
    After (CORRECT):
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale_sh = e8m0_shuffle(A_scale)  # MUST shuffle!
        A_q = A_q.view(dtypes.fp4x2)
        A_scale_sh = A_scale_sh.view(dtypes.fp8_e8m0)

⚡ WHY SHUFFLING MATTERS:
    The gemm_a4w4 kernel expects scales in a specific memory layout
    that matches the shuffled weights. Without shuffling, the scale
    values don't align with the correct weight blocks, causing
    completely wrong results.
===============================================================================
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


SCALE_GROUP_SIZE = 32


def _quant_mxfp4(x: torch.Tensor, shuffle: bool = True):
    """
    Quantize a bf16 tensor to MXFP4 format.
    
    This function:
    1. Calls dynamic_mxfp4_quant to get fp4 data and e8m0 scales
    2. Shuffles the scales using e8m0_shuffle (for gemm_a4w4 compatibility)
    3. Views the tensors as AITER's custom dtypes
    
    Args:
        x: [M, K] bfloat16 tensor (K must be divisible by 32)
        shuffle: Whether to shuffle scales (default: True)
    
    Returns:
        (fp4_data, scale_e8m0) where:
        - fp4_data: [M, K//2] packed FP4 values
        - scale_e8m0: [padded, K//32] shuffled E8M0 scales
    """
    # Step 1: Quantize to MXFP4
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    
    # Step 2: Shuffle scales for gemm_a4w4
    # This rearranges the scale values to match the shuffled weight layout
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    
    # Step 3: View as AITER dtypes
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 Matrix Multiplication: C = A @ B.T
    
    Input:
        data = (A, B, B_q, B_shuffle, B_scale_sh)
        
        A:            [M, K] bf16 - Activation matrix
        B:            [N, K] bf16 - Weight matrix (reference)
        B_q:          [N, K//2] fp4x2 - Quantized B (raw)
        B_shuffle:    [N, K//2] fp4x2 - Pre-shuffled quantized B
        B_scale_sh:   [padded, flat] e8m0 - Pre-shuffled scales for B
    
    Output:
        C: [M, N] bf16 - Result matrix
    """
    A, B, B_q, B_shuffle, B_scale_sh = data
    
    M, K = A.shape
    N = B.shape[0]
    
    # Ensure A is contiguous for optimal memory access
    A = A.contiguous()
    
    # ============================================================
    # Step 1: Quantize activation A to MXFP4
    # ============================================================
    # This is the key fix: scales MUST be shuffled!
    A_q, A_scale_sh = _quant_mxfp4(A, shuffle=True)
    
    # ============================================================
    # Step 2: Call AITER's gemm_a4w4 kernel
    # ============================================================
    output = aiter.gemm_a4w4(
        A_q,                    # [M, K//2] quantized activation
        B_shuffle,              # [N, K//2] pre-shuffled quantized weight
        A_scale_sh,             # [padded, K//32] shuffled activation scales
        B_scale_sh,             # [padded, K//32] shuffled weight scales
        dtype=dtypes.bf16,      # Output in bfloat16
        bpreshuffle=True,       # B_shuffle is already shuffled
    )
    
    # Slice to actual dimensions (output may be padded)
    return output[:M, :N]


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MXFP4-MM Corrected Implementation")
    print("=" * 60)
    print()
    print("Key fix: Added e8m0_shuffle() for activation scales")
    print()
    print("Submit via:")
    print("  popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test submission_clean.py")
