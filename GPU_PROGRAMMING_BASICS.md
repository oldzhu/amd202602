# 🔬 GPU Programming Concepts for Hackathon

[中文版本](GPU_PROGRAMMING_BASICS.zh-CN.md)

## 📚 Essential Concepts You Need to Know

This document explains the key GPU programming concepts you'll encounter in the hackathon.

---

## 1. Quantization Basics

### What is Quantization?
Quantization reduces the precision of numbers to save memory and speed up computation.

```
BF16 (16-bit):  1.23456789012345  →  1.23438  (less precise, but faster)
FP8 (8-bit):    1.23456789012345  →  1.25     (even less precise, much faster)
FP4 (4-bit):    1.23456789012345  →  1.0      (lowest precision, fastest)
```

### Why Quantize?
- **Memory**: 4-bit uses 4x less memory than 16-bit
- **Bandwidth**: Less data transfer = faster
- **Compute**: Specialized hardware for low-precision math

### Types of Quantization

#### Per-Tensor (Simplest)
```
Entire tensor shares ONE scale factor
Example: FP8 with scale = 2.5
All values: value = fp8_value * 2.5
```

#### Per-Block (MXFP4)
```
Every 32 values share ONE scale factor
Block 0: values[0:32] share scale[0]
Block 1: values[32:64] share scale[1]
...
Better accuracy than per-tensor!
```

---

## 2. FP8 Format

### Two FP8 Variants:
```
E4M3: 1 sign + 4 exponent + 3 mantissa (more range)
E5M2: 1 sign + 5 exponent + 2 mantissa (more precision for small numbers)
```

### Example:
```
Value: 1.75
E4M3: 0 0111 110 = 1.75 (exact)
```

---

## 3. MXFP4 (Microscaling FP4)

### Format:
```
E2M1: 1 sign + 2 exponent + 1 mantissa
Can represent: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

### Block Quantization:
```
Block size = 32
Each block has an E8M0 scale (power of 2)

Example:
Block values: [1.0, 2.0, 3.0, ...]  (32 values)
Max value: 6.0
Scale: 2^2 = 4.0 (E8M0 stores exponent 2)
Quantized: [0.25, 0.5, 0.75, ...] in FP4
Dequantized: [1.0, 2.0, 3.0, ...] (approximately)
```

### Why MXFP4?
- 4-bit = smallest standard format
- Block scaling = good accuracy
- Hardware support on newer GPUs

---

## 4. Memory Layout and Shuffling

### Why Shuffle?
```
Standard Layout (row-major):
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

GPU-optimal Layout (tiled):
[0, 1, 4, 5, 8, 9, ...]  # Values grouped for coalesced access
```

### 16x16 Tiling:
```
Common for AMD GPUs
Groups values into 16x16 tiles
Enables efficient MFMA (Matrix Fused Multiply-Add) operations
```

---

## 5. Attention Mechanism

### Standard Attention:
```
Q: Query   [batch, heads, seq_len, head_dim]
K: Key     [batch, heads, seq_len, head_dim]
V: Value   [batch, heads, seq_len, head_dim]

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

### Multi-Head Attention:
```
Multiple heads attend to different patterns
Output = Concat(head_1, head_2, ...) @ W_o
```

### MLA (Multi-head Latent Attention):
```
Innovation: Compress KV cache into latent space

Traditional KV: [batch, heads, seq, head_dim]
MLA KV: [batch, 1, seq, latent_dim]

Compression ratio: heads × head_dim / latent_dim
For DeepSeek-R1: 16 × 576 / 512 = 18x compression!
```

---

## 6. Mixture of Experts (MoE)

### Concept:
```
Instead of one big model, use many small "experts"
Each token is routed to a few experts
Only a fraction of the model is active per token
```

### DeepSeek-R1 MoE:
```
Total experts: 256 routed + 1 shared = 257
Active per token: 8 routed + 1 shared = 9
Activation ratio: 9/257 ≈ 3.5%

Each expert is a small FFN:
  Input: [d_hidden] → [d_expert] → [d_hidden]
  d_hidden = 7168, d_expert = 2048
```

### Expert Computation:
```
1. Router selects top-k experts for each token
2. Each expert processes: gate_proj + up_proj + down_proj
3. Outputs are weighted by router scores

gate = input @ W_gate.T      # [d_expert]
up = input @ W_up.T          # [d_expert]
intermediate = silu(gate) * up   # SwiGLU activation
output = intermediate @ W_down.T # [d_hidden]
```

---

## 7. GPU Memory Hierarchy

### Memory Levels (fastest to slowest):
```
1. Registers    - Fastest, per-thread, ~256 KB total
2. Shared Mem   - Fast, per-block, ~64 KB per block
3. L1 Cache     - Fast, per-SM, ~128 KB
4. L2 Cache     - Medium, shared, ~8 MB
5. HBM          - Slow, global, ~80+ GB
```

### Optimization Strategy:
```
1. Keep data in faster memory when possible
2. Reuse data (temporal locality)
3. Access consecutive addresses (spatial locality)
4. Minimize HBM transfers
```

---

## 8. Kernel Fusion

### Concept:
```
Instead of:
  temp = op1(input)
  output = op2(temp)

Fuse into:
  output = fused_op(input)

Benefits:
- Less memory traffic
- Better cache utilization
- Lower latency
```

### Examples in Hackathon:
```
MoE Fusion: gate + up + activation + down in one kernel
MLA Fusion: quant + attention + dequant in one kernel
GEMM Fusion: quant + matmul in one kernel
```

---

## 9. AITER Library

### What is AITER?
AMD Instinct Tensor Engine Runtime - AMD's optimized kernel library

### Key Modules:
```python
# MLA Attention
from aiter.mla import mla_decode_fwd

# MoE
from aiter.fused_moe import fused_moe

# Quantization
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

# GEMM
import aiter
output = aiter.gemm_a4w4(...)  # 4-bit activation, 4-bit weight
```

---

## 10. Common Patterns

### Dequantization Pattern:
```python
def dequantize(quantized_data, scale):
    # Step 1: Convert quantized format to float
    float_data = unpack_to_float(quantized_data)

    # Step 2: Apply scale
    return float_data * scale
```

### Attention Pattern:
```python
def attention(Q, K, V, scale):
    # Step 1: Compute attention scores
    scores = Q @ K.transpose(-2, -1) * scale

    # Step 2: Softmax
    attn = torch.softmax(scores, dim=-1)

    # Step 3: Apply to values
    return attn @ V
```

### MoE Pattern:
```python
def moe_forward(x, weights, topk_ids, topk_weights):
    output = torch.zeros_like(x)

    for i in range(batch_size):
        for k in range(top_k):
            expert_id = topk_ids[i, k]
            weight = topk_weights[i, k]

            # Expert computation
            expert_out = expert_forward(x[i], weights[expert_id])
            output[i] += weight * expert_out

    return output
```

---

## 🎯 Quick Reference

| Concept | Key Point |
|---------|-----------|
| Quantization | Lower precision = faster, less memory |
| MXFP4 | Block-32 scaling, 4-bit values |
| MLA | Compressed KV cache for attention |
| MoE | Sparse activation, many experts |
| Fusion | Combine operations, reduce memory traffic |

---

Remember: **Understanding these concepts helps you make better optimization decisions!**
