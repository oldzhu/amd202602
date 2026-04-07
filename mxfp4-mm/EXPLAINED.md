# 📖 MXFP4-MM Implementation Explained

[中文版本](EXPLAINED.zh-CN.md)

## 🎯 What This Code Does

```
INPUT:
  A: [M, K] bf16    ← Your activation matrix (e.g., neural network activations)
  B: [N, K] bf16    ← Weight matrix (already quantized to MXFP4 for you)

OUTPUT:
  C: [M, N] bf16    ← Result of A @ B.T (matrix multiplication)
```

---

## 🔢 The Math

### Standard Matrix Multiplication:
```
C[i,j] = Σ A[i,k] × B[j,k]    for k = 0 to K-1

Shape: [M, K] × [K, N] → [M, N]
       (Note: B.T gives us [K, N])
```

### With MXFP4 Quantization:
```
Instead of storing B as bf16 (16 bits per number),
we store it as MXFP4 (4 bits per number + scale)

C[i,j] = Σ (A[i,k] × scale_A) × (B_q[j,k] × scale_B)

The scales are shared per block of 32 values
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  A [M, K] bf16                      B [N, K] bf16                   │
│  ┌─────────────────┐                ┌─────────────────┐             │
│  │ Activation      │                │ Weight Matrix   │             │
│  │ (from previous  │                │ (pre-quantized  │             │
│  │  layer)         │                │  to MXFP4)      │             │
│  └─────────────────┘                └─────────────────┘             │
│         │                                    │                       │
│         │                                    │                       │
│         ▼                                    ▼                       │
│  ┌─────────────────┐                ┌─────────────────┐             │
│  │ Quantize to     │                │ Already done!   │             │
│  │ MXFP4           │                │ (B_shuffle,     │             │
│  │                 │                │  B_scale_sh     │             │
│  └─────────────────┘                │  provided)      │             │
│         │                           └─────────────────┘             │
│         ▼                                    │                       │
│  A_q [M, K//2] fp4x2                          │                       │
│  A_scale [M, K//32] e8m0                      │                       │
│         │                                    │                       │
│         └──────────────┬─────────────────────┘                       │
│                        │                                             │
│                        ▼                                             │
│              ┌─────────────────────┐                                │
│              │   gemm_a4w4 kernel  │                                │
│              │                     │                                │
│              │  Fast 4-bit matmul  │                                │
│              │  on AMD MI300 GPU   │                                │
│              └─────────────────────┘                                │
│                        │                                             │
│                        ▼                                             │
│              ┌─────────────────────┐                                │
│              │  C [M, N] bf16      │                                │
│              │  Result matrix      │                                │
│              └─────────────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Key Functions Explained

### 1. `quantize_to_mxfp4(tensor, shuffle=True)`

**Purpose:** Convert a bf16 tensor to MXFP4 format

**What it does:**
```python
# Input:  [M, K] bf16 tensor (K must be divisible by 32)
# Output: (fp4_data, scale)

# Example with K=64:
# Input tensor has 64 values per row
# After quantization:
#   - fp4_data: 32 bytes per row (64 values ÷ 2 = 32 packed bytes)
#   - scale: 2 values per row (64 ÷ 32 = 2 scale factors)

# Each scale covers 32 consecutive values
# Scale[0] covers tensor[0:32]
# Scale[1] covers tensor[32:64]
```

### 2. `aiter.gemm_a4w4(...)`

**Purpose:** Compute matrix multiplication with 4-bit quantized inputs

**Parameters explained:**
```python
aiter.gemm_a4w4(
    A_q,              # [M, K//2] Quantized activation (4-bit packed)
    B_shuffle,        # [N, K//2] Quantized weight (pre-shuffled for speed)
    A_scale_sh,       # [M, K//32] Activation scales (shuffled)
    B_scale_sh,       # [N, K//32] Weight scales (shuffled)
    dtype=dtypes.bf16,# Output data type
    bpreshuffle=True, # B is already shuffled (don't shuffle again)
)
```

**Why "shuffled"?**
```
Standard memory layout:  [0, 1, 2, 3, 4, 5, 6, 7, ...]

AMD GPU optimal layout (16×16 tiles):
┌────────────┐
│ 0  1  4  5 │  Values are rearranged so that
│ 2  3  6  7 │  each 16×16 tile is contiguous
│ 8  9 12 13 │  in memory. This enables faster
│10 11 14 15 │  matrix operations.
└────────────┘

The shuffling is pre-computed for B (B_shuffle).
We shuffle A during quantization (shuffle=True).
```

---

## 📐 MXFP4 Format Deep Dive

### FP4 Value Representation (E2M1)

```
4 bits per value:
┌───┬───────┬───────┐
│ S │ E1 E0 │ M0    │
│ 1 │   2   │   1   │   = 4 bits total
└───┴───────┴───────┘
S = Sign bit (0 = positive, 1 = negative)
E = Exponent (2 bits, values 0-3)
M = Mantissa (1 bit, values 0 or 0.5)

Possible values:
┌─────────┬─────────┬──────────┬─────────────────────┐
│ Binary  │ Sign    │ Exponent │ Value               │
├─────────┼─────────┼──────────┼─────────────────────┤
│ 0000    │ +       │ 0        │ 0                   │
│ 0001    │ +       │ 0        │ 0.5                 │
│ 0010    │ +       │ 1        │ 1                   │
│ 0011    │ +       │ 1        │ 1.5                 │
│ 0100    │ +       │ 2        │ 2                   │
│ 0101    │ +       │ 2        │ 3                   │
│ 0110    │ +       │ 3        │ 4                   │
│ 0111    │ +       │ 3        │ 6                   │
│ 1xxx    │ -       │ same     │ negative of above   │
└─────────┴─────────┴──────────┴─────────────────────┘
```

### E8M0 Scale Format

```
8 bits per scale:
┌───────────────────────┐
│ E7 E6 E5 E4 E3 E2 E1 E0│  = Exponent only (no mantissa)
└───────────────────────┘

Value = 2^(exponent - 127)  (like IEEE 754)

Example:
  E8M0 value = 0x80 (128 in decimal)
  Scale = 2^(128-127) = 2^1 = 2.0

  E8M0 value = 0x81 (129 in decimal)
  Scale = 2^(129-127) = 2^2 = 4.0
```

### Putting It Together

```
Block of 32 values with one scale:

Original values (bf16): [1.2, 2.5, 3.1, ..., 5.8]  (32 values)
Max absolute value: 5.8

Quantization:
1. Find scale: scale = 2^ceil(log2(5.8/6)) = 2.0
               (6 is max FP4 value)
2. Divide by scale: [0.6, 1.25, 1.55, ..., 2.9]
3. Round to FP4: [0.5, 1.0, 1.5, ..., 3.0]  (closest FP4 values)
4. Pack: [0.5, 1.0] → 1 byte, [1.5, 2.0] → 1 byte, ...

Dequantization:
  value = FP4_value × scale
  [0.5, 1.0, 1.5, ...] × 2.0 = [1.0, 2.0, 3.0, ...]

Error:
  Original: [1.2, 2.5, 3.1, ...]
  Dequant:  [1.0, 2.0, 3.0, ...]
  Error:    ~4-8% (acceptable for neural networks!)
```

---

## ⚡ Why This Is Fast

### 1. Memory Bandwidth
```
bf16: 16 bits × (M×K + K×N) bytes transferred
MXFP4: 4 bits × (M×K + K×N) / 2 + scales

For M=N=K=4096:
  bf16:    402 MB transferred
  MXFP4:   ~105 MB transferred
  Speedup: ~4x less memory traffic
```

### 2. Compute Density
```
AMD MI300 has specialized MFMA (Matrix Fused Multiply-Add) units:

BF16 MFMA:  16×16×16 per instruction
FP4 MFMA:   32×32×32 per instruction (4x more work!)

Same hardware, 4x more operations per cycle
```

### 3. Kernel Fusion
```
Naive approach:
  dequant(A) → dequant(B) → matmul → output
  (3 separate kernel launches, lots of memory traffic)

gemm_a4w4:
  matmul(dequant_inline(A), dequant_inline(B)) → output
  (1 kernel launch, dequantization is fused)

Fusion eliminates intermediate memory writes!
```

---

## 🎓 Key Takeaways

1. **Quantization trades precision for speed** - Small accuracy loss for 4x speedup
2. **Block scaling maintains accuracy** - Each 32-value block has its own scale
3. **Memory layout matters** - Shuffling enables optimal GPU memory access
4. **AITER does the hard work** - You just call `gemm_a4w4` with the right inputs

---

## ❓ Common Questions

**Q: Why do we quantize A but not B?**
**A:** B is already pre-quantized and shuffled (provided as B_shuffle, B_scale_sh). You just need to quantize A at runtime.

**Q: What if K is not divisible by 32?**
**A:** The problem guarantees K is divisible by 64 (which includes 32).

**Q: What is the expected accuracy?**
**A:** rtol=0.01, atol=0.01 (1% relative tolerance). MXFP4 easily meets this.

**Q: Can I use bf16 instead?**
**A:** Yes, but it will be ~4x slower. The quantization overhead is minimal compared to the speedup.
