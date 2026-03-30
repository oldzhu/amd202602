# AMD x GPU MODE - E2E Model Speedrun Hackathon
## Phase 1: Qualifiers - Comprehensive Preparation Guide

---

## 📋 Event Overview

**Event Name:** AMD x GPU MODE - E2E Model Speedrun
**Sponsor:** AMD (Advanced Micro Devices, Inc.)
**Platform:** GPU MODE (gpumode.com)
**Community:** [discord.gg/gpumode](https://discord.gg/gpumode)

The hackathon focuses on pushing the boundaries of Large Language Model (LLM) inference performance on AMD GPUs through optimized kernel implementations. Participants compete to create the fastest and most accurate GPU kernels for critical LLM operations.

---

## 🎯 Phase 1: Qualifiers - Problem Sets

Phase 1 consists of **three kernel optimization problems**, all based on operations critical to **DeepSeek-R1** model inference on AMD MI355X GPUs:

### Problem 1: Mixed-MLA (Multi-head Latent Attention) Decode Kernel

**Purpose:** Optimize the attention mechanism for DeepSeek-R1's innovative MLA architecture during decode (autoregressive) phase.

**Technical Details:**
- **Architecture:** DeepSeek-R1 forward_absorb MLA path
- **Input:** 
  - Query `q`: `(total_q, 16, 576)` bfloat16 — absorbed query
  - KV Cache: Multiple formats provided:
    - `bf16`: `(total_kv, 1, 576)` — highest precision
    - `fp8`: `(kv_buffer fp8, scalar scale)` — per-tensor quantized
    - `mxfp4`: `(kv_buffer fp4x2, fp8_e8m0 scale)` — block-32 quantized
  - Pointers: `qo_indptr`, `kv_indptr` (int32)
  - Config dict with MLA parameters

- **Output:** `(total_q, num_heads, v_head_dim)` bfloat16

**Key Constants (DeepSeek-R1 MLA):**
```python
NUM_HEADS = 16
NUM_KV_HEADS = 1           # Multi-Query Attention
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 576          # 512 + 64
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
```

**Optimization Opportunities:**
- The reference uses FP8 quantization for Q and KV (a8w8 kernel), achieving ~2-3x speedup over bf16
- MXFP4 format provides even lower memory footprint
- Persistent-mode decode with metadata precomputation
- You can choose different precision trade-offs

---

### Problem 2: MoE-MXFP4 (Mixture of Experts with MXFP4 Quantization)

**Purpose:** Implement a fused MoE (Mixture of Experts) layer with MXFP4 weight quantization, optimized for DeepSeek-R1's expert architecture.

**Technical Details:**
- **Architecture:** DeepSeek-R1 MoE layer
- **Input:**
  - `hidden_states`: `[M, d_hidden]` bf16 (e.g., `[batch, 7168]`)
  - `gate_up_weight`: `[E, 2*d_expert_pad, d_hidden_pad//2]` fp4x2
  - `down_weight`: `[E, d_hidden_pad, d_expert_pad//2]` fp4x2
  - Scale tensors (e8m0 format, both raw and shuffled)
  - `topk_weights`: `[M, total_top_k]` float32
  - `topk_ids`: `[M, total_top_k]` int32
  - Config dict

- **Output:** `[M, d_hidden]` bf16

**DeepSeek-R1 MoE Parameters:**
```python
d_hidden = 7168                    # Hidden dimension
d_expert = 2048                    # Expert intermediate size (per partition)
n_routed_experts = 256             # Number of routed experts
n_shared_experts = 1               # Always-selected shared expert
top_k = 8                          # Experts per token (routed)
total_top_k = 9                    # 8 routed + 1 shared
```

**Key Constraints:**
- MXFP4 requires 256-alignment for dimensions
- Block size = 32 for MXFP4 quantization
- Weights must be shuffled with `(16, 16)` layout

**Optimization Opportunities:**
- Fused gate+up projection with shared weight
- Expert parallelism
- Optimized topk routing
- Pre-shuffled weights for faster kernel execution

---

### Problem 3: MXFP4-MM (MXFP4 Matrix Multiplication)

**Purpose:** Implement efficient MXFP4 quantized matrix multiplication: bf16 A × MXFP4 B → bf16 C.

**Technical Details:**
- **Input:**
  - `A`: `[M, K]` bfloat16 (activation)
  - `B`: `[N, K]` bfloat16 (weight, reference)
  - `B_q`: `[N, K//2]` fp4x2 (pre-quantized)
  - `B_shuffle`: Shuffled weight for aiter kernel
  - `B_scale_sh`: `[N, K//32]` e8m0 scale factors

- **Output:** `[M, N]` bfloat16

**Constraints:**
- K must be divisible by 64 (scale group 32 and fp4 pack 2)
- Uses per-1x32 block quantization (MXFP4)
- Two FP4 values packed per byte

**Quantization Format:**
- MXFP4: Block size 32, E2M1 format (no zero point)
- Scale: E8M0 format (power-of-2 scales, 8-bit exponent)

---

## 🔧 Required Dependencies

All problems use the **AITER (AMD Instinct Tensor Engine Runtime)** library:

```bash
# Install aiter (AMD's kernel library)
pip install aiter
```

Key AITER modules used:
```python
from aiter.mla import mla_decode_fwd
from aiter import dtypes, get_mla_metadata_v1, get_mla_metadata_info_v1
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32, e8m0_shuffle
```

**Custom dtypes:**
- `dtypes.fp4x2` — Packed FP4 (2x E2M1 per byte)
- `dtypes.fp8_e8m0` — E8M0 scale factor
- `dtypes.fp8` — FP8 E4M3 or E5M2

---

## 📁 Submission Structure

For each problem, you need to create a `submission.py` file with:

```python
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    """
    Your optimized kernel implementation.
    
    Args:
        data: The input tuple as specified in task.py
        
    Returns:
        Output tensor(s) matching output_t specification
    """
    # Your implementation here
    pass
```

**Evaluation Process:**
1. **Correctness Check:** Output compared against reference with configurable tolerance
2. **Performance Timing:** Measured with CUDA synchronization
3. **Multiple Test Cases:** Different batch sizes, sequence lengths, etc.

---

## 📊 Evaluation Criteria

The `eval.py` framework provides:

1. **Correctness:** `check_implementation(data, output)` must pass
   - Uses `verbose_allclose()` with configurable `rtol` and `atol`
   - MLA allows up to 5% mismatch ratio with warning

2. **Performance:** Multiple runs with statistics
   ```python
   @dataclasses.dataclass
   class Stats:
       runs: int
       mean: float    # Average latency
       std: float     # Standard deviation
       err: float     # Standard error
       best: float    # Best run
       worst: float   # Worst run
   ```

3. **Scoring:** Lower is better (faster execution time)

---

## 🛠 Getting Started

### Step 1: Set Up Environment
```bash
# Clone the reference kernels repository
git clone https://github.com/gpu-mode/reference-kernels.git
cd reference-kernels/problems/amd_202602

# Install dependencies
pip install torch aiter numpy
```

### Step 2: Understand Each Problem
```bash
# Study the reference implementations
cat mixed-mla/reference.py
cat moe-mxfp4/reference.py
cat mxfp4-mm/reference.py

# Understand input/output schemas
cat mixed-mla/task.py
cat moe-mxfp4/task.py
cat mxfp4-mm/task.py
```

### Step 3: Test Locally
```python
# Run evaluation
python eval.py --problem mixed-mla --submission submission.py
```

### Step 4: Optimize
1. Study the reference implementation carefully
2. Identify bottlenecks using profiling tools
3. Consider:
   - Memory access patterns
   - Quantization precision trade-offs
   - Kernel fusion opportunities
   - AMD-specific optimizations (MFMA, WMMA)

---

## 💡 Optimization Strategies

### For Mixed-MLA:
1. **Precision Selection:** Choose between bf16/fp8/mxfp4 for Q and KV
2. **Persistent Kernel:** Pre-compute metadata for decode
3. **Memory Layout:** Optimize KV cache layout for coalesced access
4. **Flash Attention:** Adapt flash attention principles for MLA

### For MoE-MXFP4:
1. **Expert Batching:** Batch computations across selected experts
2. **Weight Shuffling:** Pre-shuffle weights for optimal memory access
3. **Fused Operations:** Combine gate+up projections
4. **Shared Expert Handling:** Specialize for always-selected experts

### For MXFP4-MM:
1. **Dequantization Fusion:** Avoid full dequantization, compute with quantized values
2. **Block-wise Processing:** Leverage block structure for parallelism
3. **AMD MFMA:** Use matrix factorization operations if available
4. **Triton Kernels:** Consider custom Triton implementations

---

## 📚 Key Resources

1. **AITER Documentation:** [github.com/ROCm/aiter](https://github.com/ROCm/aiter)
2. **DeepSeek-R1 Paper:** Understanding MLA and MoE architecture
3. **AMD MI355X Specs:** Understand hardware capabilities
4. **GPU MODE Discord:** Community support and discussions
5. **Reference Implementations:** Study existing solutions

---

## ⚠️ Important Notes

1. **Tolerance Settings:**
   - Mixed-MLA: `rtol=0.1, atol=0.1` (10% relative tolerance)
   - MoE-MXFP4: Check reference for specific tolerances
   - MXFP4-MM: `rtol=0.01, atol=0.01` (1% relative tolerance)

2. **CUDA Synchronization:**
   ```python
   torch.cuda.synchronize()  # Before timing
   output = custom_kernel(data)
   torch.cuda.synchronize()  # After timing
   ```

3. **Data Cloning:** Test framework clones input data to ensure fairness

4. **Seed Handling:** Public test seeds combined with secret seed for final evaluation

---

## 🎯 Success Checklist

- [ ] Set up development environment with AMD ROCm and AITER
- [ ] Understand all three problem specifications
- [ ] Run reference implementations locally
- [ ] Study AITER library APIs
- [ ] Develop custom kernel for each problem
- [ ] Verify correctness matches reference within tolerance
- [ ] Profile and optimize for performance
- [ ] Submit through GPU MODE platform

---

## 📞 Getting Help

- **Discord:** [discord.gg/gpumode](https://discord.gg/gpumode)
- **GitHub Issues:** [reference-kernels repository](https://github.com/gpu-mode/reference-kernels)
- **AITER Issues:** [ROCm/aiter repository](https://github.com/ROCm/aiter)

---

*Document generated for Phase 1 preparation - AMD x GPU MODE E2E Model Speedrun Hackathon*
