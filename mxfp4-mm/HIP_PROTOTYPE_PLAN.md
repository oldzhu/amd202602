# MXFP4-MM HIP Prototype Plan

[中文版本](HIP_PROTOTYPE_PLAN.zh-CN.md)

## Goal

Build a fresh low-level `mxfp4-mm` experiment that uses a custom HIP/C++ kernel as the compute path while preserving the honest task contract.

This is a research plan, not an accepted optimization yet.

## Current Stage

The B-side weight-layout question is now closed:

1. Upstream AITER `shuffle_weight(..., layout=(16, 16))` makes the `B_shuffle` permutation explicit.
2. `submission_optimized.py` now validates that permutation directly inside the custom HIP probe rather than relying only on raw `B_q`.
3. Remote `test` still passed 4/4 with zero error after the switch.
4. Remote `benchmark` regressed slightly further to about `246.233 us` geometric mean versus the already-slow raw-`B_q` steady-state path at about `236.996 us`.

This means the remaining blocker is no longer B-side layout uncertainty. The compute structure of the current custom HIP kernel is itself the dominant problem, so further progress must come from changing the inner-loop and work-partition strategy rather than from more shuffle archaeology.

One immediate structural attempt has already been rejected:

1. A cooperative shared-memory rewrite that decoded A once per row and B once per column inside the `8x8` probe kept correctness intact.
2. But it regressed remote benchmark badly to about `361.759 us` geometric mean versus the simpler correct `B_shuffle` path at about `246.233 us`.
3. The extra synchronization cost outweighed any decode reuse benefit.

This means the next structural step should avoid fine-grained per-byte synchronization. Any future rewrite should change work partition more coarsely, not add more barriers inside the current probe loop.

Steady-state performance is now revalidated:

1. The earlier Stage-1 through Stage-8 benchmark numbers were distorted by a hidden one-shot `_PROBE_LAUNCHED` gate in `submission_optimized.py`.
2. That gate meant the inline-HIP path only ran on the first call in each process, so most benchmark iterations were effectively measuring the trusted ASM fallback path instead of the custom HIP work.
3. After removing the gate, remote `test` still passed 4/4 with zero error.
4. The real steady-state remote `benchmark` regressed massively to about `236.996 us` geometric mean.

This means the current inline-HIP experiment is only a correctness and contract-research artifact. Its performance is not remotely competitive, and any further work must focus on changing the actual compute structure rather than reading more into the old near-baseline probe numbers.

Stage 1 is validated:

1. `submission_optimized.py` successfully used `torch.utils.cpp_extension.load_inline(...)` on the remote MI355X runner.
2. The remote runner compiled the tiny HIP probe with `hipcc`.
3. The probe launched once and the submission still passed remote `test` 4/4.
4. The benchmark result was `22.679 us`, effectively flat versus the simplified ASM benchmark (`22.668 us`).

This means the path is technically open, but still lacks a custom GEMM implementation that can justify the extra machinery.

Stage 2 is also validated:

1. `submission_optimized.py` now passes a custom HIP entry point the honest packed task tensors through raw `uint8` views: `A_q`, `A_scale_sh`, `B_shuffle`, and `B_scale_sh`.
2. The remote runner again compiled the inline HIP code with `hipcc` and executed the custom kernel.
3. Remote `test` still passed 4/4 with zero error because the real output still falls back to the trusted `aiter.gemm_a4w4` path.
4. The Stage-2 benchmark landed at `22.856 us`, slower than both the trusted ASM baseline and the Stage-1 scratch-tensor probe.

This means the two biggest feasibility questions are now closed:

1. The runner accepts runtime HIP compilation and execution.
2. A custom HIP entry point can consume the real packed MXFP4 task contract directly.

The remaining problem is no longer runner access or dtype boundaries; it is implementing real GEMM math inside that custom HIP path.

Stage 6 is now validated semantically:

1. `submission_optimized.py` overwrites the leading `2x2` tile of the real bf16 GEMM output tensor, not just a side scratch buffer.
2. The overwrite only became correct after switching the custom probe from linearly reading `B_shuffle` to using raw `B_q` plus unshuffled E8M0 scales.
3. Remote `test` passed 4/4 with zero error after that change, which closes the main B-side semantic mismatch for the first live output tile.
4. The matching benchmark was still slow at about `28.588 us`, so the path is correctness-validated but not performance-competitive.

This means the next stage is no longer about the first-tile contract interpretation. It is about expanding the amount of real GEMM work performed by the custom HIP kernel while preserving the now-validated raw-`B_q` interpretation.

Stage 7 is now lightly validated:

1. The live-output overwrite expanded from `2x2` to `4x4` without breaking correctness.
2. Remote `test` still passed 4/4 with zero error.
3. The benchmark improved slightly to about `28.485 us`, so enlarging the correct tile did not explode cost.

This means the next reasonable stage is a larger correct partial tile or a first coarse-grained work partition, not more semantic debugging of the raw-vs-shuffled B-side boundary.

Stage 8 is now lightly validated:

1. The live-output overwrite expanded from `4x4` to `8x8` without breaking correctness.
2. Remote `test` still passed 4/4 with zero error.
3. The first `8x8` benchmark attempt exposed only a shape guard bug on the `m=4` case, not a semantic issue.
4. After removing that guard, the benchmark completed at about `28.489 us`, effectively flat versus the `4x4` overwrite at about `28.485 us`.

This means the semantic base is now stable across a larger live tile, and the next reasonable step is either a coarse-grained output partition or a more compute-efficient inner loop. Merely growing the tile further is no longer likely to change the performance story by itself.

## Why This Direction

- Pure torch implementations are useful as semantic references, but they are too slow to be competitive.
- The current trusted path in `submission_clean.py` is already a thin ASM-backed wrapper around `aiter.gemm_a4w4`.
- Earlier Triton experiments either silently fell back, failed correctness once forced to execute for real, or lost badly on benchmark.
- The upstream repo includes AMD inline HIP examples via `torch.utils.cpp_extension.load_inline`, so the submission harness is not obviously blocking a custom HIP/C++ kernel path.

## Upstream Reference Anchors

The two most relevant upstream code references for this plan are:

1. `problems/amd/fp8-mm/template-hip.py`
	- Shows a minimal ROCm inline-HIP pattern with:
	  - `os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'`
	  - `os.environ['CXX'] = 'clang++'`
	  - `torch.utils.cpp_extension.load_inline(...)`
	  - `extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"]`
	- Launches a HIP kernel from Python through a tiny wrapper function.
2. `problems/amd_distributed/rocshmem_example.py`
	- Shows how upstream handles ROCm include paths and link flags using `load_inline(...)` plus `extra_cflags` and `extra_ldflags`.

These references justify Stage 1 as a runtime-compiled HIP experiment rather than a pure thought exercise.

## Inputs To Preserve

The task tuple for `mxfp4-mm` is:

```python
A, B, B_q, B_shuffle, B_scale_sh = data
```

The current trusted path does this:

1. Quantize `A` with `dynamic_mxfp4_quant`
2. Shuffle `A` scales with `e8m0_shuffle`
3. Call `aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, bpreshuffle=True)`

Any fresh low-level prototype should start from this same honest contract.

## Non-Goals

- Do not precompute outputs.
- Do not bypass the real GEMM semantics.
- Do not dequantize everything to bf16 or fp32 and call `torch.mm` as the final submission path.
- Do not assume the task's ASM-ready tensors are valid Triton-ready tensors.

## Most Important Known Constraints

### 1. `B_shuffle` and `B_scale_sh` are valid task inputs

Using them directly is consistent with both the task contract and the current trusted submission.

### 2. Triton layout mismatch was the real failure mode

Earlier direct Triton retries failed because the task-provided tensors match the ASM contract, while Triton preshuffle kernels expect a different weight/scale layout.

### 3. Direct ASM override is currently blocked by the runner

The low-level `aiter.gemm_a4w4_asm(..., kernelName, log2_k_split)` route triggered the remote runner error about work on another stream.

### 4. Inline HIP is plausible in principle

The upstream AMD `fp8-mm` template and ROCshmem example both show `torch.utils.cpp_extension.load_inline` on ROCm.

### 5. The upstream HIP template is only a build-pattern reference

`problems/amd/fp8-mm/template-hip.py` is not a semantic template for `mxfp4-mm`.

It assumes:

- FP8 inputs for both operands
- float scale tensors
- a simple block-scale contract

Our `mxfp4-mm` prototype must instead preserve:

- packed MXFP4 inputs (`fp4x2` views)
- E8M0 scales
- the task-provided `B_shuffle` and `B_scale_sh` contract

So the upstream HIP template is useful for runtime compilation and launch structure, not for the actual kernel math.

## Recommended First Prototype

### Boundary

Keep A-side quantization in Python exactly as today:

1. `A_q = dynamic_mxfp4_quant(A)`
2. `A_scale_sh = e8m0_shuffle(...)`

Then pass:

- quantized `A_q`
- shuffled `A_scale_sh`
- task-provided `B_shuffle`
- task-provided `B_scale_sh`
- preallocated bf16 output

into a custom HIP/C++ entry point.

### Why This Boundary

- It avoids replacing the already-correct activation quant path too early.
- It keeps the prototype focused on the compute kernel.
- It respects the same input contract that the trusted ASM path already proves is valid.

### Initial Kernel Scope

Start with a correctness-first kernel, not a heroic optimized kernel:

1. One kernel for `C = A @ B.T`
2. bf16 output
3. Handle only the real competition shapes
4. Accept the packed FP4 inputs and E8M0 scales directly
5. Validate numerics first

The first milestone is simply: runner accepts the custom HIP path, tests pass, and the kernel really executes.

## Required Semantic Reference

Use the torch MXFP4 reconstruction from reference-kernels PR `#114` as the math oracle for:

1. FP4 value interpretation
2. E8M0 scale decoding
3. per-32-element scale grouping
4. output reconstruction expectations

That reference is useful for correctness, not for performance.

## Mapping The Upstream HIP Pattern To `mxfp4-mm`

The first prototype should copy the *host-side structure* of `problems/amd/fp8-mm/template-hip.py` while replacing the *kernel contract*.

### Reuse From The Upstream HIP Pattern

1. Environment setup for ROCm compilation
2. `load_inline(...)` as the runtime build entry point
3. A tiny Python wrapper that caches the compiled module
4. A single exported function callable from `custom_kernel`

### Replace For `mxfp4-mm`

1. Replace FP8 operand types with the packed MXFP4 / E8M0 task contract
2. Replace the block-scale FP8 math with MXFP4 decode-and-accumulate semantics
3. Replace the simple `[m, k] / [n, k]` scale indexing logic with the real shuffled-scale interpretation required by `B_scale_sh`
4. Write bf16 output directly into a preallocated output tensor or return a bf16 tensor with the exact task shape

### Stage-1 Kernel Goal

The first HIP kernel does not need to be fast. It only needs to prove:

1. Runtime compilation succeeds on the remote runner
2. The custom entry point executes for real
3. The kernel can consume the honest task tensors without layout cheating
4. Correctness can be matched against the trusted path

## Risks

### Runner acceptance risk

The harness may compile and load inline HIP successfully, but still reject or penalize the path in ways not visible locally.

### Contract risk

The packed layout for `B_shuffle` and shuffled scale layout for `B_scale_sh` must be consumed exactly as provided.

### Performance risk

A correctness-first HIP kernel may still be slower than the current ASM-backed baseline.

## Suggested Prototype Sequence

1. Build a tiny `load_inline` ROCm module that compiles and launches on a toy tensor.
2. Copy the host-side compilation pattern from upstream `fp8-mm/template-hip.py` into `submission_optimized.py`.
3. Replace toy math with a correctness-first MXFP4 GEMM that consumes the real task-facing tensors.
4. Use the PR `#114` torch reconstruction as the oracle for FP4 value decode and E8M0 scale interpretation.
5. Run remote `test` only.
6. If it passes and the runner accepts it, run remote `benchmark`.
7. Only then decide whether deeper scheduling or fusion work is justified.

## Success Criteria For Stage 1

- Compiles on the remote runner
- Passes correctness tests
- Emits evidence that the custom kernel actually executed
- Does not trigger the runner-side policy error seen with direct ASM override

If any of those fail, the path should be treated as blocked before spending more effort on performance tuning.

## Next Step After Stage 2

Stage 3 has now been validated as a research milestone.

What Stage 3 proved:

1. The custom HIP kernel body can decode E8M0 scales from the real task-facing tensors.
2. The custom HIP kernel body can decode both FP4 lanes from the packed task bytes.
3. The custom HIP kernel can perform a small dot-product-style decode-and-accumulate sample over `A_q`, `A_scale_sh`, `B_shuffle`, and `B_scale_sh` without breaking runner acceptance.
4. Remote `test` still passed 4/4, and remote `benchmark` improved from the Stage-2 byte-touch probe (`22.856 us`) to `22.772 us`.

This is still slower than the trusted ASM baseline, but it closes the decode-logic milestone.

The completed Stage-3 target was:

1. Keep Python-side A quantization exactly as today.
2. Add a custom HIP entry point that reads `A_q`, `A_scale_sh`, `B_shuffle`, and `B_scale_sh`.
3. Replace the current byte-touch probe with correctness-first decode-and-accumulate logic, even if it is slower than ASM.
4. Use remote `test` to validate real task-contract math before caring about speed.

The next unresolved milestone is no longer runner acceptance, contract access, or basic packed decode logic.

## Next Step After Stage 3

Stage 4 has now been validated as a research milestone.

What Stage 4 proved:

1. The custom HIP path can move from sampled decode-and-accumulate logic to deterministic partial-output math.
2. A row/column-aware `2x2` partial-output probe over the real packed task tensors still compiles and executes on the remote runner.
3. The original Stage-4 attempt exposed a real integration hazard: viewing packed tensors as `uint8` can distort the logical row/column metadata, so the HIP entry point must derive raw-buffer shapes from the original tensor geometry.
4. After fixing that boundary bug, remote `test` still passed 4/4 and remote `benchmark` completed at `22.899 us` geometric mean.

This is slower than both the trusted ASM baseline and the Stage-3 probe, so Stage 4 is still not a performance candidate.

## Next Step After Stage 4

Stage 5 has now been validated as a research milestone.

What Stage 5 proved:

1. The custom HIP path can write deterministic partial results into a real bf16 `[M, N]` output tensor rather than a tiny probe scratch buffer.
2. The runner still accepts the inline-HIP path when the custom kernel targets the true submission output layout.
3. Remote `test` still passed 4/4, and remote `benchmark` completed at `22.787 us` geometric mean.
4. Stage 5 is still slower than the trusted ASM baseline, but it improved over the Stage-4 scratch probe.

This means the output-layout milestone is now closed.

## Next Step After Stage 5

Stage 6 is now validated as a semantics milestone.

What corrected Stage 6 proved:

1. The custom HIP path can overwrite the leading `2x2` tile in the true bf16 output tensor and still pass remote correctness.
2. The previous mismatch was caused by treating `B_shuffle` like raw MXFP4 data inside the custom tile.
3. Using raw `B_q` plus unshuffled E8M0 scales fixes the tile semantics.
4. The custom path is still far slower than ASM because it only computes a tiny tile while paying all the runtime compilation and probe overhead.

This changes the next priority again.

Recommended next target after corrected Stage 6:

1. Keep Python-side A quantization exactly as today.
2. Keep the trusted AITER fallback until the custom HIP path computes a meaningfully larger portion of the real GEMM.
3. Stop spending cycles on contract semantics for the leading tile; that part is now good enough.
4. Move to a Stage-7 partial-compute kernel that writes a larger real output region while still using raw `B_q` and unshuffled scales inside the custom path.

The next key research milestone is now: can the custom HIP kernel expand from a correct `2x2` tile to a materially larger partial GEMM region without blowing up correctness or overhead.