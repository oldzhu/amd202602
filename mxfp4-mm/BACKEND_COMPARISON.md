# MXFP4-MM Backend Comparison

[中文版本](BACKEND_COMPARISON.zh-CN.md)

## Purpose

This note explains why the earlier minimal FlyDSL/FLIR probes failed, why the later FLIR variants worked, what the current HIP blocker is, and what that means for extending backend experiments to the other two problems.

## Current Backend Status

| Backend | Current MM status | Remote correctness | Remote performance | Main blocker / note |
|---|---|---|---|---|
| AITER | Fully working | Yes | Competitive | Trusted baseline path |
| FLIR / FlyDSL | Working on the right module shape | Yes | Very slow so far | Execution works; current kernel path is not competitive |
| Inline HIP C++ | Compile works, launch not usable yet | No usable end-to-end path yet | Not measurable yet | Remote launch hangs even for scratch-only kernel |

## Benchmark Decomposition

The most useful current MM comparison is not only correctness, but overhead decomposition.

Measured benchmark bands on MI355X:

1. AITER-only baseline in [submission_clean.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_clean.py): about `18.9 / 33.9 / 19.6 / 19.5 / 24.6 / 23.1 us`
2. Scratch-only class-level FLIR launch in [submission_flir_min_class_level.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_class_level.py): about `796 / 806 / 857 / 819 / 797 / 776 us`
3. GEMM-signature FLIR launch wrapper in [submission_flir_min_gemm_signature.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_gemm_signature.py): about `686 / 686 / 680 / 670 / 672 / 670 us`
4. Full gist-style FLIR GEMM in [submission_flir_gist_like.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_gist_like.py): about `620-669 us`

This is the key interpretation:

The full FLIR GEMM is not slow mainly because of its math body.

The FLIR execution path itself is already costing on the order of `650-800 us`, because a minimal realistic FLIR launch wrapper plus trusted AITER output already lands in essentially the same latency band as the full FLIR GEMM.

## AITER: What Works Today

The trusted baseline is [submission_clean.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_clean.py).

Its structure is very thin:

1. Quantize A with `dynamic_mxfp4_quant(...)`.
2. Shuffle activation scales with `e8m0_shuffle(...)`.
3. Reuse task-provided `B_shuffle` and `B_scale_sh`.
4. Call `aiter.gemm_a4w4(...)`.

This path is important because almost all complexity sits in the pre-tuned backend rather than in Python or user-authored kernel code.

## FlyDSL: Two Different Failure Families

There were two separate FlyDSL failure modes, and they should not be mixed together.

### 1. README-style high-level FlyDSL failed because of harness stream policy

The public `flydsl.compiler` / `flydsl.expr` style launch path was rejected by the competition harness with `Your code contains work on another stream`.

That problem was about the launcher surface and stream behavior, not about FLIR module export.

### 2. Minimal FLIR probe failed because of module/export shape

The minimal FLIR diagnostic in [submission_flir_stream_reject.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_stream_reject.py) compiled, but earlier reduced versions did not expose a usable callable path. The key structural issue is that its FLIR entrypoints were defined inside `init_gpu_module(...)` instead of as class-level methods on the `flir.MlirModule` subclass.

That difference matters.

## Why The Earlier Minimal FLIR Did Not Work

The non-working minimal FLIR direction had these properties:

1. It used a reduced diagnostic shape rather than a realistic module boundary.
2. The crucial `@flir.kernel` and `@flir.jit` entrypoints were defined inside `init_gpu_module(...)` in earlier failing forms, not as class methods.
3. Remote introspection showed empty `_llvm_sigs` and missing/null symbol surfaces.
4. The executor existed, but the compiled object did not expose the callable boundary we expected.

In short: the minimal probe was too reduced and used the wrong FLIR registration/export shape.

## Why The Working FLIR Paths Work

The working FLIR paths are:

1. [submission_flir_gist_like.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_gist_like.py)
2. [submission_flir_min_class_level.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_class_level.py)
3. [submission_flir_min_gemm_signature.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_gemm_signature.py)

These working variants share the same core properties:

1. `@flir.kernel` is defined as a class-level method on a `flir.MlirModule` subclass.
2. `@flir.jit __call__` is also defined as a class-level module method.
3. The module has a real GPU module name and target.
4. The launch wrapper is defined through `flir.gpu_ext.LaunchFuncOp(...)` in a normal module method, not in an ad hoc nested diagnostic function.

The strongest conclusion from the passing probes is:

The full gist compute body is not required just to make FLIR executable.

What is required is the correct class-level FLIR module shape.

## Minimal Non-Working FLIR vs Working FLIR

### Non-working direction

Representative file: [submission_flir_stream_reject.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_stream_reject.py)

Key properties:

1. Built as a diagnostic rather than a realistic module boundary.
2. Earlier failing versions defined FLIR entrypoints inside `init_gpu_module(...)`.
3. No evidence of a properly exported callable boundary on the remote executor.
4. Good for introspection, not a trustworthy model of how a real FLIR module should be authored.

### Working direction

Representative files:

1. [submission_flir_min_class_level.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_class_level.py)
2. [submission_flir_min_gemm_signature.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_min_gemm_signature.py)
3. [submission_flir_gist_like.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_flir_gist_like.py)

Key properties:

1. Real class-level FLIR module methods.
2. Real `__call__` wrapper.
3. Real launch op.
4. Realistic signature and launch shape in the GEMM-signature probe and full gist.

## What We Fully Understand Now

We do understand the main boundary now.

### What we know confidently

1. Public README-style FlyDSL launch and FLIR module execution are different failure categories.
2. The README-style `flyc/fx` path is blocked by harness stream policy.
3. FLIR itself is not blocked by that same conclusion.
4. Class-level FLIR module methods are sufficient to make remote execution work.
5. The original minimal FLIR failure came from the wrong module/export shape, not from FLIR being unsupported.

### What we do not fully know yet

1. Exactly which internal registration step is skipped when `@flir.kernel` and `@flir.jit` are nested under `init_gpu_module(...)`.
2. Whether there are additional hidden constraints beyond class-level placement for more complex kernels.
3. Why the current gist FLIR kernel is so slow compared with AITER.

## Current HIP Issue

The current HIP state is represented by [submission_hip_scratch_timeout.py](/home/oldzhu/gpumode/amd202602_vs/mxfp4-mm/submission_hip_scratch_timeout.py).

What we know:

1. Inline HIP compilation through `torch.utils.cpp_extension.load_inline(...)` works remotely.
2. The problem is not specific to MXFP4 tensor handling.
3. Even a scratch-only kernel that just writes one `int32` value hangs on the remote launch path.
4. That means the blocker is at or below the extension launch/runtime boundary, not in the MM math contract.

Current HIP conclusion:

HIP is not yet a usable execution backend in this harness.

It is only a compile-validated path today, not a launch-validated one.

## What This Means For Other Problems

If we later try to extend backend experiments to MoE or MLA, the current evidence suggests:

### AITER

Safest path to extend first.

Reason:

1. Already accepted by the harness.
2. Already competitive.
3. Thin user-side code.

### FLIR / FlyDSL

Worth extending only through the FLIR module style that now works.

Rules to carry forward:

1. Do not start from README-style `flyc/fx` launches for harness-critical work.
2. Do not start from nested diagnostic FLIR entrypoints.
3. Start from class-level `flir.MlirModule` with class-level `@flir.kernel` and class-level `@flir.jit __call__`.
4. Prefer minimal runnable probes that preserve correctness through a trusted backend while validating FLIR launch.

### HIP C++

Do not treat as a portable backend yet.

Reason:

1. Compilation works.
2. Launch does not.
3. The launch failure reproduces even on scratch-only code.

## Recommendation For Cross-Problem Backend Comparison

To compare AITER, FLIR, and HIP across problems, the current best decision rule is:

1. AITER is the only end-to-end proven competitive backend right now.
2. FLIR is now a proven executable research backend, but not a performance backend yet.
3. HIP is still blocked at launch and should not be used for cross-problem performance comparison until a remote launch path succeeds.

For MM specifically, current evidence says that more time on FLIR GEMM math tuning is lower value than understanding and reducing FLIR launch/runtime overhead.

So if the goal is to make all three "work" before comparing them:

1. AITER: already works.
2. FLIR: now works, but only with the correct module shape.
3. HIP: still does not work end-to-end; the missing step is remote launch success.