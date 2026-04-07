# Project Status - 2026-04-07

[中文版本](PROJECT_STATUS_2026-04-07.zh-CN.md)

## Purpose

This note freezes the project state after the AMD GPU MODE Phase 1 deadline so work can pause cleanly and resume later without reconstructing the context from terminal history.

## Final Public Leaderboard Snapshot

### MXFP4-MM

- Public score for `oldzhu`: `22.511 us`
- Public submission shown on leaderboard: `submission_clean.py`
- Top score: `4.354 us`
- Top-10 cutoff: about `8.094 us`

### MoE-MXFP4

- Public score for `oldzhu`: `177.770 us`
- Public submission shown on leaderboard: `submission_clean.py`
- Top score: `69.917 us`
- Top-10 cutoff: about `120.672 us`

### Mixed-MLA

- Public score for `oldzhu`: `73.592 us`
- Public submission shown on leaderboard: `submission_clean.py`
- Top score: `22.129 us`
- Top-10 cutoff: about `32.536 us`

## Most Important Project Outcome

The only realistic late-stage target was `mxfp4-mm`, and the work there produced real local and remote improvements, but not enough public leaderboard movement before the deadline.

The strongest late result was the hybrid MM path:

- `submission_hybrid.py`
- Strategy: BF16 matmul for small `K`, ASM MXFP4 for larger `K`
- Ranked benchmark geometric mean from remote run: about `18.99 us`
- Improvement over the older public `22.511 us` entry: about `15.6%`

However, the public leaderboard continued to display the older `22.511 us` entry at deadline, even after later public and secret runs reported success. For that reason, the authoritative public finish should still be treated as the displayed leaderboard value.

## Per-Problem Technical Summary

### MXFP4-MM

- Best practical direction found in this repo was hybrid dispatch, not a pure Python wrapper tweak.
- `torch.mm(A, B.t())` was unexpectedly strong on small-`K` shapes because it removes quantization and multi-kernel launch overhead.
- The ASM-backed MXFP4 path stayed better for larger `K` because the compressed B-side memory traffic dominates there.
- The AITER `gemm_a16wfp4` path was promising in theory, but runner-dependent Triton FP4 support made it unreliable.
- A key hard blocker was `tl.dot_scaled(...)` behavior: using `uint8` as a surrogate for FP4 packing changes scale expectations, so a naive dtype workaround does not preserve semantics.
- CUDAGraph was not a viable leaderboard path because KernelGuard rejected it.

### MoE-MXFP4

- The repo already sat on the best high-probability API-level path: `aiter.fused_moe.fused_moe(...)`.
- Multiple wrapper-level experiments did not close the large gap to the top of the leaderboard.
- The late conclusion was that meaningful additional gains likely require kernel-level or backend-level changes, not more Python orchestration changes.

### Mixed-MLA

- The repo already used the persistent AITER decode path with metadata and buffer reuse.
- Several structural and split-tuning experiments were attempted in later work, but the public baseline remained far from the top-10 cutoff.
- As with MoE, the remaining gap appears to require deeper backend or kernel wins rather than more lightweight wrapper tuning.

## Documentation State At Pause

- `PROGRESS.md` in each problem folder remains the primary bilingual experiment log.
- A project-wide bilingual documentation policy has now been added to `.github/copilot-instructions.md` and aligned in `AGENTS.md`.
- English-only Markdown files now have Chinese companion files with reciprocal links.

## Recommended Resume Point

If this project resumes later, start from these items in order:

1. Treat the public leaderboard values above as the official Phase 1 finish unless GPU MODE later reflects the delayed MM runs.
2. Use `mxfp4-mm/submission_clean.py` as the current MM candidate base, but verify it again before resuming because post-deadline edits may have occurred.
3. Use the backend notes in `mxfp4-mm/BACKEND_COMPARISON.md` and `mxfp4-mm/HIP_PROTOTYPE_PLAN.md` before reopening FLIR or HIP experiments.
4. For future documentation changes, update both English and Chinese companion files together.