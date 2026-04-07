# AGENTS.md - AMD GPU MODE Hackathon Phase 1 Agent Guide

[中文版本](AGENTS.zh-CN.md)

## Project Overview

This workspace targets the AMD x GPU MODE E2E Model Speedrun Phase 1 qualifiers.
The goal is to improve three DeepSeek-R1 inference kernels on MI355X before 2026-04-06 23:59 PST and push at least one submission into the top 10.

The three active problem folders are:

| Folder | Leaderboard | Kernel |
|---|---|---|
| `mxfp4-mm/` | `amd-mxfp4-mm` | bf16 A x MXFP4 B -> bf16 C |
| `moe-mxfp4/` | `amd-moe-mxfp4` | DeepSeek-R1 MoE with MXFP4 weights |
| `mixed-mla/` | `amd-mixed-mla` | DeepSeek-R1 MLA decode |

Prefer this file as the single source of workspace-wide agent guidance.

## Build And Submission

Agents should use Popcorn CLI for all remote validation and ranking work:

```bash
popcorn-cli submit --gpu MI355X --leaderboard <leaderboard> --mode test submission_clean.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard <leaderboard> --mode benchmark submission_clean.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard <leaderboard> --mode leaderboard submission_clean.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard <leaderboard> --mode profile submission_clean.py --no-tui
```

Submission modes are `test`, `benchmark`, `leaderboard`, and `profile`.

Use these leaderboard URLs when rank lookup is needed:

- `amd-mixed-mla`: https://www.gpumode.com/leaderboard/765?tab=rankings
- `amd-moe-mxfp4`: https://www.gpumode.com/leaderboard/764?tab=rankings
- `amd-mxfp4-mm`: https://www.gpumode.com/leaderboard/763?tab=rankings

## Repo Structure

Each problem directory should keep:

- `submission_clean.py`: current candidate implementation
- `PROGRESS.md`: bilingual change log and measurement history
- `submission_optimized.py`: optional side path for experiments

Use the existing markdown docs for background instead of re-embedding them:

- `AMD_GPU_MODE_Hackathon_Phase1_Preparation.md`: problem summaries and baseline context
- `GPU_PROGRAMMING_BASICS.md`: GPU optimization notes
- `hackathon_learning_path.md`: preparation checklist
- `STEP_BY_STEP_GUIDE.md`: legacy onboarding notes; verify commands against this file before reusing because some older examples are MI300-era

## Coding Conventions

- Python only, 4-space indentation, target line length 100.
- Keep implementations minimal and hot-path oriented; avoid extra abstractions in submission files.
- Do not modify input tensors in place.
- Preserve bf16 outputs unless the task explicitly requires another dtype.
- Avoid explicit synchronization calls; the evaluation framework handles timing and sync.
- Only make tensors contiguous when the kernel needs it or layout is uncertain.

## Kernel-Specific Rules

### MXFP4-MM

- Use `dynamic_mxfp4_quant` for A-side activation quantization.
- Always shuffle activation scales with `e8m0_shuffle` before `aiter.gemm_a4w4`.
- `B_shuffle` and `B_scale_sh` are already prepared by the task input; do not reshuffle them.

### MoE-MXFP4

- Use `aiter.fused_moe.fused_moe` with `ActivationType.Silu` and `QuantType.per_1x32`.
- Use the pre-shuffled weights and scales from the task input.
- Keep an eye on input layout and per-call overhead; the fused kernel is already the compute-optimized path.

### Mixed-MLA

- Baseline should stay on the AITER persistent decode path with fp8 Q and fp8 KV.
- Reuse metadata workspaces, output buffers, and helper tensors where possible to reduce per-call overhead.
- Treat MXFP4 KV experiments as speculative unless a supported decode kernel is available.

## Optimization Workflow

When changing performance code, focus on the highest-probability wins first:

1. Remove redundant allocations, clones, reshapes, and host-device sync points.
2. Reuse metadata and scratch buffers for repeated benchmark shapes.
3. Keep memory layout compatible with the underlying AITER kernel rather than reformatting in Python.
4. Only attempt algorithmic changes after the AITER reference path is fully exploited.

Useful review checklist for submissions:

- Are there redundant `.item()`, `.clone()`, `.contiguous()`, or allocation-heavy helper calls in the hot path?
- Are kernel-required pre-shuffled weights or scales being reused rather than rebuilt?
- Are per-shape work buffers cached when benchmark loops reuse the same dimensions?
- Is the implementation still aligned with the current reference kernel behavior?

## Documentation Requirements

Every performance-oriented code change must update that problem's `PROGRESS.md` in both English and Chinese.

For general project documentation outside `PROGRESS.md`:

- Keep an English source file plus a Chinese companion file with the same base name and `.zh-CN.md` suffix.
- Add a link near the top of each pair so the English and Chinese documents point to each other.
- When one side is updated materially, update the companion document in the same change when practical.
- A single-file bilingual exception is allowed for `PROGRESS.md`, which should remain bilingual in one file unless the project structure changes intentionally.

Each change entry must include:

```markdown
## Change: [Brief Title]

### English
- **What**: [Description of the change]
- **Why**: [Reason for the change]
- **Result**: [Observed or expected impact]

### 中文
- **内容**: [描述]
- **原因**: [原因]
- **结果**: [已观察到或预期的影响]

### Profile Measurement
- **Before**: [time in us or prior status]
- **After**: [time in us or pending]
- **Improvement**: [measured delta or pending]
- **Leaderboard Rank**: [rank or pending]
```

Submission history should also track date, mode, pass/fail, timing, and leaderboard rank when available.

## External References

- Competition page: https://luma.com/cqq4mojz
- Official reference kernels: https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602
- AITER: https://github.com/ROCm/aiter
- Popcorn CLI: https://github.com/gpu-mode/popcorn-cli
- GPU MODE Discord: https://discord.gg/gpumode

## Phase 1 Reality Check

- Phase 1 runs from 2026-03-06 to 2026-04-06 23:59 PST.
- Top 10 individuals or teams advance to Phase 2 finals.
- Public leaderboard position matters more than local elegance.
- For this repo, prioritize measurable latency improvements and documented experiments over broad refactors.
