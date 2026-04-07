# PROGRESS.md - MoE-MXFP4

## Change: Rejected block_size_M=None for all shapes

### English
- **What**: Created `submission_no_blockm.py` with `block_size_M=None` for ALL shapes (letting AITER auto-select). Based on romepen788's 145.8μs entry using "no_blocksizeM".
- **Why**: AITER's auto-selection chooses block_m=64 for E=33 shapes (vs our forced 32). Auto might be better.
- **Result**: WORSE. Non-ranked geomean ~183μs, ranked ~185μs (vs clean.py's 177.8μs). AITER's auto block_m is suboptimal for E=257 shapes.

### 中文
- **内容**: 所有形状使用 block_size_M=None。
- **结果**: 更差。Ranked 185μs (原 177.8μs)。

## Change: torch.empty caching for quant buffers — Test PASSED

### English
- **What**: Created `submission_prealloc_quant_v2.py` with safe torch.empty interception. Monkey-patches `aiter.fused_moe.fused_dynamic_mxfp4_quant_moe_sort` (correct module binding). During the quant function call, torch.empty returns cached tensors by shape/dtype/device key.
- **Why**: Each fused_moe call makes 4 torch.empty allocations (2 per stage × 2 stages) for MXFP4 quant buffers. Caching saves allocation overhead.
- **Result**: Test PASSED 3/3. Leaderboard submission was killed by pipe; result unknown but leaderboard didn't update (score likely ≈177μs or worse).
- **Key finding**: AITER auto-selects `block_m=64` for E=33 shapes (test STDERR), confirming that clean.py's forced 32 differs from auto.

### 中文
- **内容**: 通过 torch.empty 拦截预分配量化缓冲区。
- **结果**: 测试通过 3/3。Leaderboard 结果未知。

## Change: Combined prealloc sort + quant caching — NO IMPROVEMENT (177.7μs)

### English
- **What**: Created `submission_combined.py` combining: (1) pre-alloc sort buffers for M≤256, (2) torch.empty caching for quant buffers, (3) block_m=32 for M≤256 via direct fused_moe_2stages, (4) fused_moe wrapper fallback for M>256.
- **Why**: Maximizes buffer reuse across both sorting and quantization steps.
- **Result**: Test PASSED 3/3. Leaderboard ranked geomean: **177.7μs** — IDENTICAL to clean.py's 177.770μs. The allocation savings (~1-2μs per call) are too small to move the geomean.

### 中文
- **内容**: 合并预分配排序 + 量化缓存 + 直接 fused_moe_2stages。
- **结果**: Leaderboard 177.7μs，与 clean.py 的 177.8μs 基本相同。分配节省太小。

### Profile Measurement
- **Before**: clean.py 177.770μs (leaderboard)
- **After**: combined 177.7μs (leaderboard, ranked geomean)
- **Improvement**: ~0.04% (negligible)
- **Leaderboard Rank**: Unchanged at 177.770μs
- **Result**: Test queued, pending rate limit.

### 中文
- **内容**: 组合预分配排序+量化缓冲区。
- **结果**: 测试待提交。

## Change: Rejected doweight_stage1=True (FAILED)

### English
- **What**: Tested `doweight_stage1=True` parameter in `fused_moe()` which changes the JIT module from `mulWeightStage2_` to `mulWeightStage1_`.
- **Why**: Moving weight application from stage 2 to stage 1 might reduce inter-stage data movement.
- **Result**: FAILED all 7 benchmark shapes with wrong numerical results. The weight scaling in stage 1 is incompatible with our per_1x32 + Silu configuration.

### 中文
- **内容**: 测试了 `fused_moe()` 中的 `doweight_stage1=True` 参数。
- **原因**: 将权重应用从 stage 2 移到 stage 1 可能减少阶段间数据搬运。
- **结果**: 所有 7 个形状测试失败，数值结果错误。

### Profile Measurement
- **Before**: 177.770µs leaderboard
- **After**: Test failed — dead end
- **Improvement**: None
- **Leaderboard Rank**: Unchanged at #86

## Change: OPUS sorting experiment (PENDING)

### English
- **What**: Created `submission_opus.py` with `AITER_USE_OPUS_MOE_SORTING=1` environment variable to use alternative sorting kernel.
- **Why**: AITER has two sorting implementations: `moe_sorting_fwd` (default) and `moe_sorting_opus_fwd`. The OPUS variant may be faster.
- **Result**: Test submitted, awaiting results.

### 中文
- **内容**: 创建 `submission_opus.py`，设置 `AITER_USE_OPUS_MOE_SORTING=1` 环境变量。
- **原因**: AITER 有两种排序实现，OPUS 变体可能更快。
- **结果**: 已提交测试，等待结果。

## Change: AITER_KSPLIT=2 + per-shape block_size_M experiment (PENDING)

### English
- **What**: Created `submission_ksplit.py` with `AITER_KSPLIT=2` env var and adaptive block_size_M (32 for M≤16, 64 for mid-range, 128 for large).
- **Why**: For small token counts where token*topk ≤ experts, K-dimension splitting can improve CU utilization. Also testing larger block sizes.
- **Result**: Test submitted, awaiting results.

### 中文
- **内容**: 创建 `submission_ksplit.py`，设置 `AITER_KSPLIT=2` 并自适应 block_size_M。
- **结果**: 已提交测试，等待结果。

## Change: block_size_M=64 for all shapes experiment (PENDING)

### English
- **What**: Created `submission_block64.py` with `block_size_M=64` for all shapes uniformly.
- **Why**: The MI355X has 304 CUs; larger tiles (64 vs 32) might balance CU load differently.
- **Result**: Test submitted, awaiting results.

### 中文
- **内容**: 创建 `submission_block64.py`，所有形状统一使用 block_size_M=64。
- **结果**: 已提交测试，等待结果。

## Change: Rejected forcing `block_size_M=64` for the large `d_expert=2048` case

### English
- **What**: Added a narrower follow-up override on top of the accepted small-token rule so the large routed-expert benchmark case (`num_tokens >= 512` and `d_expert >= 2048`) used `block_size_M=64` while the existing `num_tokens <= 128 and d_expert >= 512 -> 32` rule stayed in place.
- **Why**: The remaining high-signal MoE benchmark gap sat in the largest `d_expert=2048` case, and remote logs showed AITER defaulting that shape to `block_m=128`. This was the smallest untried block-size change that directly targeted that slowest case without disturbing the accepted small-token path.
- **Result**: Rejected. Remote `test` still passed 3/3 and plain `benchmark` improved materially to about `174.760 us` geometric mean, but the leaderboard submission did not beat the standing public score of `177.770 us`. The leaderboard-mode ranked cases landed around `140, 224, 257, 98.3, 128, 218, 342 us`, about `185.509 us` geometric mean, and the public API stayed at submission `678525` / rank `#105`, so the override was reverted.

### 中文
- **内容**: 在已接受的小 token 规则之上又增加了一个更窄的覆盖：让大的 routed-expert benchmark case（`num_tokens >= 512` 且 `d_expert >= 2048`）使用 `block_size_M=64`，同时保留原有的 `num_tokens <= 128 and d_expert >= 512 -> 32` 规则。
- **原因**: 当前 MoE 剩余最明显的 benchmark 缺口集中在最大的 `d_expert=2048` case，而远端日志显示 AITER 对这个形状默认选择的是 `block_m=128`。因此这是一个最小且未测试过的 block-size 变更，能够直接打到最慢的那个 case，同时尽量不扰动已经接受的小 token 路径。
- **结果**: 已放弃。远端 `test` 仍然 3/3 通过，普通 `benchmark` 的几何平均也明显改善到约 `174.760 us`；但 leaderboard 提交没有超过现有公开成绩 `177.770 us`。leaderboard 模式下的 ranked cases 约为 `140, 224, 257, 98.3, 128, 218, 342 us`，几何平均约 `185.509 us`，公开 API 也仍停留在 submission `678525`、排名 `#105`，因此该覆盖已回退。

### Profile Measurement
- **Before**: Accepted public-winning baseline at `181.328 us` remote benchmark geometric mean and `177.770 us` public score
- **After**: Large-`d_expert=2048` `block_size_M=64` candidate at `174.760 us` remote benchmark geometric mean; leaderboard-mode ranked cases at about `185.509 us` geometric mean; public score unchanged at `177.770 us`
- **Improvement**: `3.62%` lower plain benchmark geometric mean, but no public leaderboard improvement and a worse leaderboard-mode ranked geometric mean
- **Leaderboard Rank**: Public snapshot remained `#105`; candidate reverted after submission `709001`

## Change: Rejected conditional-only contiguous normalization in the hot path

### English
- **What**: Changed the MoE wrapper to call `.contiguous()` on `hidden_states`, `topk_weights`, and `topk_ids` only when the incoming tensors reported non-contiguous layouts, leaving the accepted small-token `block_size_M=32` rule unchanged.
- **Why**: The earlier rejected wrapper cleanup mixed padding-cache changes with conditional layout normalization, so this narrower retry isolated the contiguity check itself as a lower-risk Python-side micro-optimization.
- **Result**: Rejected. Remote `test` still passed 3/3, but remote `benchmark` regressed to about `183.688 us` geometric mean versus about `182.720 us` for the simpler baseline and `181.328 us` for the accepted public-winning baseline. The largest routed cases also drifted upward, so the unconditional `.contiguous()` calls were restored.

### 中文
- **内容**: 将 MoE 包装层里的 `hidden_states`、`topk_weights`、`topk_ids` 改成只有在输入张量报告为非连续布局时才调用 `.contiguous()`，其余逻辑保持不变，并继续保留已接受的小 token `block_size_M=32` 规则。
- **原因**: 之前被放弃的包装层清理实验把 padding 缓存和条件连续化混在了一起；这次更窄的重试是为了单独验证 contiguity 检查本身是否是一个低风险的 Python 侧微优化点。
- **结果**: 已放弃。远端 `test` 仍然 3/3 通过，但远端 `benchmark` 的几何平均退化到约 `183.688 us`，相比更简单的基线约 `182.720 us` 和当前公开获胜基线约 `181.328 us` 都更差，较大的 routed case 也有上升，因此已恢复无条件 `.contiguous()`。

### Profile Measurement
- **Before**: Simpler baseline at about `182.720 us` benchmark geometric mean; accepted public-winning baseline at `181.328 us`
- **After**: Conditional-only contiguous candidate at about `183.688 us` benchmark geometric mean
- **Improvement**: None; about `0.53%` worse than the simpler baseline and about `1.30%` worse than the accepted baseline
- **Leaderboard Rank**: Not submitted; reverted after benchmark

## Change: Confirmed that FlyDSL is installed remotely but not selected for this MoE path

### English
- **What**: Ran a one-shot remote `test` probe that logged AITER's live `get_2stage_cfgs` dispatch metadata for the current `submission_clean.py` shapes, then reverted the diagnostic code after the run.
- **Why**: The earlier FlyDSL experiment only proved that the runner stayed on the CK path; this follow-up was meant to distinguish between "FlyDSL unavailable on the runner" and "FlyDSL available but not chosen by AITER for this submission path".
- **Result**: Confirmed. Remote stderr showed `flydsl_available: True`, but the selected functions were still `ck_moe_stage1` and `ck_moe_stage2_fwd`, and AITER logged `using 2stage default` for all tested shapes. This means FlyDSL is present in the environment, but this MoE submission path is not dispatching to it under the current tuned/default selection.

### 中文
- **内容**: 做了一次远端 `test` 探针，在当前 `submission_clean.py` 的形状上把 AITER `get_2stage_cfgs` 的实时 dispatch 元数据打印到日志中；拿到结果后已回退这段诊断代码。
- **原因**: 之前的 FlyDSL 实验只能说明 runner 最终留在 CK 路径上；这次跟进是为了区分到底是“远端没有安装 FlyDSL”，还是“FlyDSL 已安装但 AITER 不会为当前提交路径选中它”。
- **结果**: 结论明确。远端 stderr 显示 `flydsl_available: True`，但实际选择的函数仍然是 `ck_moe_stage1` 和 `ck_moe_stage2_fwd`，而且 AITER 对所有测试形状都记录为 `using 2stage default`。这说明远端环境里确实有 FlyDSL，但当前这条 MoE 提交路径在现有 tuned/default 选择下并不会 dispatch 到 FlyDSL。

### Profile Measurement
- **Before**: Prior note only established that the runner stayed on CK 2-stage; FlyDSL install state on the remote runner was unconfirmed
- **After**: Remote `test` passed 3/3 and explicitly reported `flydsl_available: True` while selecting CK `ck_moe_stage1` and `ck_moe_stage2_fwd`
- **Improvement**: No performance change; this was a backend reachability verification only
- **Leaderboard Rank**: Not submitted; diagnostic code reverted after the remote test

## Change: Rejected narrowing `block_size_M=32` to the `bs=128` routed-expert case only

### English
- **What**: Narrowed the accepted small-token `block_size_M=32` override from `num_tokens <= 128` down to only the `bs=128` routed-expert case while leaving `bs=16` back on AITER's default block-size selection.
- **Why**: Remote `benchmark` improved materially, which suggested that the earlier public gain might be concentrated in the `bs=128` shape rather than shared across the full small-token path.
- **Result**: Rejected. Remote `test` still passed 3/3 and remote `benchmark` geometric mean improved from `181.328 us` to `176.685 us`, but the leaderboard submission did not improve the public result. The public snapshot remained `177.770 us` at rank `#86`, so the narrower override was reverted and the broader `num_tokens <= 128` rule remains the accepted baseline.

### 中文
- **内容**: 将已经接受的小 token `block_size_M=32` 覆盖从 `num_tokens <= 128` 收窄到仅针对 `bs=128` 的 routed-expert case，同时让 `bs=16` 回到 AITER 默认的 block size 选择。
- **原因**: 远端 `benchmark` 有明显改善，这说明之前的公开榜收益可能主要集中在 `bs=128` 这个形状，而不一定来自整个 small-token 路径。
- **结果**: 已放弃。远端 `test` 仍然 3/3 通过，远端 `benchmark` 几何平均也从 `181.328 us` 改善到 `176.685 us`，但 leaderboard 提交没有带来新的公开成绩。公开快照仍然是 `177.770 us`、排名 `#86`，因此该收窄方案已回退，继续保留更宽的 `num_tokens <= 128` 规则作为已接受基线。

### Profile Measurement
- **Before**: Accepted `num_tokens <= 128` override at `181.328 us` benchmark geometric mean; public best `177.770 us`
- **After**: Narrowed `bs == 128`-only override at `176.685 us` benchmark geometric mean; leaderboard snapshot still `177.770 us`
- **Improvement**: `2.56%` lower remote benchmark geometric mean, but no public leaderboard improvement
- **Leaderboard Rank**: Unchanged at `#86`; latest submission did not beat submission `678525`

## Change: Rejected forcing `block_size_M=32` for all `d_expert=512` cases

### English
- **What**: Expanded the new `block_size_M=32` override from the validated small-token routed-expert path to all `d_expert=512` cases, including the larger `bs=512` benchmark shape.
- **Why**: The small-token-only override was a real win, so the next question was whether the same tile size would also help the larger `d_expert=512` case instead of letting AITER keep its default larger block.
- **Result**: Rejected. Remote `test` still passed 3/3, but remote `benchmark` regressed from `181.328 us` to `183.935 us` geometric mean. The largest `d_expert=512` case got noticeably worse, so the submission was reverted to the small-token-only override.

### 中文
- **内容**: 将已经验证有效的 `block_size_M=32` 覆盖从小 token routed-expert 路径扩展到所有 `d_expert=512` 的 case，包括更大的 `bs=512` benchmark 形状。
- **原因**: 既然小 token 专用覆盖已经证明有效，下一步自然是验证同样的 tile size 是否也能改善更大的 `d_expert=512` case，而不是继续让 AITER 选择默认的大 block。
- **结果**: 已放弃。远端 `test` 仍然 3/3 通过，但远端 `benchmark` 几何平均从 `181.328 us` 退化到 `183.935 us`。最大的 `d_expert=512` case 明显变差，因此已回退到仅针对小 token 的覆盖方案。

### Profile Measurement
- **Before**: Accepted small-token-only override at `181.328 us` benchmark geometric mean; public best `177.770 us`
- **After**: Broader `d_expert=512 -> block_size_M=32` attempt at `183.935 us` benchmark geometric mean
- **Improvement**: None; candidate regressed by `1.44%` versus the accepted baseline
- **Leaderboard Rank**: Not submitted; reverted after benchmark

## Change: Force `block_size_M=32` for the small-token routed-expert cases

### English
- **What**: Added a shape-specific `block_size_M=32` override for the small-token MoE path (`num_tokens <= 128` and `d_expert >= 512`) while leaving the larger cases on AITER's default block-size selection.
- **Why**: The earlier forced-`64` experiment showed that `block_size_M` is a real control surface but that `64` was the wrong direction for the most latency-sensitive routed-expert cases. The next low-risk probe was to force the smaller tile instead.
- **Result**: Kept. Remote `test` passed 3/3, remote `benchmark` geometric mean improved from `182.720 us` to `181.328 us`, and the public leaderboard submission improved the score from `180.445 us` to `177.770 us`, moving rank from `#135` to `#86`.

### 中文
- **内容**: 为小 token 的 MoE 路径（`num_tokens <= 128` 且 `d_expert >= 512`）增加了一个形状特定的 `block_size_M=32` 覆盖，而更大的 case 仍保持 AITER 默认的 block size 选择。
- **原因**: 之前强制 `64` 的实验说明 `block_size_M` 确实是一个可控面，但对最敏感的 routed-expert 小 token case 来说，`64` 的方向是错的。因此下一个低风险探测就是尝试更小的 tile。
- **结果**: 保留。远端 `test` 3/3 通过，远端 `benchmark` 几何平均从 `182.720 us` 改善到 `181.328 us`，公开榜提交也把分数从 `180.445 us` 提升到 `177.770 us`，排名从 `#135` 上升到 `#86`。

### Profile Measurement
- **Before**: Reverted simple wrapper benchmark `182.720 us` geometric mean; public best `180.445 us`
- **After**: Remote benchmark `181.328 us`; ranked run `177.862 us`; public score `177.770 us`
- **Improvement**: `0.76%` lower remote benchmark geometric mean and `1.48%` better public score versus the prior public best
- **Leaderboard Rank**: `#86` after submission `678525`

### Follow-up Result
- **Leaderboard Submission**: `678525`
- **Public Outcome**: Improved public score from `180.445 us` to `177.770 us`, and rank from `#135` to `#86`
- **Ranked Cases**: `130, 215, 246, 90.5, 122, 215, 345 us`
- **Decision**: Keep the `block_size_M=32` override as the new `moe-mxfp4` baseline

## Change: Probed FlyDSL-backed FusedMoE dispatch on the remote runner

### English
- **What**: Added a guarded MoE wrapper experiment that tried to steer AITER `fused_moe` onto FlyDSL A4W4 stage1/stage2 kernels when FlyDSL was available, while preserving the existing CK path as fallback.
- **Why**: Among the three Phase 1 kernels, MoE is the only one where upstream AITER explicitly supports optional FlyDSL dispatch for MXFP4 FusedMoE, so this was the most realistic backend-switch experiment with a chance to improve rank.
- **Result**: Rejected. Remote `test` passed, but stderr showed the runner still used the default CK 2-stage path (`module_moe_ck2stages...`) and never selected FlyDSL. A follow-up `profile` submission ended with `Stream ended unexpectedly without a final result or error event`, so the experiment did not produce a usable performance signal and was reverted.

### 中文
- **内容**: 增加了一个带保护的 MoE 包装层实验，尝试在远端环境具备 FlyDSL 时，把 AITER `fused_moe` 导向 FlyDSL 的 A4W4 stage1/stage2 内核；如果不可用则回退到现有 CK 路径。
- **原因**: 在三个 Phase 1 题目里，只有 MoE 这条路径被上游 AITER 明确标注支持可选的 FlyDSL FusedMoE dispatch，因此这是最现实、最有机会带来排名提升的后端切换实验。
- **结果**: 已放弃。远端 `test` 虽然通过，但 stderr 明确显示 runner 仍然使用默认 CK 2-stage 路径（`module_moe_ck2stages...`），并没有真正切到 FlyDSL。随后一次 `profile` 提交又以 `Stream ended unexpectedly without a final result or error event` 结束，没有产出可用性能数据，因此该实验已回退。

### Profile Measurement
- **Before**: Stable CK-backed baseline, public best `180.445 us`
- **After**: `test` passed with unchanged CK backend selection; `profile` did not return a usable result
- **Improvement**: None; remote environment did not expose a usable FlyDSL path for this submission
- **Leaderboard Rank**: Not submitted; candidate reverted after backend probe

## Change: Attempted fixed block_size_M=64 for small-token routed-expert cases

### English
- **What**: Tried forcing `block_size_M=64` for the 33-expert, `num_tokens<=128`, `d_expert>=512` path instead of letting AITER pick the tile size automatically.
- **Why**: `block_size_M` is one of the few remaining correctness-preserving tuning knobs in `fused_moe`, and the small-token routed-expert cases are where our public gap is most visible.
- **Result**: Rejected. Remote `test` passed, but `benchmark` regressed the most important small-token case, so the wrapper was reverted.

### 中文
- **内容**: 针对 33 个 experts、`num_tokens<=128`、`d_expert>=512` 的路径，尝试强制使用 `block_size_M=64`，而不是让 AITER 自动选择 tile 大小。
- **原因**: 在 `fused_moe` 现有接口里，`block_size_M` 是少数仍然保持正确性的可调参数之一，而我们与公开榜的差距主要集中在小 token 的 routed-expert case。
- **结果**: 已放弃。远端 `test` 通过，但 `benchmark` 在最关键的小 token case 上出现回退，因此已回退该包装层改动。

### Profile Measurement
- **Before**: Known benchmark best around `bs=16,dexpert=512` at 89.7 us on the stable wrapper
- **After**: Remote benchmark `bs=16,dexpert=512` 98.5 us; `bs=128,dexpert=512` 127 us; `bs=512,dexpert=2048` 348 us
- **Improvement**: None; the key small-token case regressed by about 9 us
- **Leaderboard Rank**: Not submitted; candidate reverted after benchmark

## Change: Attempted single-kernel sorting override for small-token cases

### English
- **What**: Tried forcing `moe_sorting_dispatch_policy=1` for small-token, many-expert inputs so `fused_moe` would stay on the single-kernel sorting path instead of auto-dispatch.
- **Why**: AITER documents policy `1` as the single-kernel path, which looked like a plausible way to reduce sorting workspace and launch overhead for the `bs=16/128` benchmark cases.
- **Result**: Rejected. Remote `test` passed, but `benchmark` did not improve the relevant cases and remained behind the current public best, so the wrapper was reverted.

### 中文
- **内容**: 尝试在小 token、expert 数较多的输入上强制设置 `moe_sorting_dispatch_policy=1`，让 `fused_moe` 固定走 single-kernel sorting 路径，而不是自动分派。
- **原因**: AITER 文档中将策略 `1` 标记为 single-kernel 路径，看起来有机会减少 `bs=16/128` 这类 benchmark case 的 sorting 工作区和 kernel launch 开销。
- **结果**: 已放弃。远端 `test` 全部通过，但 `benchmark` 没有改善关键 case，整体仍落后于当前公开最好成绩，因此已回退该包装层改动。

### Profile Measurement
- **Before**: Stable fused_moe wrapper, public best rank #121 / 180.445 us
- **After**: Remote benchmark `bs=16,dexpert=512` 92.7 us; `bs=128,dexpert=512` 128 us; `bs=512,dexpert=2048` 348 us
- **Improvement**: None; small-token cases were slightly worse than the known baseline
- **Leaderboard Rank**: Not submitted; candidate reverted after benchmark

## Change: Attempted expert-mask pruning for active experts

### English
- **What**: Tried building an `expert_mask` from `topk_ids` and passing it into `fused_moe` so the kernel could skip experts that were unused in the current routing pattern.
- **Why**: AITER exposes `expert_mask`, and the benchmark cases often activate only a subset of the full expert set, so this looked like a plausible wrapper-level way to reduce sorting and dispatch overhead.
- **Result**: Rejected. Remote test mode timed out after 12 minutes, so this path is not practical in the current submission environment.

### 中文
- **内容**: 尝试根据 `topk_ids` 构造 `expert_mask` 并传给 `fused_moe`，希望内核跳过当前路由中未使用到的 expert。
- **原因**: AITER 提供了 `expert_mask` 参数，而 benchmark case 通常只会激活全部 experts 中的一部分，因此这看起来是一个可能降低 sorting 和 dispatch 开销的包装层优化点。
- **结果**: 已放弃。远端 test 模式在 12 分钟后超时，这条路径在当前提交环境中不可行。

### Profile Measurement
- **Before**: Stable fused_moe wrapper, rank #121 / 180.445 us public score
- **After**: Test workflow timed out
- **Improvement**: None; candidate was discarded
- **Leaderboard Rank**: Not submitted

## Change: Attempted routed/shared expert split with two fused_moe calls

### English
- **What**: Tried separating routed experts and the always-selected shared expert into two `fused_moe` calls, so the routed path would handle only routed experts and the shared path would run as a tiny single-expert call.
- **Why**: The problem README explicitly calls out shared-expert fusion as an opportunity, and this was the simplest way to approximate that idea without writing a custom kernel.
- **Result**: Rejected. Remote test mode failed correctness checks even after preserving the shuffled-layout marker on sliced tensors.

### 中文
- **内容**: 尝试把 routed experts 和始终选中的 shared expert 拆成两个 `fused_moe` 调用，让 routed 路径只处理 routed experts，而 shared 路径单独跑一个极小的单-expert 调用。
- **原因**: 题目的 README 明确提到 shared expert fusion 是一个优化方向，而这是在不写自定义内核的前提下最简单的近似实现方式。
- **结果**: 已放弃。即使在切片张量上保留了 shuffled-layout 标记，远端 test 仍然无法通过正确性检查。

### Profile Measurement
- **Before**: Stable fused_moe wrapper, rank #121 / 180.445 us public score
- **After**: Test failed correctness checks
- **Improvement**: None; candidate was discarded
- **Leaderboard Rank**: Not submitted

## Change: Cache padding metadata and normalize hot-path layouts

### English
- **What**: Cached derived padding values from `config` and only materialized contiguous `hidden_states`, `topk_weights`, and `topk_ids` when their layouts require it.
- **Why**: `fused_moe` is already the right compute kernel, so the remaining low-risk work is to trim Python overhead and avoid surprise layout penalties before the call.
- **Result**: Expected to reduce per-call setup overhead for repeated benchmark shapes without changing numerics.

### 中文
- **内容**: 对 `config` 派生出的 padding 参数做缓存，并且只在布局确实不连续时才对 `hidden_states`、`topk_weights`、`topk_ids` 做连续化处理。
- **原因**: `fused_moe` 已经是正确的高性能计算内核，剩余的低风险优化空间主要在 Python 侧准备开销和输入布局处理。
- **结果**: 预期可以降低重复 benchmark 形状下的调用准备开销，同时不改变数值结果。

### Profile Measurement
- **Before**: Public best 180.445 us / rank #125; current wrapper benchmarked at 184.736 us geometric mean
- **After**: Reverted simpler wrapper benchmarked at 182.720 us geometric mean
- **Improvement**: The cached-padding/conditional-contiguous wrapper was not a win; reverting to the simpler baseline recovered about 1.1% benchmark geometric mean but still did not beat the public best
- **Leaderboard Rank**: Not resubmitted after revert; public best remains 180.445 us

### Follow-up Result
- **Current-wrapper benchmark**: 141, 222, 255, 94.4, 129, 214, 353 us
- **Reverted baseline benchmark**: 138, 219, 251, 93.5, 128, 214, 350 us
- **Decision**: Keep the simpler baseline and treat the padding cache plus conditional `.contiguous()` normalization as a rejected micro-optimization

## Change: Initial implementation based on reference kernel

### English
- **What**: Implemented MoE layer using AITER's fused_moe kernel with MXFP4 quantized weights
- **Why**: Reference implementation provides correct and optimized MoE computation
- **Result**: Passed correctness tests

### 中文
- **内容**: 使用AITER的fused_moe内核和MXFP4量化权重实现MoE层
- **原因**: 参考实现提供正确且优化的MoE计算
- **结果**: 通过正确性测试

### Profile Measurement
- **Before**: Baseline (passed test)
- **After**: Optimized (passed test)
- **Improvement**: Using fused_moe for optimal performance
- **Leaderboard Rank**: Pending benchmark submission

---

## Key Implementation Details

- Uses `ActivationType.Silu` for SwiGLU activation
- Uses `QuantType.per_1x32` for MXFP4 block quantization
- Pre-shuffled weights passed to kernel for optimal memory access
- Proper padding handling for hidden and expert dimensions

## Submission History

| Date | Mode | Result | Time | Notes |
|------|------|--------|------|-------|
| 2026-03-27 | test | pass | - | Initial submission |
| 2026-03-28 | test | pass (3/3) | ~9min | Using fused_moe kernel |
| 2026-03-31 | test | pass (3/3) | ~7min | Forced `block_size_M=32` for the small-token routed-expert path |
| 2026-03-31 | benchmark | pass | ~7min | Benchmark gm improved to `181.328 us` |
| 2026-03-31 | leaderboard | pass | ~8min | Public score improved to `177.770 us`, rank `#86` |
| 2026-03-31 | leaderboard | pass | ~8min | Narrowed `bs=128`-only `block_size_M=32` candidate did not beat the public best; reverted |
| 2026-03-31 | test | pass (3/3) | ~3min | FlyDSL probe reverted; remote runner stayed on default CK 2-stage path |
| 2026-03-31 | profile | failed | - | Stream ended unexpectedly before a final profile result |
| 2026-04-02 | test | pass (3/3) | ~3min | Confirmed `flydsl_available: True` remotely, but AITER still selected CK `ck_moe_stage1` + `ck_moe_stage2_fwd`; probe reverted |

## Benchmark Results

| Date | Mode | Result | Time (μs) | Notes |
|------|------|--------|-----------|-------|
| 2026-03-28 | benchmark | ✅ | 89.7-341 | 7 test cases |
| 2026-03-28 | leaderboard | ✅ | 87.5-348 | Official ranking |
| 2026-03-29 | api-check | ✅ | 180.445 | Verified public score for submission 652655 |
| 2026-03-31 | benchmark | ✅ | 181.328 | Forced `block_size_M=32` for the small-token routed-expert path |
| 2026-03-31 | leaderboard | ✅ | 177.770 | Submission 678525 |

**Best times:**
- bs:16, dexpert:512: 89.7 μs
- bs:128, dexpert:512: 126 μs
- bs:16, dexpert:256: 130 μs

**Leaderboard Rank:** #86 (verified 2026-03-31 via public API)
- Top 10: 129.378 μs
- Our score: 177.770 μs (submission 678525)

## Gap Analysis
- Rank 1: 109.79 μs
- Rank 10: 129.38 μs
- Our rank: ~180 μs (40% slower than rank 10)

## Note
Implementation still uses AITER's fused_moe kernel, but the remote runner clearly responds to shape-specific `block_size_M` steering on the small-token routed-expert cases.

## Optimization Attempts
- fast_mode=True: Not applicable to MoE
- Remaining realistic knobs are limited to `block_size_M`, sorting policy, and any backend exposure the runner actually honors

## Change: Rejected AITER_KSPLIT and dispatch_policy experiments

### English
- **What**: Tested two AITER kernel-level tuning parameters: (1) `AITER_KSPLIT=2` environment variable to force 4-way K-splitting in the 2-stage CK MoE path. (2) `moe_sorting_dispatch_policy=1` to force single-phase sorting kernel.
- **Why**: Explored whether CU utilization could be improved for shapes with few tokens per expert via K-splitting, or whether a different sorting dispatch could reduce overhead.
- **Result**: Both rejected. KSPLIT=2 caused a GM regression from ~177.8 µs to ~192 µs (+8%). The K-splitting overhead outweighs the benefit for shapes with enough parallelism. Dispatch_policy=1 was worse: GM ~218 µs (+23%). The multi-phase auto sorting (policy=0) is already optimal.

### 中文
- **内容**: 测试了两个 AITER 内核级调优参数：(1) `AITER_KSPLIT=2` 环境变量强制 2 阶段 CK MoE 路径进行 4 路 K 分裂。(2) `moe_sorting_dispatch_policy=1` 强制使用单阶段排序内核。
- **原因**: 探索能否通过 K 分裂改善每专家令牌数少的形状的 CU 利用率，或通过不同排序分派减少开销。
- **结果**: 均被拒绝。KSPLIT=2 导致 GM 从 ~177.8 µs 退化到 ~192 µs (+8%)。Dispatch_policy=1 更差：GM ~218 µs (+23%)。

### Profile Measurement
- **Before**: ~177.770 µs baseline leaderboard GM
- **After**: KSPLIT=2: ~192 µs (+8%); dispatch_policy=1: ~218 µs (+23%)
- **Improvement**: None; all experiments regressed
- **Leaderboard Rank**: Unchanged from baseline

### Note on splitk parameter
- The `splitk` parameter in `fused_moe()` does NOT propagate to the 2-stage CK path
- The 2-stage path computes ksplit independently via `get_ksplit()` function
- Only the `AITER_KSPLIT` environment variable can override the 2-stage ksplit
- The `splitk` parameter only affects the 1-stage or legacy code paths

## Change: Tested OPUS MoE Sorting (env var)

### English
- **What**: Tested `AITER_USE_OPUS_MOE_SORTING=1` environment variable to use OPUS sorting kernel instead of default sorting.
- **Why**: OPUS sorting is an experimental variant that might be faster for small batch sizes with many experts.
- **Result**: Inconclusive. The env var does not propagate through popcorn-cli remote execution. Stderr shows standard `module_moe_sorting` was built (not OPUS variant). Benchmark timings appeared similar to baseline, but without proper A/B comparison on identical shapes.

### 中文
- **内容**: 测试了 `AITER_USE_OPUS_MOE_SORTING=1` 环境变量以使用 OPUS 排序内核。
- **原因**: OPUS 排序是一种实验性变体，可能对小批量多专家的情况更快。
- **结果**: 不确定。环境变量未通过 popcorn-cli 远程执行传播。Stderr 显示仍构建了标准 `module_moe_sorting`。

### Profile Measurement
- **Before**: ~177.770 µs baseline
- **After**: OPUS env var didn't propagate; no valid comparison
- **Improvement**: None (env var not effective)
- **Leaderboard Rank**: Unchanged

## Change: OPUS benchmark completed — comparable to default (NEUTRAL)

### English
- **What**: Benchmarked `submission_opus.py` with OPUS sorting. Same tuned CK kernels dispatched for TP=8 shapes.
- **Why**: Testing if OPUS sorting kernel improves token dispatch.
- **Result**: Shapes 4-7 visible: (TP=4 bs=16)~95µs, (TP=4 bs=128)125µs, (TP=4 bs=512)215µs, (EP bs=512)353µs. Near-identical to default (351µs for EP). No significant improvement. OPUS uses `moe_sorting_fwd` (standard module) — the env var sets OPUS mode internally but the compiled module name doesn't change.

### 中文
- **内容**: 完成了 OPUS 排序的 benchmark。使用了与默认相同的调优 CK 内核。
- **结果**: 与默认方案基本相同。OPUS 无显著改善。

### Profile Measurement
- **Before**: ~177.770 µs baseline
- **After**: OPUS benchmark ~similar (353µs for shape 7 vs 351µs default)
- **Improvement**: Negligible
- **Leaderboard Rank**: Not submitted

## Change: block_size_M=64 for all shapes — GPU CRASH (REJECTED)

### English
- **What**: Benchmarked `submission_block64.py` with `block_size_M=64` for all shapes uniformly.
- **Why**: Testing if larger uniform tiles improve utilization on 304 CUs.
- **Result**: CRASHED. Test passed 3/3 but benchmark hit "Memory access fault by GPU node-2" and timed out at 540s. The uniform block_m=64 causes invalid memory access for some shapes — likely because tuned CSV kernels have fixed internal tile sizes that conflict with the forced block_m.

### 中文
- **内容**: 使用 block_size_M=64 对所有形状进行了 benchmark。
- **结果**: 崩溃。GPU 内存访问错误。测试通过但 benchmark 崩溃超时。

### Profile Measurement
- **Before**: ~177.770 µs baseline
- **After**: GPU crash (memory access fault)
- **Improvement**: None — dead end
- **Leaderboard Rank**: Not submitted

## Change: Expand block_size_M=32 to M≤256, remove .contiguous() calls

### English
- **What**: Changed `block_size_m = 32 if M <= 128 and d_expert >= 512 else None` to `block_size_m = 32 if M <= 256 else None`, and removed unnecessary `.contiguous()` calls on `topk_weights` and `topk_ids`.
- **Why**: The d_expert condition was always true for our task (d_expert=2048), so it was dead logic. Expanding to M≤256 extends the block_m=32 benefit to more shapes. The `.contiguous()` calls were verified unnecessary since inputs are already contiguous.
- **Result**: Test PASSED 3/3 (max error 0.015625). Leaderboard submission queued.

### 中文
- **内容**: 将 `block_size_m` 条件从 `M <= 128 and d_expert >= 512` 简化为 `M <= 256`，并移除了 `topk_weights` 和 `topk_ids` 上不必要的 `.contiguous()` 调用。
- **原因**: d_expert 条件对我们的任务始终为真（d_expert=2048），是无效逻辑。扩展到 M≤256 可以将 block_m=32 的收益扩展到更多形状。`.contiguous()` 调用经验证是不必要的。
- **结果**: 测试通过 3/3（最大误差 0.015625）。排行榜提交已排队。

### Profile Measurement
- **Before**: 177.770 µs leaderboard
- **After**: Pending benchmark results
- **Improvement**: Pending
- **Leaderboard Rank**: Pending (submitted 2025-04-06)

## Change: ActivationType.Swiglu — wrong numerical results (REJECTED)

### English
- **What**: Tested `ActivationType.Swiglu` instead of `ActivationType.Silu` in `fused_moe()`.
- **Why**: Swiglu fuses gate+up computation into a single kernel, potentially reducing kernel overhead.
- **Result**: **REJECTED — completely wrong numerical results.** Massive mismatches across all 3 test shapes (30617, 215440, 491114 mismatched elements). Swiglu changes computation semantics fundamentally — it expects a different weight layout where gate and up projections are interleaved, not concatenated.

### 中文
- **内容**: 测试了在 `fused_moe()` 中使用 `ActivationType.Swiglu` 替代 `ActivationType.Silu`。
- **原因**: Swiglu 将 gate+up 计算融合到单个内核中，可能减少内核开销。
- **结果**: **已拒绝——数值结果完全错误。** 所有 3 个测试形状都存在大量不匹配（30617、215440、491114 个不匹配元素）。Swiglu 从根本上改变了计算语义。

### Profile Measurement
- **Before**: N/A
- **After**: Test FAILED (massive numerical errors)
- **Improvement**: None — dead end
- **Leaderboard Rank**: Not submitted

## Change: fused_moe_ Python bypass — impossible (REJECTED)

### English
- **What**: Investigated bypassing the `fused_moe()` wrapper to call underlying kernels directly with pre-allocated buffers.
- **Why**: Source probe revealed `fused_moe_()` is `torch.ops.aiter.fused_moe_` — a C++ registered operator. Sorting, metadata resolution, and dispatch all happen in C++.
- **Result**: **REJECTED — architecturally impossible.** Cannot bypass the C++ dispatch from Python. Only API-level parameters can be tuned.

### 中文
- **内容**: 调查了绕过 `fused_moe()` 包装器直接调用底层内核的可能性。
- **原因**: 源代码探针揭示 `fused_moe_()` 是 `torch.ops.aiter.fused_moe_`——一个 C++ 注册算子。排序、元数据解析和分派都在 C++ 中完成。
- **结果**: **已拒绝——架构上不可能。** 无法从 Python 绕过 C++ 分派。

### Profile Measurement
- **Before**: N/A
- **After**: N/A
- **Improvement**: None — architectural dead end
- **Leaderboard Rank**: Not applicable

## Change: Pre-allocate sorting buffers + call fused_moe_2stages directly

### English
- **What**: Created `submission_prealloc2.py` that bypasses `fused_moe()` to call `aiter.moe_sorting_fwd()` with pre-allocated sorting buffers (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf), then calls `fused_moe_2stages()` directly.
- **Why**: While `fused_moe_()` is a C++ op, the higher-level `fused_moe()` Python wrapper can be bypassed. The sorting step allocates 5 tensors per call. Pre-allocating these per shape should reduce allocation overhead. Top competitor guojun21 (#9, 122µs) uses "prealloc_quant_buffers" approach based on submission filename.
- **Result**: Test submitted, awaiting results.

### 中文
- **内容**: 创建 `submission_prealloc2.py`，绕过 `fused_moe()` 直接调用 `aiter.moe_sorting_fwd()`（使用预分配的排序缓冲区），然后直接调用 `fused_moe_2stages()`。
- **原因**: 虽然 `fused_moe_()` 是 C++ 算子无法绕过，但更高层的 `fused_moe()` Python 包装器可以绕过。排序步骤每次调用分配 5 个张量。按形状预分配应减少分配开销。排名第 9 的竞争者 guojun21（122µs）根据文件名使用了 "prealloc_quant_buffers" 方法。
- **结果**: 测试已提交，等待结果。

### Profile Measurement
- **Before**: 177.770 µs leaderboard
- **After**: Pending
- **Improvement**: Pending
- **Leaderboard Rank**: Pending

## Change: prealloc2 q_dtype fix — TEST PASSED

### English
- **What**: Fixed `submission_prealloc2.py` to pass `q_dtype_a=aiter_dtypes.fp4x2` and `q_dtype_w=gate_up_weight_shuffled.dtype` to `fused_moe_2stages()`. Previous version omitted these parameters, causing incorrect kernel dispatch.
- **Why**: `fused_moe_2stages` uses `q_dtype_a` and `q_dtype_w` to select the correct CK preshuffle module. Without them, `None` values cause config lookup failures and wrong code paths (original `module_moe_asm` instead of `module_moe_ck2stages_fp4x2_fp4x2_preshuffle`).
- **Result**: **TEST PASSED 3/3** with max error 0.015625. The fix triggers '`ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_`' module — a different (and presumably correct) kernel path.

### 中文
- **内容**: 修复 `submission_prealloc2.py`，向 `fused_moe_2stages()` 传入 `q_dtype_a=aiter_dtypes.fp4x2` 和 `q_dtype_w=gate_up_weight_shuffled.dtype`。
- **原因**: `fused_moe_2stages` 使用这些参数选择正确的 CK preshuffle 模块。缺少它们会导致配置查找失败和误入 ASM 路径。
- **结果**: **测试通过 3/3**，最大误差 0.015625。修复后触发了 `ck2stages_fp4x2_fp4x2_preshuffle` 模块。

### Profile Measurement
- **Before**: 177.770 µs leaderboard (submission_clean.py)
- **After**: Benchmark pending
- **Improvement**: Pending
- **Leaderboard Rank**: Pending

## Change: prealloc3 block_size_m=16 — FAILED

### English
- **What**: Created `submission_prealloc3.py` with `block_size_m=16` for M≤32.
- **Why**: Smaller block_size might improve CU utilization for very small batch sizes.
- **Result**: **FAILED — `assert block_size % BLOCK_SIZE_M == 0`** in `fused_dynamic_mxfp4_quant_moe_sort`. MXFP4 quantization requires block_size to be a multiple of the Triton BLOCK_SIZE_M constant (32). block_size_m=16 is below minimum.

### 中文
- **内容**: 创建 `submission_prealloc3.py`，对 M≤32 使用 `block_size_m=16`。
- **原因**: 更小的 block_size 可能改善极小批量的 CU 利用率。
- **结果**: **失败——`block_size % BLOCK_SIZE_M == 0` 断言失败。** MXFP4 量化要求 block_size 是 Triton BLOCK_SIZE_M 常量 (32) 的倍数。

### Profile Measurement
- **Before**: N/A
- **After**: Test FAILED
- **Improvement**: None — block_size_m=16 invalid for MXFP4
- **Leaderboard Rank**: Not submitted

## Change: CKTile dispatch via is_shuffled attribute (TESTING)

### English
- **What**: Created `submission_cktile.py` and `submission_cktile_simple.py` that set `gate_up_weight_shuffled.is_shuffled = True` and `down_weight_shuffled.is_shuffled = True` to trigger the CKTile dispatch path.
- **Why**: When `is_shuffled=True` and `ksplit > 1`, AITER routes to the CKTile backend which **skips FP4 activation quantization entirely** for applicable shapes. This eliminates two `fused_dynamic_mxfp4_quant_moe_sort` calls (~10-20µs each). Top competitors use filenames referencing "cktile" and "cktile_d2048".
- **Result**: Tests submitted, awaiting results.

### 中文
- **内容**: 创建了 `submission_cktile.py` 和 `submission_cktile_simple.py`，设置 `is_shuffled=True` 以触发 CKTile 分派路径。
- **原因**: 当 `is_shuffled=True` 且 `ksplit > 1` 时，AITER 路由到 CKTile 后端，**完全跳过 FP4 激活量化**。这消除了两次 `fused_dynamic_mxfp4_quant_moe_sort` 调用。
- **结果**: 测试已提交，等待结果。

### Profile Measurement
- **Before**: 177.770 µs leaderboard
- **After**: Pending
- **Improvement**: Expected ~20-40µs savings from skipping quant
- **Leaderboard Rank**: Pending

---

## Change: Correct Benchmark Shapes Discovery (2026-04-07)

### English
- **What**: Discovered the ACTUAL benchmark shapes from task.yml (via GitHub reference-kernels). Previous assumptions were WRONG.
- **Why**: Prior CSV and block_m tuning targeted wrong shapes (bs∈{4,64,256} for E=257, and E=33/d=2048 only). Actual shapes are:
  - TP=8: bs∈{16,128,512}, E=257, d_expert=256
  - TP=4: bs∈{16,128,512}, E=33, d_expert=512 ← COMPLETELY MISSING before
  - EP-on: bs=512, E=33, d_expert=2048
  Total: 7 benchmark shapes scored by geometric mean.
- **Result**: All submissions updated with correct shapes. CK CSV benchmark: 180.4µs geomean (WORSE than baseline 177.77µs because block_m=32 suboptimal for E=33 shapes).

### 中文
- **内容**: 从 task.yml 发现了实际的基准测试形状。之前的假设是错误的。
- **原因**: 之前的 CSV 和 block_m 调优针对了错误的形状。实际形状包括一个完全缺失的 E=33/d=512 组。
- **结果**: 所有提交已更新为正确形状。CK CSV 基准: 180.4µs（比基线 177.77µs 更差）。

### Profile Measurement
- **Before**: 177.770 µs leaderboard
- **After**: CK CSV all-block_m=32: 180.4 µs geomean
- **Improvement**: -1.5% regression (uniform block_m=32 hurts E=33 shapes)
- **Leaderboard Rank**: Not submitted

---

## Change: FlyDSL Availability Confirmed on Runner (2026-04-07)

### English
- **What**: Confirmed FlyDSL IS available on MI355X runner (v0.0.1.dev). But the default `dsv3_fp4_tuned_fmoe.csv` has ONLY CK kernel names for benchmark token sizes (1-1024). FlyDSL stage2 entry only at token=16384.
- **Why**: The subagent earlier claimed 50% FlyDSL speedup but that was from a newer/different CSV version. The actual runner CSV uses CK kernels for all benchmark shapes.
- **Result**: Created `submission_flydsl_v2.py` with FlyDSL kernel names to attempt JIT compilation of FlyDSL kernels via custom CSV. Test submitted, awaiting results.

### 中文
- **内容**: 确认 FlyDSL 在 MI355X 运行器上可用（v0.0.1.dev）。但默认 CSV 对基准测试令牌大小仅有 CK 内核名。
- **原因**: 运行器上的实际 CSV 对所有基准形状使用 CK 内核。
- **结果**: 创建了带有 FlyDSL 内核名的自定义 CSV 提交，等待测试结果。

### Profile Measurement
- **Before**: 177.770 µs leaderboard (CK kernels)
- **After**: Pending FlyDSL test
- **Improvement**: Expected significant if FlyDSL JIT compiles successfully
- **Leaderboard Rank**: Pending

---

## Change: Noblock Benchmark Results (2026-04-07)

### English
- **What**: Benchmarked `submission_noblock.py` (no block_size_M, no custom CSV). E=33 shapes used heuristic block_m (32→64→64→128). E=257 shapes used default CSV CK kernels.
- **Why**: Testing whether removing the block_size_M=32 override helps E=33/d=512 shapes.
- **Result**: Partial data: E=33/d=512/bs=128: 128µs(bm=64), bs=512: 213µs(bm=64). E=33/d=2048/bs=512: 349µs(bm=128). Heuristic block_m=64 was SLOWER than forced block_m=32 for E=33/d=512 shapes (vs CK CSV's 117µs and 181µs).

### 中文
- **内容**: 测试了无 block_size_M 覆盖的提交。E=33 形状使用启发式 block_m。
- **原因**: 测试移除 block_size_M=32 覆盖是否有帮助。
- **结果**: 启发式 block_m=64 对 E=33/d=512 形状反而更慢。

### Profile Measurement
- **Before**: 177.770 µs leaderboard
- **After**: E=33 shapes slower with heuristic block_m
- **Improvement**: Negative — heuristic block_m suboptimal for these shapes
- **Leaderboard Rank**: Not submitted
