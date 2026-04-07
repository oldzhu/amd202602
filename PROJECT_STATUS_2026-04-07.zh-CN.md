# 项目状态 - 2026-04-07

[English Version](PROJECT_STATUS_2026-04-07.md)

## 目的

本文档用于在 AMD GPU MODE Phase 1 截止后冻结当前项目状态，方便后续暂停与恢复时直接接续，而不必重新翻阅大量终端日志。

## 最终公开榜单快照

### MXFP4-MM

- `oldzhu` 的公开成绩：`22.511 us`
- 榜单上显示的公开提交：`submission_clean.py`
- 第一名：`4.354 us`
- Top 10 分数线：约 `8.094 us`

### MoE-MXFP4

- `oldzhu` 的公开成绩：`177.770 us`
- 榜单上显示的公开提交：`submission_clean.py`
- 第一名：`69.917 us`
- Top 10 分数线：约 `120.672 us`

### Mixed-MLA

- `oldzhu` 的公开成绩：`73.592 us`
- 榜单上显示的公开提交：`submission_clean.py`
- 第一名：`22.129 us`
- Top 10 分数线：约 `32.536 us`

## 本阶段最重要结论

在截止前的后半段，唯一仍具现实冲榜可能性的方向是 `mxfp4-mm`。这部分工作确实带来了远端验证过的性能改进，但在截止前没有体现在公开榜单上。

本仓库里后期最强的 MM 结果是混合策略路径：

- `submission_hybrid.py`
- 策略：小 `K` 用 BF16 matmul，大 `K` 用 ASM MXFP4
- 远端 ranked benchmark 几何平均：约 `18.99 us`
- 相对旧公开成绩 `22.511 us` 的提升：约 `15.6%`

但直到截止时，公开榜单仍显示旧的 `22.511 us` 记录，即使之后的公开与 secret run 都显示成功。因此，Phase 1 的正式公开结果仍应以榜单当时显示的值为准。

## 分问题技术总结

### MXFP4-MM

- 本仓库最终找到的最有效方向是混合调度，而不是单纯继续微调 Python 包装层。
- 在小 `K` 场景下，`torch.mm(A, B.t())` 的表现意外地强，因为它去掉了量化和多次 kernel launch 的开销。
- 在大 `K` 场景下，ASM MXFP4 路径依旧更优，因为此时 B 侧压缩后的带宽收益更重要。
- AITER 的 `gemm_a16wfp4` 理论上很有潜力，但不同 runner 对 Triton FP4 的支持不一致，导致其不稳定。
- 一个关键硬阻塞是 `tl.dot_scaled(...)` 的行为：把 `uint8` 当作 FP4 打包替身会改变 scale 维度预期，因此简单 dtype 绕过并不能保持语义正确。
- CUDAGraph 不能作为 leaderboard 路径，因为会被 KernelGuard 拒绝。

### MoE-MXFP4

- 当前仓库已经在高概率有效的 API 层路径上：`aiter.fused_moe.fused_moe(...)`。
- 多个包装层实验都没能缩小与榜单前列之间的巨大差距。
- 后期结论是：如果还要继续提升，基本需要内核级或后端级改动，而不是继续做 Python 侧编排优化。

### Mixed-MLA

- 当前仓库已经使用了 AITER persistent decode 路径，并做了 metadata 与 buffer 复用。
- 后续也尝试过若干结构与 split 调参实验，但公开基线仍远离 Top 10。
- 和 MoE 类似，剩余差距更像是需要更深层的 backend 或 kernel 改进，而不是再做轻量包装层优化。

## 暂停时的文档状态

- 每个问题目录下的 `PROGRESS.md` 仍然是主要的双语实验日志。
- 现在已经在 `.github/copilot-instructions.md` 中补充了项目级双语文档规范，并在 `AGENTS.md` 中保持一致。
- 当前英文 Markdown 文档均已补充中文配套文件，并在中英文文件顶部互相链接。

## 后续恢复时建议起点

如果后面要继续这个项目，建议按以下顺序恢复：

1. 除非 GPU MODE 后续补录了延迟更新的 MM 结果，否则以上公开榜单数值应视为 Phase 1 最终公开成绩。
2. 把 `mxfp4-mm/submission_clean.py` 视为当前 MM 候选基线，但恢复前要重新确认，因为截止后可能又有额外编辑。
3. 若要重新开启 FLIR 或 HIP 方向，先读 `mxfp4-mm/BACKEND_COMPARISON.md` 与 `mxfp4-mm/HIP_PROTOTYPE_PLAN.md`。
4. 今后新增或修改文档时，默认同步更新英文与中文配套文件。