# AGENTS.md - AMD GPU MODE Hackathon 第一阶段代理指南

[English Version](AGENTS.md)

## 项目概览

本工作区对应 AMD x GPU MODE E2E Model Speedrun Phase 1 资格赛。
目标是在 MI355X 上优化三个 DeepSeek-R1 推理内核，并尽量提升榜单成绩。

当前三个主要问题目录为：

| 目录 | 榜单 | 内核 |
|---|---|---|
| `mxfp4-mm/` | `amd-mxfp4-mm` | bf16 A x MXFP4 B -> bf16 C |
| `moe-mxfp4/` | `amd-moe-mxfp4` | DeepSeek-R1 MoE with MXFP4 weights |
| `mixed-mla/` | `amd-mixed-mla` | DeepSeek-R1 MLA decode |

建议把英文版 `AGENTS.md` 视为更详细的主说明，本中文文件用于同步关键规则与工作约定。

## 构建与提交

- 统一使用 Popcorn CLI 做远端 `test`、`benchmark`、`leaderboard`、`profile`。
- 提交命令模板见英文版原文。
- 查询排名时使用对应 leaderboard URL。

## 仓库结构约定

每个问题目录应保留：

- `submission_clean.py`：当前候选实现
- `PROGRESS.md`：双语变更日志与测量历史
- `submission_optimized.py`：可选实验分支

## 代码约定

- 仅使用 Python，4 空格缩进，目标行宽 100。
- 实现以热路径为中心，避免不必要抽象。
- 不要原地修改输入张量。
- 默认保持 bf16 输出，除非任务明确要求其它 dtype。
- 避免显式同步调用。

## 核心优化工作流

优先级顺序：

1. 去掉冗余分配、clone、reshape 与 host-device 同步。
2. 复用 metadata、scratch buffer 与常见形状输出缓冲。
3. 尽量顺应底层 AITER 内核布局，不要在 Python 中反复重排。
4. 只有在现有 AITER 路径已经吃干榨净后，才考虑更激进的算法变化。

## 文档要求

- 任何性能相关代码改动，都必须更新对应问题目录下的 `PROGRESS.md`，并在单文件中同时包含英文与中文。
- 对于一般项目 Markdown 文档，默认采用英文文件加中文配套文件的形式。
- 中文配套文件命名为同名 `.zh-CN.md`。
- 中英文文件顶部应互相链接，方便切换阅读。
- 若一侧文档发生实质性更新，另一侧应尽量在同一次变更中同步。

## 项目现状提醒

- 对本仓库而言，可测量的性能提升比更大范围的重构更重要。
- Public leaderboard 排名比本地代码优雅程度更重要。
- 若恢复后继续工作，请同时参考英文版与 `PROJECT_STATUS_2026-04-07.md`。