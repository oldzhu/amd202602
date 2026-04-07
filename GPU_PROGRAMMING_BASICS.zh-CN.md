# GPU 编程基础概念（Hackathon 中文配套版）

[English Version](GPU_PROGRAMMING_BASICS.md)

## 说明

本文件是英文版的中文配套摘要，按原文主题顺序整理关键概念，便于在项目中快速回忆。精确定义与示例仍以英文原文为准。

## 1. 量化基础

- 量化的本质：用更低精度表示数值，以换取更低内存占用和更高吞吐
- 常见直觉：`bf16 > fp8 > fp4`，精度逐步下降，但带宽和吞吐通常逐步改善
- 两类常见量化方式：
  - per-tensor：整张量共享一个 scale
  - per-block：每个 block 共享一个 scale，精度通常更好

## 2. FP8

- 文中介绍了 FP8 变体与基本表示思想
- 在本项目里，FP8 常作为较平衡的精度/带宽折中方案

## 3. MXFP4

- MXFP4 使用 FP4 值加 block scale 组合表示
- 常见 block size 为 `32`
- scale 常用 `E8M0` 表示，适合 power-of-two 缩放
- 优势：极低带宽与较好的硬件友好性

## 4. 内存布局与 Shuffle

- GPU 往往更适合按 tile 或特定访问模式排列数据
- shuffle 的目的不是改变数学语义，而是让底层 kernel 的读写更高效
- 在本项目中，某些 B-side 权重与 scale 已经是预洗牌格式，不能随意重排

## 5. Attention 机制

- 标准注意力：`softmax(QK^T / sqrt(d))V`
- Multi-head attention：多个 head 并行处理不同模式
- MLA：通过 latent 空间压缩 KV cache，以显著降低 decode 期缓存压力

## 6. MoE

- MoE 通过路由让每个 token 只激活少量 expert
- DeepSeek-R1 的 routed/shared expert 组合让活跃比例很低
- 关键优化点通常在融合投影、路由后的聚合与数据布局

## 7. GPU 内存层级

- 从快到慢大致为：寄存器、shared memory、L1、L2、HBM
- 实战重点：
  - 提高数据复用
  - 保持访问连续
  - 减少 HBM 流量

## 8. Kernel Fusion

- 把多个串行操作融合成一个 kernel，可以减少内存往返与 launch 开销
- 本项目多个问题都体现了这一点：MoE、MLA、量化 GEMM 都受益于融合

## 9. AITER

- AITER 是本项目最关键的 AMD 内核运行时库
- 常用模块覆盖 MLA、MoE、量化和 GEMM
- 对当前仓库而言，它也是最可信、最稳定的高性能后端基础

## 10. 常见模式

- 反量化模式
- 注意力计算模式
- MoE expert 计算模式
- 这些模式的共同目标是：减少准备开销，尽量把工作放进已优化好的内核路径中

## 快速查阅结论

- 低精度格式的收益常常来自带宽节省，而不只是算力本身
- 数据布局错误会让正确内核变成错误结果或明显退化
- 如果已有成熟 AITER 路径，优先吃透它，再考虑自定义后端