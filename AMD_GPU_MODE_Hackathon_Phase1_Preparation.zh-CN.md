# AMD x GPU MODE - E2E Model Speedrun Hackathon
## 第一阶段资格赛准备指南（中文配套版）

[English Version](AMD_GPU_MODE_Hackathon_Phase1_Preparation.md)

## 说明

这是英文原文的中文配套文档，保留相同主题结构并提炼关键内容，便于快速查阅。涉及精确命令、代码示例和英文术语时，以英文原文为准。

## 活动概览

- 活动名称：AMD x GPU MODE - E2E Model Speedrun
- 平台：GPU MODE
- 社区：GPU MODE Discord
- 主题：围绕 AMD GPU 上的大模型推理关键内核，进行性能与正确性竞赛

## 第一阶段问题集

### 问题 1：Mixed-MLA

- 目标：优化 DeepSeek-R1 在 decode 阶段的 MLA 注意力路径
- 输入包含：`q`、多种格式的 KV cache、指针数组与配置字典
- 输出：bf16 注意力结果
- 优化方向：FP8 / MXFP4 精度路径、persistent decode、metadata 复用

### 问题 2：MoE-MXFP4

- 目标：实现带 MXFP4 权重的融合 MoE 层
- 输入包含：hidden states、gate/up/down 权重及 scale、top-k routing 信息与配置字典
- 输出：bf16 hidden states
- 关键点：MXFP4 维度对齐、预洗牌权重、fused gate/up/down 计算

### 问题 3：MXFP4-MM

- 目标：实现 bf16 A × MXFP4 B -> bf16 C 的矩阵乘
- 输入包含：原始 `A/B`、预量化 `B_q`、`B_shuffle`、`B_scale_sh`
- 关键点：每 32 元素 block scale、每字节打包 2 个 FP4 值

## 依赖与工具

- 核心库：AITER
- 常用模块：MLA、fused_moe、shuffle、dynamic_mxfp4_quant、FP4/E8M0 工具函数
- 常见自定义 dtype：`fp4x2`、`fp8_e8m0`、`fp8`

## 提交结构

每个问题都需要 `submission.py` 风格入口：

- 从 `task` 导入 `input_t` 与 `output_t`
- 实现 `custom_kernel(data)`
- 返回与任务约定一致的输出

## 评估标准

- 正确性：与 reference 做数值对比
- 性能：多次运行统计均值、方差、最好/最差值
- 评分：越快越好

## 快速开始建议

1. 先搭环境并阅读 reference 与 task schema
2. 先做可运行的基线实现
3. 通过本地或远端 `test`
4. 再逐步进入优化阶段

## 优化策略摘要

### Mixed-MLA

- 优先复用 metadata 与输出缓冲
- 减少 Q / KV 量化与准备阶段的额外开销

### MoE-MXFP4

- 优先使用成熟 fused kernel
- 关注输入布局和每次调用的额外准备成本

### MXFP4-MM

- 优先减少 launch 开销与多余 host 侧工作
- 尽量利用预洗牌的 B 张量与现成 scale

## 注意事项

- 正确性是第一门槛
- 不同问题的最佳路径可能差异很大
- 远端 runner 与本地环境可能不完全一致

## 成功清单

- 能正确提交并通过测试
- 理解三类问题的输入输出和精度约束
- 能基于数据而不是猜测做优化决策

## 帮助渠道

- GPU MODE Discord
- reference-kernels 仓库
- AITER 仓库与示例