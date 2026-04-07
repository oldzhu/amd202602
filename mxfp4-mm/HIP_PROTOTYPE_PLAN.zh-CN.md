# MXFP4-MM HIP 原型计划

[English Version](HIP_PROTOTYPE_PLAN.md)

## 目标

为 `mxfp4-mm` 构建一个新的低层实验方向：在保持真实任务契约的前提下，用自定义 HIP/C++ kernel 作为计算路径。

这是研究计划，不是已经被证明可接受的优化方案。

## 当前阶段总结

英文原文记录了多个阶段的验证结果。中文总结如下：

- B 侧权重布局问题已经基本厘清，不再是主要阻塞
- 当前自定义 HIP kernel 的核心问题在于计算结构本身，而不是输入语义不清
- 细粒度 shared-memory 协作重写已经试过，但同步成本太高，性能更差
- 早期某些 benchmark 结果曾被一次性 `_PROBE_LAUNCHED` 门控误导，后续已确认真实 steady-state 性能远不具竞争力
- 目前这个方向的价值主要是“验证 runner 是否允许某类 HIP 路径”，不是已经接近可提交方案

## 为什么还考虑这个方向

- 纯 PyTorch 参考实现语义清晰，但速度太慢
- 当前可信路径 `submission_clean.py` 只是对 AITER/ASM 的薄包装
- Triton 路径此前遇到了布局和 runner 支持问题
- 上游仓库确实存在 `load_inline(...)` 风格的 AMD/HIP 示例，因此值得做可行性验证

## 需要保留的任务契约

`mxfp4-mm` 的任务输入是：

- `A`
- `B`
- `B_q`
- `B_shuffle`
- `B_scale_sh`

可信路径会：

1. 对 A 做 `dynamic_mxfp4_quant`
2. 对 A 的 scale 做 `e8m0_shuffle`
3. 使用任务提供的 `B_shuffle` 与 `B_scale_sh`
4. 调用 `aiter.gemm_a4w4`

任何 HIP 原型都不应绕开这个基本契约。

## 非目标

- 不预计算输出
- 不把最终路径退化成 dequantize 后再 `torch.mm`
- 不假设 ASM-ready 张量天然适合别的后端

## 已知关键约束

1. `B_shuffle` 与 `B_scale_sh` 是有效的任务输入
2. 先前 Triton 失败的核心问题是布局契约不匹配
3. 直接 ASM override 在 runner 上受限
4. Inline HIP 在“可编译”层面是可行的
5. 上游 HIP 模板只能借鉴构建/编译模式，不能直接套用其数学契约

## 推荐的首个原型边界

- 继续在 Python 中做 A 侧量化与 scale shuffle
- 把以下内容传入自定义 HIP 入口：
  - `A_q`
  - `A_scale_sh`
  - `B_shuffle`
  - `B_scale_sh`
  - 预分配 bf16 输出

这样做的原因是：

- 不在第一步就替换已验证正确的 A 侧量化路径
- 让实验聚焦在 compute kernel 本身
- 保持与当前可信 ASM 路径一致的输入边界

## 语义参考

- 使用 reference-kernels 中的 torch MXFP4 重建逻辑作为数学 oracle
- 关注 FP4 值解释、E8M0 解码、每 32 元素 scale 分组与输出重建规则

## 与上游 HIP 模板的映射关系

- 可复用的是 host 侧构建、编译、load_inline 的结构
- 不能复用的是原模板里的 FP8 语义假设与 scale 契约
- `mxfp4-mm` 需要坚持自己的 packed FP4 + E8M0 + 预洗牌 B 侧契约

## 风险与判断

- runner 接受风险：编译通过不代表 launch 可用
- 契约风险：一旦误解 B 侧布局，结果看似接近也可能完全不可信
- 性能风险：即便语义正确，HIP 原型仍可能远慢于 AITER/ASM

## 当前结论

- HIP 在这个仓库里仍是研究性方向
- 它帮助澄清了 runner 与契约边界
- 但在真正打通远端 launch 并获得竞争级性能之前，不应把它视作主线竞赛方案