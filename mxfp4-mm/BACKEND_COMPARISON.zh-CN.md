# MXFP4-MM 后端对比

[English Version](BACKEND_COMPARISON.md)

## 目的

解释为什么早期最小 FlyDSL/FLIR 探针失败、后来哪些 FLIR 形态可以工作、HIP 当前卡在哪里，以及这些结论对其它问题意味着什么。

## 当前后端状态

- `AITER`：端到端可用、正确、性能也有竞争力，是当前可信基线
- `FLIR / FlyDSL`：在正确模块形态下可以运行，但性能很慢，当前更像研究后端而不是竞赛后端
- `Inline HIP C++`：远端编译可过，但 launch 仍不可靠，尚未形成可用端到端路径

## Benchmark 分解结论

- AITER 基线路径在几十微秒量级，仍具实战价值
- 最小或 gist 风格的 FLIR 路径都落在数百微秒量级
- 这说明 FLIR 当前主要问题并不只是算子数学体本身，而是整体执行路径开销就已经很重

## AITER 目前为什么有效

- 路径很薄：量化 A、shuffle activation scale、复用任务提供的 B-side 张量、调用 `aiter.gemm_a4w4`
- 大部分复杂度都已经被放在预调优的后端中，而不是 Python 层

## FlyDSL / FLIR 的两类失败

### 1. README 风格高层 FlyDSL 路径

- 失败原因是 harness 的 stream 策略限制
- 本质上不是 FLIR module export 问题

### 2. 早期最小 FLIR 探针

- 问题是 module / export 形态不对
- 关键入口定义在错误位置，导致远端可调用边界没有正确暴露

## 为什么后来工作的 FLIR 版本能运行

成功版本共享的核心特征：

1. `@flir.kernel` 是 `flir.MlirModule` 子类上的类级方法
2. `@flir.jit __call__` 也是类级方法
3. 使用了正常 module target 与 launch wrapper

强结论：

- 让 FLIR 可执行，不一定需要完整复杂的 gist 计算体
- 真正必要的是正确的 class-level module shape

## 当前最确定的认识

### 已经明确的部分

- README 风格 FlyDSL 与 class-level FLIR module 是两类不同路径
- 前者会被 harness stream policy 卡住
- 后者在正确形态下可以远端执行
- 因此，早期最小 FLIR 失败并不意味着 FLIR 整体不可用

### 还不完全清楚的部分

- 为什么某些嵌套定义方式会跳过关键注册步骤
- 是否还有更复杂 kernel 才会触发的隐含限制
- 当前 FLIR 路径为什么相对 AITER 仍然这么慢

## HIP 当前问题

- `load_inline(...)` 远端编译可行
- 问题不在 MXFP4 语义本身
- 即使 scratch-only kernel，远端 launch 也会挂住或不可用
- 因此 HIP 目前只是“编译已验证”，不是“运行已验证”

## 对其它问题的意义

### AITER

- 仍是最安全、最值得优先扩展的后端

### FLIR / FlyDSL

- 如果要扩展到其它问题，必须沿用当前已验证的 class-level FLIR module 风格
- 不建议从 README 风格 launcher 或嵌套诊断 module 重新开始

### HIP C++

- 在远端 launch 真正跑通前，不应视为可迁移的竞赛后端

## 总体建议

对当前仓库来说：

1. AITER 是唯一已证明“能跑又有竞争力”的主后端
2. FLIR 是已证明可执行、但暂不具备竞赛性能的研究路径
3. HIP 还停留在 launch 阻塞阶段

如果未来继续做跨问题后端比较，应以上述判断作为默认前提。