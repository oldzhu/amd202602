# AMD GPU MODE Hackathon Phase 1 分步指南（中文配套版）

[English Version](STEP_BY_STEP_GUIDE.md)

## 概览

本文件对应英文版的流程指南，适合在重新启动项目或交接时快速回顾参与流程。

## 第 1 步：注册 Popcorn CLI

- 先完成 GitHub 注册绑定
- 用 `popcorn-cli whoami` 验证当前身份

## 第 2 步：准备提交文件

- 每个问题都需要可提交的实现文件
- 最重要的是 `submission.py` 或本仓库采用的 `submission_clean.py`
- 明确每个问题目录、入口文件与可选辅助文件

## 第 3 步：识别可用榜单

- 查清 problem 对应的 leaderboard 名称与 GPU 类型
- 若官方文档示例落后，要以当前平台真实榜单为准

## 第 4 步：提交方案

- 常见模式包括：`test`、`benchmark`、`leaderboard`、`profile`
- 推荐顺序：先 `test`，再 `benchmark`，最后 `leaderboard`

## 第 5 步：理解结果

- `pass/fail` 代表正确性结果
- `time.mean` 代表平均耗时
- `time.std` 代表波动情况

## 第 6 步：迭代改进

典型循环：

1. 编辑代码
2. 提交测试
3. 看结果
4. 修正或优化
5. 重复
6. 最后提交 leaderboard

## 快速启动策略

### 第一天

- 先让最简单的问题跑通
- 及时学习错误信息与平台反馈

### 第二天

- 阅读 reference
- 弄清输入输出格式与布局

### 第三天

- 学习 AITER
- 基于已有正确路径尝试优化

## 常见坑

- 输出 shape 错误
- dtype 错误
- 无谓的额外同步
- 原地修改输入张量

## 学习资源

- reference 实现
- DeepSeek-R1 相关背景
- AITER 仓库与示例
- GPU 内存层级和量化基础知识

## 完成标准

- 能稳定提交并通过测试
- 对问题有足够理解，能解释自己的实现取舍
- 性能是否极致取决于赛况，但稳定正确是必要基础

## 求助渠道

- GPU MODE Discord
- reference-kernels 仓库
- Popcorn CLI 帮助命令

## 备注模板

建议继续使用英文版中的问题跟踪模板，或在 `PROGRESS.md` 中记录实验过程。