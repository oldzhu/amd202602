# 🎓 AMD GPU MODE Hackathon - Learning Path for Beginners

[中文版本](hackathon_learning_path.zh-CN.md)

## 📚 Your Learning Journey

This guide is designed for someone with **basic GPU programming concepts** to progress from understanding to implementation. We'll work through the problems from **simplest to most complex**.

---

## 🗺️ Recommended Learning Order

```
Step 1: MXFP4-MM (Simplest - Single Matrix Multiply)
    ↓
Step 2: MoE-MXFP4 (Medium - Multiple Experts with MXFP4)
    ↓
Step 3: Mixed-MLA (Most Complex - Attention with Multiple Data Formats)
```

---

## 📋 Prerequisites Check

Before starting, make sure you have:
- [x] Popcorn CLI installed (you have this!)
- [x] GitHub account for registration
- [ ] Understanding of the problems (we'll cover this)

---

## 🎯 Phase 1 Goals

To qualify for Phase 2, you need to:
1. **Pass correctness tests** - Your output must match reference within tolerance
2. **Submit working solutions** - Even a PyTorch baseline counts!
3. **Learn and improve** - Optimization comes with practice

---

## 💡 Key Insight for Beginners

**You don't need to write custom CUDA kernels to participate!**

The evaluation framework compares your output to the reference. As long as:
- Your output is numerically correct (within tolerance)
- Your implementation runs successfully

You can start with simple PyTorch implementations and optimize later!

---
