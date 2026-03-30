# 🚀 Step-by-Step Guide: AMD GPU MODE Hackathon Phase 1

## 📋 Overview

This guide will walk you through the entire process of participating in Phase 1 of the AMD x GPU MODE Hackathon.

---

## 🔧 Step 1: Register with Popcorn CLI

On your local machine (where popcorn-cli is installed):

```bash
# Register with GitHub (one-time setup)
popcorn-cli register github

# Verify registration
popcorn-cli whoami
```

---

## 📁 Step 2: Prepare Your Submission Files

You have three problems to solve. Copy the starter code from the files provided:

### Problem Files Location:
```
problem1_mxfp4_mm/submission.py    ← Start here (easiest)
problem2_moe_mxfp4/submission.py   ← Second
problem3_mixed_mla/submission.py   ← Last (hardest)
```

### For each problem, you need:
1. `submission.py` - Your implementation
2. (Optional) `task.py` - Already provided by the platform

---

## 🎯 Step 3: Available Leaderboards

Based on the popcorn-cli example, the available leaderboards include:

| Leaderboard Name | Problem | GPU |
|-----------------|---------|-----|
| `amd-fp8-mm` | FP8 Matrix Multiply | MI300 |
| `amd-moe-mxfp4` | MoE with MXFP4 | MI300 |
| `amd-mixed-mla` | Mixed MLA Attention | MI300 |

Run this to see all available leaderboards:
```bash
popcorn-cli list-leaderboards
```

---

## 📤 Step 4: Submit Your Solutions

### Submission Command Format:
```bash
popcorn-cli submit --gpu MI300 --leaderboard <LEADERBOARD_NAME> --mode <MODE> <FILE>
```

### Modes:
- `test` - Quick test (few cases, fast feedback)
- `benchmark` - Performance benchmarking
- `leaderboard` - Official submission
- `profile` - Detailed profiling

### Example Submissions:

```bash
# Test MXFP4-MM (simplest problem)
popcorn-cli submit --gpu MI300 --leaderboard amd-fp8-mm --mode test submission.py

# Test MoE-MXFP4
popcorn-cli submit --gpu MI300 --leaderboard amd-moe-mxfp4 --mode test submission.py

# Test Mixed-MLA
popcorn-cli submit --gpu MI300 --leaderboard amd-mixed-mla --mode test submission.py
```

---

## 📊 Step 5: Understand Results

### What Results Look Like:
```
test.0.spec: m=128; n=256; k=512; seed=42
test.0.status: pass
test.1.spec: m=256; n=512; k=1024; seed=42
test.1.status: pass
check: pass
time.mean: 1234.56  # microseconds
time.std: 12.34
```

### Key Metrics:
- **check: pass/fail** - Did your solution pass correctness tests?
- **time.mean** - Average execution time (lower is better)
- **time.std** - Standard deviation (consistency)

---

## 🔄 Step 6: Iterate and Improve

### Development Cycle:
```
┌─────────────────┐
│  Edit Code      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Submit (test)  │ ← Use test mode for quick feedback
└────────┬────────┘
         ↓
┌─────────────────┐
│  Check Results  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Fix/Optimize   │
└────────┬────────┘
         ↓
    (repeat)
         ↓
┌─────────────────┐
│ Final Submit    │ ← Use leaderboard mode for official
│ (leaderboard)   │
└─────────────────┘
```

---

## 💡 Quick Start Strategy

### Day 1: Get Something Working
1. Copy the starter `submission.py` for MXFP4-MM
2. Submit in `test` mode
3. Even if it fails, you'll learn from the error messages

### Day 2: Understand the Problem
1. Read the reference implementation
2. Understand input/output formats
3. Try to pass correctness tests

### Day 3: Optimize
1. Study the AITER library
2. Implement the optimized version
3. Compare performance with baseline

---

## ⚠️ Common Pitfalls

1. **Wrong output shape**: Check tensor dimensions carefully
2. **Wrong dtype**: Ensure output is bfloat16
3. **Missing CUDA sync**: The framework handles this, but don't add extra syncs
4. **Copy issues**: Don't modify input tensors in-place

---

## 📚 Learning Resources

### While Waiting for GPU:
- Read the reference implementations
- Study DeepSeek-R1 paper (MLA, MoE)
- Learn about quantization (FP8, MXFP4)
- Understand GPU memory hierarchy

### AITER Documentation:
- GitHub: https://github.com/ROCm/aiter
- Study the kernel implementations
- Look at test files for usage examples

---

## 🎯 Phase 1 Completion Criteria

To advance to Phase 2, you typically need:
- Working solutions for all problems (pass correctness)
- Competitive performance (not necessarily the best)
- Demonstrated understanding of the problems

---

## 🆘 Getting Help

1. **Discord**: [discord.gg/gpumode](https://discord.gg/gpumode)
2. **GitHub Issues**: Reference-kernels repository
3. **Popcorn CLI Help**: `popcorn-cli --help`

---

## 📝 Notes Template

Use this to track your progress:

```
Problem: MXFP4-MM
Date Started: ____
First Submission: ____ (pass/fail)
Best Time: ____ microseconds
Notes: _______________________________

Problem: MoE-MXFP4
Date Started: ____
First Submission: ____ (pass/fail)
Best Time: ____ microseconds
Notes: _______________________________

Problem: Mixed-MLA
Date Started: ____
First Submission: ____ (pass/fail)
Best Time: ____ microseconds
Notes: _______________________________
```

---

Good luck! Remember: **Start simple, iterate quickly, learn continuously!**
