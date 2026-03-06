# GPU Testing Guide

This guide is for the GPU engineer who will test and validate the SonicMoE optimizations.

## Prerequisites

- NVIDIA Hopper GPU (H100/H200) or Blackwell GPU (B200/GB200)
- CUDA 12.6+ (12.9+ for Blackwell)
- Python 3.12+
- PyTorch 2.7+

## Setup

```bash
git clone <repo-url>
cd sonic-moe
git checkout optimizations/megakernel-blackwell
pip install -e .
```

## Step 1: Run Config Tests (No GPU Required)

```bash
pytest tests/megakernel_blackwell_test.py -v -k "Config or Architecture"
```

**Expected:** All tests PASS.

## Step 2: Run Correctness Tests

```bash
pytest tests/megakernel_blackwell_test.py -v -k "Correctness"
```

**Expected:** All tests PASS. If `test_matches_sequential_baseline` fails, report the error traceback.

## Step 3: Run Benchmarks

```bash
pytest tests/megakernel_blackwell_test.py -v -k "benchmark" -s
```

**Expected:** Prints timing numbers. Record them.

## Step 4: Run Original SonicMoE Benchmarks for Comparison

```bash
# Baseline (original SonicMoE)
python benchmarks/moe-cute.py --thiek 4096,4096,1024,64,2 --activation swiglu
```

Record the timing and compare with Step 3.

## What to Report

Please provide:

1. **GPU model** (e.g., H100 80GB SXM5)
2. **CUDA version** (`nvcc --version`)
3. **Test results** (all pass / which ones fail)
4. **Benchmark numbers:**
   - Original SonicMoE forward time (ms)
   - Megakernel forward time (ms)
   - Speedup percentage
5. **Any error tracebacks** if tests fail

## Troubleshooting

### Import errors
```bash
pip install -e .  # reinstall
```

### CUDA out of memory
Reduce the benchmark size in `tests/megakernel_blackwell_test.py`:
- Change `T = 4096` to `T = 2048`
- Change `H = 4096` to `H = 2048`

### Compilation errors
The CuTeDSL kernels compile on first run. This takes 2-5 minutes. If compilation fails:
1. Check CUDA version: `nvcc --version` (need 12.6+)
2. Check GPU capability: `python -c "import torch; print(torch.cuda.get_device_capability())"`
   - Expect `(9, 0)` for Hopper or `(10, 0)` for Blackwell
