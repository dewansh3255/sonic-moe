# SonicMoE Optimizations — Developer Guide

## What We're Building

SonicMoE is a high-performance Mixture-of-Experts (MoE) implementation using CuTeDSL/CUTLASS on NVIDIA Hopper & Blackwell GPUs. Our optimization fuses the **up-projection** and **down-projection** GEMM kernels into a single **megakernel** to eliminate memory bandwidth waste.

### The Problem

In the baseline SonicMoE forward pass, two separate CUDA kernels run sequentially:

```
Kernel 1 (Up-proj):  X × W1 → SwiGLU → Y1    [writes Y1 to HBM]
                                    ↓
                              Y1 in HBM (32 MB @ T=4096, I=4096, BF16)
                                    ↓
Kernel 2 (Down-proj): Y1 × W2 → Y2            [reads Y1 from HBM]
```

The intermediate activation **Y1** makes a full round-trip through HBM (GPU main memory). This is wasteful because:
- HBM bandwidth on H100 ≈ 3.35 TB/s
- Writing + reading Y1 = 64 MB of HBM traffic per MoE layer
- This costs ~19 μs per layer — significant at the kernel level

### Our Solution: Megakernel Fusion

We fuse both kernels into a single execution unit:

```
Megakernel (single custom_op):
  Phase 1: X × W1 → SwiGLU → Y1  [writes Y1 to HBM, warming L2 cache]
  Phase 2: Y1 × W2 → Y2          [reads Y1 from L2 cache, NOT cold HBM]
```

**Phase 1 (Current):** L2-cache-warm fusion. Y1 still goes to HBM, but Phase 2 reads it from L2 cache (which was warmed by Phase 1's write). L2 bandwidth ≈ 12 TB/s → ~3.6x faster than cold HBM.

**Phase 2 (Planned):** True SMEM fusion. Y1 stays in shared memory (on-chip SRAM) between phases. Zero HBM access for Y1.

### Key Terminology

| Term | Meaning |
|---|---|
| **Megakernel** | Multiple kernel operations fused into a single CUDA kernel launch |
| **SMEM fusion** | Keeping intermediate data in shared memory (on-chip) between operations |
| **L2-warm fusion** | Exploiting L2 cache locality between back-to-back kernel phases |
| **HBM round-trip** | Writing data to GPU main memory and reading it back — the waste we're eliminating |
| **TMA** | Tensor Memory Accelerator — H100's hardware for async memory copies |
| **WGMMA** | Warpgroup Matrix Multiply-Accumulate — H100's tensor core instruction |

---

## Expected Performance Improvements

### Hopper (H100)

| Optimization | Expected Speedup | Status |
|---|:---:|---|
| L2-cache-warm megakernel fusion | ~1-2% | ✅ Implemented |
| Kernel launch overhead elimination | ~0.3-0.5% | ✅ Implemented |
| Expert metadata sharing | ~0.1-0.2% | ✅ Implemented |
| **Phase 1 Total** | **~1.5-2.5%** | **✅ Ready for testing** |
| True SMEM fusion (Phase 2) | ~1.5-2.5% | 🔜 Planned |
| **Phase 1 + 2 Total** | **~3-5%** | |

### Blackwell (B200/GB200)

| Optimization | Expected Speedup | Status |
|---|:---:|---|
| L2-cache-warm megakernel fusion | ~1.5-2.5% | ✅ Implemented |
| L2 tuning (`L2_group_size=16`) | ~0.5-1% | ✅ Implemented |
| **Phase 1 Total** | **~2-3%** | **✅ Ready for testing** |
| True SMEM fusion (Phase 2) | ~2-3% | 🔜 Planned |
| **Phase 1 + 2 Total** | **~4-6%** | |

> **Why Blackwell gets more benefit:** B200 has 72 MB L2 cache vs H100's 50 MB, so more of Y1 fits in L2. It also has 192 SMs (vs 132) and 256 KB TMEM per SM.

---

## Repository Structure

```
sonic-moe/
├── sonicmoe/
│   └── functional/
│       ├── grouped_gemm.py          # Original 3070-line Hopper WGMMA kernel
│       ├── moe_config.py            # Kernel configs (tile shapes, pipeline depth)
│       ├── forward.py               # Original separate up/down-proj wrappers
│       ├── backward.py              # Original backward kernels
│       ├── megakernel_kernel.py      # NEW: Two-phase megakernel architecture
│       ├── megakernel_forward.py     # MODIFIED: Fused forward/backward custom ops
│       └── grouped_gemm_blackwell.py # Blackwell architecture tuning
├── tests/
│   └── megakernel_blackwell_test.py  # MODIFIED: Tests + benchmarks
└── docs/
    ├── OPTIMIZATIONS.md              # Technical description of optimizations
    └── GPU_TESTING_GUIDE.md          # This guide
```

---

## Testing Instructions

### Prerequisites

- **GPU**: NVIDIA H100 (SM90) or B200/GB200 (SM100)
- **CUDA**: 12.x
- **Python packages**: PyTorch 2.x (with CUDA), CuTeDSL (`cutlass_cute_dsl`), `quack`

### Setup

```bash
cd sonic-moe
pip install -e .

# Verify environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"
python -c "import cutlass; import cutlass.cute; print('CuTeDSL OK')"
```

### Step 1: Quick Validation (No GPU Compute Needed)

```bash
pytest tests/megakernel_blackwell_test.py -v -k "config or arch"
```

**Expected**: All pass. This tests imports, SMEM budget math, and Blackwell constants.

### Step 2: Correctness Tests

```bash
pytest tests/megakernel_blackwell_test.py -v -k "Correctness"
```

**Expected results**:
- `test_output_shape` → PASS (output shapes match)
- `test_matches_sequential_baseline` → PASS with `atol=1e-3, rtol=1e-3`
- `test_gradient_flow` → PASS (all gradients non-zero)
- `test_expert_frequency_invariant` → PASS (deterministic routing)

### Step 3: Performance Benchmarks ⭐

This is the most important test. It compares our megakernel against the original SonicMoE baseline:

```bash
pytest tests/megakernel_blackwell_test.py -v -k "benchmark" -s
```

**What we need back from these results:**

For each config, the test prints:
```
Config: T=4096, H=4096, I=4096, E=8, K=2
Y1 size: 32.0 MB (L2 fit: Yes)
Sequential baseline: X.XXX ms
Megakernel fusion:   X.XXX ms
Speedup:             +X.XX%
```

**Please report ALL of these numbers.** We specifically need:

1. **Sequential baseline time** (ms) for each config
2. **Megakernel fusion time** (ms) for each config
3. **Speedup percentage** for each config
4. **GPU model** (H100 SXM, H100 PCIe, B200, etc.)

### Step 4: L2 Cache Sweep

```bash
pytest tests/megakernel_blackwell_test.py -v -k "l2_cache_sweep" -s
```

**What this measures**: How speedup changes as Y1 grows beyond L2 cache capacity.

**Expected pattern**:
- T=1024 (Y1=8 MB): Higher speedup (fits in L2)
- T=2048 (Y1=16 MB): Good speedup
- T=4096 (Y1=32 MB): Good speedup
- T=8192 (Y1=64 MB): Lower speedup (exceeds L2 on H100)

### Step 5: Original Baseline Benchmark (for comparison)

Also run the original SonicMoE benchmark to establish the unfused baseline:

```bash
python benchmarks/moe-cute.py --E 8 --K 2 --T 4096 --H 4096 --I 4096 --dtype bf16
python benchmarks/moe-cute.py --E 64 --K 2 --T 4096 --H 4096 --I 4096 --dtype bf16
```

### Step 6: NSight Compute Profiling (Optional but Valuable)

This verifies that Y1 is being served from L2 cache in Phase 2:

```bash
ncu --metrics \
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg,\
  lts__t_sectors_op_read.avg,\
  dram__sectors_read.avg \
  -o megakernel_profile \
  pytest tests/megakernel_blackwell_test.py -v -k "test_output_shape" -s
```

**What to look for**:
- High `lts__t_sectors_op_read` → Y1 hitting L2 cache ✅
- Low `dram__sectors_read` in Phase 2 → Y1 NOT going to HBM ✅

---

## Results Template

Please fill this out and share back:

```
GPU: _____________ (e.g., H100 SXM 80GB)
CUDA version: _____________
PyTorch version: _____________

Config 1: T=2048, H=4096, I=4096, E=8, K=2
  Sequential: _____ ms
  Megakernel: _____ ms
  Speedup: _____%

Config 2: T=4096, H=4096, I=4096, E=8, K=2
  Sequential: _____ ms
  Megakernel: _____ ms
  Speedup: _____%

Config 3: T=4096, H=4096, I=4096, E=64, K=2
  Sequential: _____ ms
  Megakernel: _____ ms
  Speedup: _____%

L2 Cache Sweep (T sweep at H=4096, I=4096, E=8, K=2):
  T=1024: _____ ms
  T=2048: _____ ms
  T=4096: _____ ms
  T=8192: _____ ms

Notes / Issues:
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: cutlass` | Install: `pip install cutlass-cute-dsl` |
| `CUDA out of memory` | Reduce T in test configs or use a GPU with more VRAM |
| `RuntimeError: CUDA error` | Check CUDA driver version matches toolkit |
| Test hangs at compilation | First run compiles CuTeDSL kernels (~30s). Subsequent runs use cache. |
| Speedup is negative | Normal for very small configs where overhead dominates |
| `ImportError: quack` | Install quack: check SonicMoE README for installation |
