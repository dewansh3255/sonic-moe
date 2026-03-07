# SonicMoE Optimizations

## What Is a Megakernel?

A **megakernel** fuses multiple separate CUDA kernel launches into a single kernel. Instead of:

```
Launch Kernel 1 (up-proj) → sync → Launch Kernel 2 (down-proj)
```

We have:
```
Launch Megakernel [Phase 1: up-proj → Phase 2: down-proj]
```

Within the megakernel, we can apply **SMEM fusion**: keeping intermediate data in shared memory (on-chip SRAM) instead of writing it to HBM (off-chip main memory).

| Level | What Happens | Speedup |
|---|---|---|
| **Separate kernels** (baseline) | Y1 → HBM → Y1 (cold read) | 0% |
| **Megakernel** (Phase 1, current) | Y1 → HBM → Y1 (L2 cache hot) | ~1.5-2.5% |
| **Megakernel + SMEM fusion** (Phase 2, planned) | Y1 stays in SMEM | ~3-5% |

---

## Phase 1: L2-Cache-Warm Megakernel (✅ Implemented)

### How It Works

Both GEMMs execute back-to-back in a single `torch.library.custom_op`:

1. **Phase 1**: Up-proj GEMM → SwiGLU → Y1 written to HBM via TMA
   - TMA store **populates the L2 cache** with Y1 data
2. **Phase 2**: Down-proj reads Y1 → hits **L2 cache** (not cold HBM)
   - L2 bandwidth ≈ 12 TB/s vs HBM ≈ 3.35 TB/s → **3.6x faster** for Y1 read

### Performance Estimates

| GPU | L2 Size | Y1 L2 Fit (T≤) | Phase 1 Speedup |
|---|:---:|:---:|:---:|
| H100 | 50 MB | T ≈ 6400 | ~1.5-2.5% |
| B200 | 72 MB | T ≈ 9200 | ~2-3% |

### Files

| File | Description |
|---|---|
| [megakernel_kernel.py](../sonicmoe/functional/megakernel_kernel.py) | Two-phase kernel architecture |
| [megakernel_forward.py](../sonicmoe/functional/megakernel_forward.py) | Forward/backward custom ops and public API |
| [megakernel_blackwell_test.py](../tests/megakernel_blackwell_test.py) | Correctness tests and benchmark suite |

---

## Phase 2: True SMEM Fusion (🔜 Planned)

### How It Will Work

Instead of Phase 1 writing Y1 to HBM, the kernel writes Y1 to a **SMEM staging buffer**. Phase 2 reads Y1 directly from SMEM:

```
Phase 1 epilogue: SwiGLU(acc) → Y1 → SMEM buffer (stays on-chip!)
Phase 2 mainloop: Read Y1 from SMEM → WGMMA with W2
```

### Technical Challenge: SMEM Budget

H100 has **232 KB SMEM per SM**. A single GEMM kernel uses ~164 KB. We solve this with **time-multiplexed SMEM**: Phase 1 uses sA for X input + sB for W1. After Phase 1 completes, Phase 2 reuses sA for Y1 intermediate + sB for W2.

### Additional Performance

| GPU | SMEM Fusion Speedup | Total (Phase 1 + 2) |
|---|:---:|:---:|
| H100 | +1.5-2.5% | **~3-5%** |
| B200 | +2-3% | **~4-6%** |

---

## Blackwell Architecture Scaffolding (✅ Implemented)

| Feature | Description | File |
|---|---|---|
| `BlackwellArch` constants | Accurate SM100 specs (TMEM=256KB, L2=72MB, 192 SMs) | `grouped_gemm_blackwell.py` |
| `L2_group_size=16` | Larger L2 → better tile scheduling (vs Hopper's 8) | `grouped_gemm_blackwell.py` |
| Runtime dispatch | Auto-detect GPU architecture and select config | `grouped_gemm_blackwell.py` |
| Wrapper classes | Ready for native `tcgen05.mma` when CuTeDSL supports SM100 | `grouped_gemm_blackwell.py` |

---

## Testing

See [GPU_TESTING_GUIDE.md](GPU_TESTING_GUIDE.md) for full instructions and a results template.

```bash
# Quick validation (no GPU needed)
pytest tests/megakernel_blackwell_test.py -v -k "config or arch"

# Correctness + benchmarks (H100/B200 required)
pytest tests/megakernel_blackwell_test.py -v -s
```
