# SonicMoE Optimizations: Megakernel Fusion & Blackwell Port

This document describes the optimizations added to SonicMoE as part of ongoing research to improve MoE kernel performance. These changes are additive — **no original SonicMoE files were modified**.

## Overview

Two optimizations were implemented:

1. **Megakernel Fusion** — Fuses up-proj + down-proj into a single `torch.library.custom_op`
2. **Blackwell Architecture Tuning** — Auto-detects SM100 GPUs and applies optimized L2/tile parameters

## New Files

| File | Description |
|------|-------------|
| `sonicmoe/functional/megakernel_forward.py` | Fused forward/backward pass |
| `sonicmoe/functional/grouped_gemm_blackwell.py` | Blackwell-tuned kernel wrappers |
| `tests/megakernel_blackwell_test.py` | Config, correctness, and benchmark tests |

## Quick Start

### Installation
```bash
pip install -e .
```

### Using the Megakernel
```python
from sonicmoe.functional.megakernel_forward import moe_megakernel_forward
from sonicmoe.enums import ActivationType

# Drop-in replacement for moe_TC_softmax_topk_layer()
output, router_logits, expert_freq = moe_megakernel_forward(
    x=x,                              # (T, H) input hidden states
    router_w=router_weight,            # (E, H) router weights
    w1=w1,                             # (2*I, H, E) up-proj weights
    b1=None,                           # optional bias
    w2=w2,                             # (H, I, E) down-proj weights
    b2=None,                           # optional bias
    K=8,                               # top-K experts per token
    stream_id=torch.cuda.current_stream().cuda_stream,
    activation_type=ActivationType.SWIGLU,
)
```

### Using Blackwell Detection
```python
from sonicmoe.functional.grouped_gemm_blackwell import (
    get_gpu_architecture,
    get_kernel_config_classes,
)

# Auto-detect GPU and get appropriate kernel classes
arch = get_gpu_architecture()  # "hopper" or "blackwell"
kernels = get_kernel_config_classes()  # auto-picks best kernels
```

### Running Tests
```bash
# Config tests (no GPU needed)
pytest tests/megakernel_blackwell_test.py -v -k "Config or Architecture"

# Full test suite (requires Hopper/Blackwell GPU)
pytest tests/megakernel_blackwell_test.py -v

# Benchmarks with timing output
pytest tests/megakernel_blackwell_test.py -v -k "benchmark" -s
```

## Technical Details

### Megakernel Fusion

The original SonicMoE forward pass launches up-proj and down-proj as separate CUDA kernels:

```
_up_projection_forward()   → CUDA kernel 1
_down_projection_forward() → CUDA kernel 2
```

Our `_fused_up_down_projection_forward` combines both into one `torch.library.custom_op`:

```
_fused_up_down_projection_forward() → CUDA kernel 1 + CUDA kernel 2
```

**Key savings:**
- Eliminates 1 Python→CUDA launch overhead (~5-10μs)
- Shares expert metadata (expert_frequency_offset, x_gather_idx) between phases
- CUDA driver can pipeline W2 weight loads with Phase 1 compute
- Backward pass similarly fused: 4 calls → 1 call

### Blackwell Tuning

The Blackwell wrappers use the same Hopper WGMMA kernels but with SM100-optimized parameters:

- `L2_group_size = 16` (vs 8 on Hopper) — exploits 2.5x larger L2 cache (126 MB vs 50 MB)
- `max_active_clusters = 160` (vs 132) — uses all Blackwell SMs

When CuTeDSL adds native SM100 support (`tcgen05.mma` + TMEM), the wrappers will be updated to use native Blackwell instructions for additional 15-30% gains.

## Expected Performance Improvements

### On Hopper (H100)
| Source | Estimated Gain |
|---|:---:|
| Kernel launch elimination | 0.3-0.5% |
| Expert metadata sharing | 0.2-0.3% |
| CUDA driver pipelining | 0.5-1.0% |
| Fused backward | 0.3-0.5% |
| **Total** | **1.3-2.3%** |

### On Blackwell (B200) vs Hopper (H100)
| Source | Estimated Gain |
|---|:---:|
| HBM bandwidth 2.4x | 5-10% |
| L2 cache 2.5x | 2-5% |
| SM count 1.2x | 5-10% |
| L2 tile tuning | 1-3% |
| **Total** | **13-28%** |
