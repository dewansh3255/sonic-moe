#!/usr/bin/env python3
"""
SMEM Budget Validation Utility for the A-Tensor Fused Kernel (O1).

Validates whether the fused up-proj + down-proj kernel fits within 
Hopper's 227KB SMEM budget using temporal SMEM reuse between phases.

Usage:
    python tools/check_smem_budget.py
"""


def check_fused_kernel_smem_budget(
    n: int,
    d: int,
    Mtile: int = 128,
    tile_K: int = 64,
    ab_stages: int = 2,
    w2_stages: int = 2,
    glu: bool = True,
) -> dict:
    """
    Check if the A-tensor fused kernel fits in 227KB Hopper SMEM.
    Uses temporal SMEM reuse: Phase 1 SMEM is freed before Phase 2.
    """
    SMEM_CAP = 227 * 1024  # bytes (Hopper sm_90a)
    OVERHEAD = 8 * 1024  # mbarriers, tensormap management
    elem_size = 2  # BF16

    gate_factor = 2 if glu else 1  # SwiGLU uses 2n pre-activation

    # Phase 1 peak SMEM (up-proj mainloop)
    a_pipe = ab_stages * Mtile * tile_K * elem_size  # X tiles
    b_pipe = ab_stages * (gate_factor * n) * tile_K * elem_size  # W1 tiles
    z_epi = 2 * Mtile * tile_K * elem_size  # z epilogue buffer (2-stage)
    phase1_peak = a_pipe + b_pipe + z_epi + OVERHEAD

    # Phase 2 peak SMEM (down-proj inner loop)
    # W1 pipeline slots freed and repurposed for W2
    sA_fused = Mtile * n * elem_size  # persistent y1 slot
    w2_pipe = w2_stages * n * tile_K * elem_size  # W2 tiles (reuse a_pipe slots)
    y2_epi = 2 * Mtile * tile_K * elem_size  # y2 epilogue buffer
    phase2_peak = sA_fused + w2_pipe + y2_epi + OVERHEAD

    peak = max(phase1_peak, phase2_peak)

    status = "✓ OK" if peak <= SMEM_CAP else "✗ EXCEEDS"
    print(
        f"n={n:4d}, d={d:4d}, Mtile={Mtile}: "
        f"Phase1={phase1_peak / 1024:.0f}KB, Phase2={phase2_peak / 1024:.0f}KB, "
        f"Peak={peak / 1024:.0f}KB / 227KB → {status}"
    )
    return {
        "n": n,
        "d": d,
        "Mtile": Mtile,
        "phase1_kb": phase1_peak / 1024,
        "phase2_kb": phase2_peak / 1024,
        "peak_kb": peak / 1024,
        "feasible": peak <= SMEM_CAP,
    }


def bandwidth_savings(T: int, K: int, n: int, hbm_bw_tbs: float = 3.35) -> dict:
    """Calculate HBM bandwidth savings from eliminating y1 round-trip."""
    y1_bytes = T * K * n * 2  # BF16
    total_eliminated = 2 * y1_bytes  # write + read
    time_saved_us = (total_eliminated / (hbm_bw_tbs * 1e12)) * 1e6
    return {
        "y1_size_mb": y1_bytes / 1e6,
        "total_eliminated_mb": total_eliminated / 1e6,
        "time_saved_us": time_saved_us,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SMEM Budget Validation for Fused Up-Down Projection Kernel")
    print("=" * 70)

    print("\n--- Standard Mtile=128 configs ---")
    for n in [64, 128, 256, 512]:
        for d in [1536, 4096]:
            check_fused_kernel_smem_budget(n=n, d=d, Mtile=128)

    print("\n--- n=512 with reduced Mtile=64 ---")
    for d in [1536, 4096]:
        check_fused_kernel_smem_budget(n=512, d=d, Mtile=64)

    print("\n--- Bandwidth Savings (7B config: T=24576, K=8, n=256) ---")
    savings = bandwidth_savings(T=24576, K=8, n=256)
    print(f"  y1 size: {savings['y1_size_mb']:.1f} MB")
    print(f"  HBM traffic eliminated: {savings['total_eliminated_mb']:.1f} MB")
    print(f"  Time saved: {savings['time_saved_us']:.1f} µs")

    print("\n--- Bandwidth Savings (30B config: T=32768, K=16, n=256) ---")
    savings = bandwidth_savings(T=32768, K=16, n=256)
    print(f"  y1 size: {savings['y1_size_mb']:.1f} MB")
    print(f"  HBM traffic eliminated: {savings['total_eliminated_mb']:.1f} MB")
    print(f"  Time saved: {savings['time_saved_us']:.1f} µs")
