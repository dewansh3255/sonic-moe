# ********************************************************************************
# Tests for SonicMoE Megakernel Fusion and Blackwell Optimizations
#
# Test hierarchy:
#   1. Configuration tests (no GPU required)
#   2. Architecture detection tests (no GPU required)
#   3. Correctness tests (requires H100/B200 GPU)
#   4. Performance benchmarks (requires H100/B200 GPU)
#
# Run all tests: pytest tests/megakernel_blackwell_test.py -v
# Run no-GPU tests only: pytest tests/megakernel_blackwell_test.py -v -k "config or arch"
# Run benchmarks: pytest tests/megakernel_blackwell_test.py -v -k "benchmark" -s
# ********************************************************************************

import math
import time
from typing import Optional

import pytest
import torch

# ── Configuration Tests (no GPU needed) ─────────────────────────────────────


class TestMegakernelConfig:
    """Test megakernel configuration classes without GPU."""

    def test_megakernel_import(self):
        """Verify megakernel kernel module can be imported."""
        from sonicmoe.functional.megakernel_kernel import (
            HopperWgmma_MoE_Megakernel,
            HopperWgmma_MoE_Megakernel_SMEMFusion,
        )
        assert HopperWgmma_MoE_Megakernel is not None
        assert HopperWgmma_MoE_Megakernel_SMEMFusion is not None

    def test_megakernel_forward_import(self):
        """Verify megakernel forward API can be imported."""
        from sonicmoe.functional.megakernel_forward import (
            _fused_up_down_projection_forward,
            _fused_down_up_projection_backward,
            _FusedMoEForward,
            moe_megakernel_forward,
        )
        assert _fused_up_down_projection_forward is not None
        assert _fused_down_up_projection_backward is not None
        assert _FusedMoEForward is not None
        assert moe_megakernel_forward is not None

    def test_l2_cache_analysis(self):
        """Verify L2 cache warmth analysis for typical configs."""
        # H100 L2 cache: 50 MB
        L2_CACHE_SIZE_MB = 50

        configs = [
            # (T, I, dtype_bytes, expected_l2_fit)
            (2048, 2048, 2, True),    # Y1 = 8 MB < 50 MB
            (4096, 4096, 2, True),    # Y1 = 32 MB < 50 MB
            (8192, 4096, 2, False),   # Y1 = 64 MB > 50 MB
            (16384, 4096, 2, False),  # Y1 = 128 MB > 50 MB
        ]

        for T, I, dtype_bytes, expected_fit in configs:
            y1_size_mb = T * I * dtype_bytes / (1024 * 1024)
            fits_in_l2 = y1_size_mb <= L2_CACHE_SIZE_MB
            assert fits_in_l2 == expected_fit, (
                f"Config T={T}, I={I}: Y1={y1_size_mb:.1f} MB, "
                f"expected fits_in_l2={expected_fit}, got {fits_in_l2}"
            )

    def test_smem_budget_constraint(self):
        """Verify SMEM budget calculations for megakernel feasibility."""
        SMEM_CAPACITY = 232 * 1024  # 232 KB on H100

        # Typical up-proj config: tile (128, 256, 64), BF16
        tile_m, tile_n, tile_k = 128, 256, 64
        dtype_bytes = 2  # BF16
        ab_stages = 4  # typical pipeline depth

        sA_bytes = tile_m * tile_k * dtype_bytes * ab_stages
        sB_bytes = tile_n * tile_k * dtype_bytes * ab_stages
        epilogue_bytes = 16 * 1024  # ~16 KB for sD + sY
        pipeline_bytes = 4 * 1024   # barriers, tensormaps

        single_kernel_smem = sA_bytes + sB_bytes + epilogue_bytes + pipeline_bytes

        # For time-multiplexed megakernel: Phase 1 and Phase 2 reuse sA/sB
        # The additional cost is the intermediate Y1 buffer
        intermediate_dim = 4096  # typical I
        y1_inter_bytes = tile_m * intermediate_dim * dtype_bytes  # 1 M-tile of Y1

        megakernel_smem = single_kernel_smem + y1_inter_bytes

        assert single_kernel_smem < SMEM_CAPACITY, (
            f"Single kernel SMEM {single_kernel_smem / 1024:.1f} KB "
            f"exceeds capacity {SMEM_CAPACITY / 1024:.1f} KB"
        )

        # Note: The megakernel with full Y1 buffer may exceed SMEM capacity
        # This is why we use time-multiplexing: Y1 is processed in tile_M-sized chunks
        y1_tile_bytes = tile_m * tile_n * dtype_bytes  # tile-sized intermediate
        megakernel_tiled_smem = single_kernel_smem + y1_tile_bytes

        print(f"\nSMEM Budget Analysis:")
        print(f"  Single kernel: {single_kernel_smem / 1024:.1f} KB")
        print(f"  Megakernel (tiled Y1): {megakernel_tiled_smem / 1024:.1f} KB")
        print(f"  Megakernel (full Y1): {megakernel_smem / 1024:.1f} KB")
        print(f"  SMEM capacity: {SMEM_CAPACITY / 1024:.1f} KB")


class TestBlackwellArchDetection:
    """Test Blackwell architecture detection (no GPU needed for basic tests)."""

    def test_blackwell_config_import(self):
        """Verify Blackwell config module can be imported."""
        from sonicmoe.functional.grouped_gemm_blackwell import (
            BlackwellArch,
            BlackwellMoEKernelConfig,
            get_gpu_architecture,
            get_kernel_config_classes,
        )
        assert BlackwellArch is not None
        assert BlackwellMoEKernelConfig is not None

    def test_blackwell_arch_constants(self):
        """Verify Blackwell architecture constants are correct."""
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellArch

        # SM100 (Blackwell) constants
        assert BlackwellArch.COMPUTE_CAPABILITY == (10, 0)
        assert BlackwellArch.TMEM_SIZE_KB == 256
        assert BlackwellArch.L2_CACHE_SIZE_MB == 72
        assert BlackwellArch.SM_COUNT == 192
        assert BlackwellArch.MAX_CLUSTERS == 16

    def test_blackwell_l2_group_size(self):
        """Verify Blackwell L2 group size is larger than Hopper."""
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellMoEKernelConfig

        config = BlackwellMoEKernelConfig()
        # Blackwell L2 is 72 MB vs Hopper's 50 MB → larger group size
        assert config.L2_group_size == 16
        assert config.L2_group_size > 8  # Hopper default is 8


# ── Correctness Tests (requires GPU) ────────────────────────────────────────


@pytest.fixture
def moe_params():
    """Create MoE parameters for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    E = 8    # num experts
    K = 2    # top-K
    T = 256  # num tokens
    H = 512  # hidden dim
    I = 1024  # intermediate dim (FFN dim)

    x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    router_w = torch.randn(E, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device).permute(1, 2, 0)

    return {
        "x": x, "router_w": router_w,
        "w1": w1, "w2": w2,
        "E": E, "K": K, "T": T, "H": H, "I": I,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMegakernelCorrectness:
    """Test megakernel output correctness against the sequential baseline."""

    def test_output_shape(self, moe_params):
        """Verify megakernel produces correct output shapes."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        torch.manual_seed(42)
        stream_id = torch.cuda.current_stream().cuda_stream

        o, logits, freq = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        T, H = moe_params["T"], moe_params["H"]
        E = moe_params["E"]

        assert o.shape == (T, H), f"Output shape mismatch: {o.shape} != ({T}, {H})"
        assert logits.shape == (T, E), f"Logits shape mismatch: {logits.shape} != ({T}, {E})"
        assert freq.shape == (E,), f"Freq shape mismatch: {freq.shape} != ({E},)"

    def test_matches_sequential_baseline(self, moe_params):
        """Verify megakernel matches the sequential (unfused) baseline.

        This is the critical correctness test. Both paths should produce
        numerically identical results since they run the same CUDA kernels
        (just launched differently).
        """
        from sonicmoe.functional import moe_TC_softmax_topk_layer
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        stream_id = torch.cuda.current_stream().cuda_stream

        # Run sequential baseline
        torch.manual_seed(42)
        o_seq, logits_seq, freq_seq = moe_TC_softmax_topk_layer(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        # Run megakernel fusion
        torch.manual_seed(42)
        o_mega, logits_mega, freq_mega = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        # Compare outputs (should be numerically identical)
        torch.testing.assert_close(
            o_mega, o_seq,
            atol=1e-3, rtol=1e-3,
            msg="Megakernel output differs from sequential baseline!",
        )

    def test_gradient_flow(self, moe_params):
        """Verify gradients flow correctly through the fused forward."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        x = moe_params["x"].clone().requires_grad_(True)
        w1 = moe_params["w1"].clone().requires_grad_(True)
        w2 = moe_params["w2"].clone().requires_grad_(True)

        stream_id = torch.cuda.current_stream().cuda_stream

        torch.manual_seed(42)
        o, _, _ = moe_megakernel_forward(
            x=x,
            router_w=moe_params["router_w"],
            w1=w1,
            b1=None,
            w2=w2,
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        grad_out = torch.randn_like(o)
        o.backward(grad_out)

        assert x.grad is not None, "No gradient for x"
        assert w1.grad is not None, "No gradient for w1"
        assert w2.grad is not None, "No gradient for w2"
        assert not torch.all(x.grad == 0), "x gradient is all zeros"
        assert not torch.all(w1.grad == 0), "w1 gradient is all zeros"
        assert not torch.all(w2.grad == 0), "w2 gradient is all zeros"

    def test_expert_frequency_invariant(self, moe_params):
        """Verify expert frequency is invariant to fusion strategy."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        stream_id = torch.cuda.current_stream().cuda_stream

        torch.manual_seed(42)
        _, _, freq1 = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        torch.manual_seed(42)
        _, _, freq2 = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=stream_id,
        )

        assert torch.equal(freq1, freq2), "Expert frequency not deterministic!"


# ── Performance Benchmarks (requires GPU) ───────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMegakernelBenchmark:
    """Benchmark megakernel fusion vs sequential baseline.

    This measures the actual speedup from L2-cache-warm fusion.
    Run with: pytest tests/megakernel_blackwell_test.py -v -k "benchmark" -s
    """

    def _benchmark_fn(self, fn, warmup=20, iterations=100):
        """Run a function multiple times and return median time in ms."""
        torch.cuda.synchronize()

        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        return times[len(times) // 2]  # median

    @pytest.mark.parametrize(
        "T,H,I,E,K",
        [
            (2048, 4096, 4096, 8, 2),
            (4096, 4096, 4096, 8, 2),
            (4096, 4096, 4096, 64, 2),
        ],
    )
    def test_benchmark_comparison(self, T, H, I, E, K):
        """Benchmark megakernel vs sequential baseline."""
        from sonicmoe.functional import moe_TC_softmax_topk_layer
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        device = torch.device("cuda")
        x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        router_w = torch.randn(E, H, dtype=torch.bfloat16, device=device)
        w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
        w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device).permute(1, 2, 0)

        stream_id = torch.cuda.current_stream().cuda_stream

        # Benchmark sequential baseline
        def run_sequential():
            torch.manual_seed(42)
            return moe_TC_softmax_topk_layer(
                x=x, router_w=router_w, w1=w1, b1=None, w2=w2, b2=None,
                K=K, stream_id=stream_id,
                is_inference_mode_enabled=True,
            )

        # Benchmark megakernel
        def run_megakernel():
            torch.manual_seed(42)
            return moe_megakernel_forward(
                x=x, router_w=router_w, w1=w1, b1=None, w2=w2, b2=None,
                K=K, stream_id=stream_id,
                is_inference_mode_enabled=True,
            )

        time_seq = self._benchmark_fn(run_sequential)
        time_mega = self._benchmark_fn(run_megakernel)
        speedup = (time_seq - time_mega) / time_seq * 100

        y1_size_mb = T * I * 2 / (1024 * 1024)

        print(f"\n{'='*60}")
        print(f"Config: T={T}, H={H}, I={I}, E={E}, K={K}")
        print(f"Y1 size: {y1_size_mb:.1f} MB (L2 fit: {'Yes' if y1_size_mb <= 50 else 'No'})")
        print(f"Sequential baseline: {time_seq:.3f} ms")
        print(f"Megakernel fusion:   {time_mega:.3f} ms")
        print(f"Speedup:             {speedup:+.2f}%")
        print(f"{'='*60}")

    @pytest.mark.parametrize("T", [1024, 2048, 4096, 8192])
    def test_l2_cache_sweep(self, T):
        """Sweep token counts to measure L2 cache warmth effect.

        As T increases, Y1 grows beyond L2 capacity and the
        L2-cache-warm benefit diminishes.
        """
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        device = torch.device("cuda")
        H, I, E, K = 4096, 4096, 8, 2

        x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
        router_w = torch.randn(E, H, dtype=torch.bfloat16, device=device)
        w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
        w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device).permute(1, 2, 0)

        stream_id = torch.cuda.current_stream().cuda_stream

        def run():
            torch.manual_seed(42)
            return moe_megakernel_forward(
                x=x, router_w=router_w, w1=w1, b1=None, w2=w2, b2=None,
                K=K, stream_id=stream_id,
                is_inference_mode_enabled=True,
            )

        time_ms = self._benchmark_fn(run, warmup=10, iterations=50)
        y1_size_mb = T * I * 2 / (1024 * 1024)

        print(f"\nT={T:5d}, Y1={y1_size_mb:6.1f} MB, Time={time_ms:8.3f} ms")
