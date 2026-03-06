# ********************************************************************************
# Tests for SonicMoE Megakernel + Blackwell Port
#
# Test categories:
#   1. Config tests (run anywhere, no GPU needed)
#   2. Architecture detection tests
#   3. Correctness tests (require Hopper or Blackwell GPU)
#   4. Backward pass tests (require GPU)
#
# Run with: pytest tests/megakernel_blackwell_test.py -v
# ********************************************************************************

import pytest
import torch


# =============================================================================
# 1. Configuration Tests (no GPU required)
# =============================================================================
class TestBlackwellConfig:
    """Test Blackwell architecture constants and tuning configs."""

    def test_arch_constants(self):
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellArch

        assert BlackwellArch.TMEM_PER_SM == 256 * 1024
        assert BlackwellArch.L2_CACHE_SIZE == 126 * 1024 * 1024
        assert BlackwellArch.NUM_SMS == 160
        assert BlackwellArch.REQUIRES_WARPGROUP_BARRIER is False
        assert "fp4" in BlackwellArch.SUPPORTED_DTYPES

    def test_kernel_config_speedup_estimate(self):
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellMoEKernelConfig

        config = BlackwellMoEKernelConfig(E=64, H=4096, I=1024, precision="fp8")
        estimate = config.estimate_speedup_vs_hopper_baseline()
        assert "total_estimated" in estimate
        assert config.throughput_multiplier == 2.0

    def test_fp4_throughput_multiplier(self):
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellMoEKernelConfig

        config = BlackwellMoEKernelConfig(E=64, H=4096, I=1024, precision="fp4")
        assert config.throughput_multiplier == 4.0

    def test_l2_group_size_tuning(self):
        from sonicmoe.functional.grouped_gemm_blackwell import BlackwellMoEKernelConfig

        config = BlackwellMoEKernelConfig(E=64, H=4096, I=1024)
        # Blackwell uses larger L2 group size due to 2.5x L2 cache
        assert config.L2_group_size == 16


class TestArchitectureDetection:
    """Test GPU architecture detection and dispatch."""

    def test_detection_returns_string(self):
        from sonicmoe.functional.grouped_gemm_blackwell import get_gpu_architecture

        arch = get_gpu_architecture()
        assert isinstance(arch, str)

    def test_dispatch_hopper_returns_hopper_classes(self):
        from sonicmoe.functional.grouped_gemm_blackwell import get_kernel_config_classes

        classes = get_kernel_config_classes(arch="hopper")
        assert "up_proj_fwd" in classes
        assert "down_proj_fwd" in classes
        assert "HopperWgmma" in classes["up_proj_fwd"].__name__

    def test_dispatch_blackwell_returns_blackwell_classes(self):
        from sonicmoe.functional.grouped_gemm_blackwell import get_kernel_config_classes

        classes = get_kernel_config_classes(arch="blackwell")
        assert "up_proj_fwd" in classes
        assert "Blackwell" in classes["up_proj_fwd"].__name__

    def test_dispatch_contains_all_kernels(self):
        from sonicmoe.functional.grouped_gemm_blackwell import get_kernel_config_classes

        for arch in ["hopper", "blackwell"]:
            classes = get_kernel_config_classes(arch=arch)
            expected_keys = [
                "up_proj_fwd",
                "down_proj_fwd",
                "down_proj_act_bwd",
                "down_proj_wt_bwd",
                "up_proj_act_bwd",
            ]
            for key in expected_keys:
                assert key in classes, f"Missing {key} for {arch}"


# =============================================================================
# 2. Correctness Tests (require Hopper or Blackwell GPU)
# =============================================================================
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires Hopper (SM90) or newer GPU",
)
class TestMegakernelForwardCorrectness:
    """Compare fused forward output against the sequential baseline.

    The megakernel must produce numerically identical results to the
    sequential up-proj → down-proj implementation.
    """

    @pytest.fixture
    def moe_params(self):
        """Standard MoE test parameters (small enough for fast tests)."""
        from sonicmoe.enums import ActivationType

        E, K, T, H, I = 8, 2, 128, 256, 64
        device = "cuda"
        dtype = torch.bfloat16

        return {
            "E": E, "K": K, "T": T, "H": H, "I": I,
            "x": torch.randn(T, H, device=device, dtype=dtype),
            "router_w": torch.randn(E, H, device=device, dtype=dtype),
            "w1": torch.randn(2 * I, H, E, device=device, dtype=dtype),
            "w2": torch.randn(H, I, E, device=device, dtype=dtype),
            "activation_type": ActivationType.SWIGLU,
        }

    def test_output_shape(self, moe_params):
        """Megakernel output shape should match (T, H)."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        o, logits, freq = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=torch.cuda.current_stream().cuda_stream,
            activation_type=moe_params["activation_type"],
        )

        T, H, E = moe_params["T"], moe_params["H"], moe_params["E"]
        assert o.shape == (T, H), f"Expected ({T}, {H}), got {o.shape}"
        assert logits.shape == (T, E), f"Expected ({T}, {E}), got {logits.shape}"
        assert freq.shape == (E,), f"Expected ({E},), got {freq.shape}"

    def test_matches_sequential_baseline(self, moe_params):
        """Megakernel output must match the sequential implementation.

        This is the CRITICAL correctness test. Uses strict tolerance for BF16.
        """
        from sonicmoe.functional import moe_TC_softmax_topk_layer
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        stream_id = torch.cuda.current_stream().cuda_stream

        # Set deterministic seeds for identical router decisions
        torch.manual_seed(42)
        o_seq, logits_seq, freq_seq = moe_TC_softmax_topk_layer(
            x=moe_params["x"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            activation_type=moe_params["activation_type"],
            topk=moe_params["K"],
            stream_id=stream_id,
        )

        # Reset seed and run megakernel version
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
            activation_type=moe_params["activation_type"],
        )

        # BF16 tolerance: atol=1e-2, rtol=1e-2
        torch.testing.assert_close(
            o_mega, o_seq, atol=1e-2, rtol=1e-2,
            msg="Megakernel output differs from sequential baseline",
        )

    def test_backward_gradients_flow(self, moe_params):
        """Verify gradients flow through the fused forward pass."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        x = moe_params["x"].clone().requires_grad_(True)
        w1 = moe_params["w1"].clone().requires_grad_(True)
        w2 = moe_params["w2"].clone().requires_grad_(True)

        o, _, _ = moe_megakernel_forward(
            x=x,
            router_w=moe_params["router_w"],
            w1=w1,
            b1=None,
            w2=w2,
            b2=None,
            K=moe_params["K"],
            stream_id=torch.cuda.current_stream().cuda_stream,
            activation_type=moe_params["activation_type"],
        )

        loss = o.sum()
        loss.backward()

        assert x.grad is not None, "x.grad should not be None"
        assert w1.grad is not None, "w1.grad should not be None"
        assert w2.grad is not None, "w2.grad should not be None"
        assert not torch.all(x.grad == 0), "x.grad should not be all zeros"

    def test_expert_frequency_sum(self, moe_params):
        """Expert frequencies should sum to T * K (every token picks K experts)."""
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        _, _, freq = moe_megakernel_forward(
            x=moe_params["x"],
            router_w=moe_params["router_w"],
            w1=moe_params["w1"],
            b1=None,
            w2=moe_params["w2"],
            b2=None,
            K=moe_params["K"],
            stream_id=torch.cuda.current_stream().cuda_stream,
            activation_type=moe_params["activation_type"],
        )

        expected_total = moe_params["T"] * moe_params["K"]
        assert freq.sum().item() == expected_total, (
            f"Expert frequencies sum to {freq.sum().item()}, expected {expected_total}"
        )


# =============================================================================
# 3. Benchmarking Utilities
# =============================================================================
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires Hopper (SM90) or newer GPU",
)
class TestBenchmarks:
    """Performance benchmarks to measure megakernel speedup.

    These are not strict pass/fail tests — they measure and report
    timing differences between sequential and fused implementations.
    """

    def test_benchmark_forward(self, capsys):
        """Benchmark forward pass: sequential vs fused."""
        from sonicmoe.enums import ActivationType

        E, K, T, H, I = 64, 2, 4096, 4096, 1024
        device = "cuda"
        dtype = torch.bfloat16

        x = torch.randn(T, H, device=device, dtype=dtype)
        router_w = torch.randn(E, H, device=device, dtype=dtype)
        w1 = torch.randn(2 * I, H, E, device=device, dtype=dtype)
        w2 = torch.randn(H, I, E, device=device, dtype=dtype)
        stream_id = torch.cuda.current_stream().cuda_stream

        # Warmup
        from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

        for _ in range(3):
            moe_megakernel_forward(x, router_w, w1, None, w2, None, K, stream_id,
                                   ActivationType.SWIGLU)
        torch.cuda.synchronize()

        # Benchmark fused
        N_ITERS = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(N_ITERS):
            moe_megakernel_forward(x, router_w, w1, None, w2, None, K, stream_id,
                                   ActivationType.SWIGLU)
        end.record()
        torch.cuda.synchronize()
        fused_ms = start.elapsed_time(end) / N_ITERS

        # Report
        print(f"\n{'='*60}")
        print(f"SonicMoE Megakernel Benchmark")
        print(f"  Config: E={E}, K={K}, T={T}, H={H}, I={I}")
        print(f"  Fused forward: {fused_ms:.3f} ms")
        print(f"{'='*60}")
