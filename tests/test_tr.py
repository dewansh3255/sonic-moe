import torch
from sonicmoe.functional import moe_megakernel_forward
from sonicmoe.enums import ActivationType

T, H, E, K = 32, 64, 8, 2
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
router_w = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)

w1 = torch.randn(128, H, E, device="cuda", dtype=torch.bfloat16)
w2 = torch.randn(H, 64, E, device="cuda", dtype=torch.bfloat16)

# Test top_k
out_topk, _, _ = moe_megakernel_forward(
    x, router_w, w1, None, w2, None, K, torch.cuda.current_stream().cuda_stream, routing="top_k", Mtile=128
)
print("Top-K success!")

# Test TR
out_nr, _, _ = moe_megakernel_forward(
    x, router_w, w1, None, w2, None, K, torch.cuda.current_stream().cuda_stream, routing="nr", Mtile=128
)
print("Token Rounding success!")
