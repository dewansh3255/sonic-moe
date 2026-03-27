import torch
import time
from sonicmoe.functional import moe_TC_softmax_topk_layer
from sonicmoe.functional.megakernel_forward import moe_megakernel_forward

T, H, I, E, K = 4096, 4096, 4096, 8, 2
x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda')
router_w = torch.randn(E, H, dtype=torch.bfloat16, device='cuda')
w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda').permute(1, 2, 0)
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda').permute(1, 2, 0)

stream_id = torch.cuda.current_stream().cuda_stream

def run_seq():
    return moe_TC_softmax_topk_layer(x, router_w, w1, None, w2, None, K, stream_id, is_inference_mode_enabled=True)

def run_mega():
    return moe_megakernel_forward(x, router_w, w1, None, w2, None, K, stream_id, is_inference_mode_enabled=True)

# Warmup
for _ in range(10): run_seq()
for _ in range(10): run_mega()
torch.cuda.synchronize()

st = time.time()
for _ in range(50): run_seq()
torch.cuda.synchronize()
time_seq = (time.time() - st) * 1000 / 50

st = time.time()
for _ in range(50): run_mega()
torch.cuda.synchronize()
time_mega = (time.time() - st) * 1000 / 50

print(f"Sequential: {time_seq:.3f} ms")
print(f"Megakernel: {time_mega:.3f} ms")
print(f"Speedup: {(time_seq - time_mega)/time_seq*100:.2f}%")
