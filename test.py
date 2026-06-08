import torch
import time

device = "cuda"

x = torch.rand(10000, 10000).to(device)
y = torch.rand(10000, 10000).to(device)

start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize()

print("Temps:", time.time() - start)
print(z.device)