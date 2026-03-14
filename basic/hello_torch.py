# hello_pytorch.py
import torch

print("=" * 60)
print("🚀 欢迎使用 PyTorch GPU 环境！")
print("=" * 60)

# 1. 基础信息
print(f"✓ PyTorch 版本: {torch.__version__}")
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")

# 2. GPU 信息（如果可用）
if torch.cuda.is_available():
    print(f"✓ GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"✓ 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 3. 创建张量并移至 GPU
x = torch.tensor([1.0, 2.0, 3.0]).cuda()  # .cuda() = 使用 GPU
y = x * 2 + 1
print(f"\n✓ GPU 张量计算: y = 2x + 1")
print(f"  x = {x}")
print(f"  y = {y}")

# 4. 自动求导演示
z = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
loss = (z ** 2).sum()  # loss = z1² + z2² + z3²
loss.backward()
print(f"\n✓ 自动求导验证:")
print(f"  z = {z}")
print(f"  loss = {loss.item():.2f}")
print(f"  dz/dloss = {z.grad}")

print("=" * 60)
print("🎉 恭喜！PyTorch GPU 环境工作正常")
print("=" * 60)