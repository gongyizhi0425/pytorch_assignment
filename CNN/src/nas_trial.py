import torch
import torch.nn as nn
from thop import profile
import numpy as np
import torchvision.models as models

def measure_model_metrics(model, input_res):
    """测量模型的 FLOPs, 参数量和峰值激活内存"""
    dummy_input = torch.randn(1, 3, input_res, input_res)
    
    # 1. 计算 FLOPs (使用 thop)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 2. 估算峰值激活内存 (Peak Activation Memory)
    # 遍历网络层，找到产生最大输出张量的那一层
    max_activation_size = 0
    
    def hook_fn(module, input, output):
        nonlocal max_activation_size
        if isinstance(output, torch.Tensor):
            # batch_size * channels * height * width * 4 bytes (float32)
            mem_bytes = output.numel() * 4 
            if mem_bytes > max_activation_size:
                max_activation_size = mem_bytes
                
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6)):
            hooks.append(layer.register_forward_hook(hook_fn))
            
    # 前向传播以触发 hooks
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    # 将内存转换为 MB
    peak_mem_mb = max_activation_size / (1024 * 1024)
    flops_m = flops / 1e6 # 转换为 Mega FLOPs
    
    return flops_m, peak_mem_mb




def build_dynamic_mobilenet(width_mult=1.0, num_classes=10):
    """
    这里以修改 width 为例。
    如果是高阶作业，你需要重写 InvertedResidual 模块来支持变动的 Kernel Size 
    和重复次数 (Depth)。
    """
    # 使用 torchvision 的内置参数修改宽度
    model = models.mobilenet_v2(width_mult=width_mult)
    # 修改最后的分类头适配 CIFAR-10
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model