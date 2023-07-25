# -*- coding: utf-8 -*-

import torch

def show_gpu_usage(name:str = ""):
    if torch.cuda.is_available():
        # 获取当前设备
        device = torch.cuda.current_device()
        # 获取显存信息
        memory_info = torch.cuda.memory_stats(device=device)
        # 获取显存使用量
        memory_used = memory_info["allocated_bytes.all.current"] / 1024 ** 2
        # 获取总共显存量
        memory_total = torch.cuda.get_device_properties(device=device).total_memory / 1024 ** 2
        # 获取剩余可用显存
        memory_free = memory_total - memory_used
        if len(name) > 0:
            name = f"[{name}]: "
        else:
            name = "> "
        print(f"{name} GPU Memory usage: {memory_used:,.0f} MiB")
        print(f"{name} GPU Memory free: {memory_free:,.0f} MiB")
        return {"total": memory_total, "used": memory_used, "free": memory_free}
    else:
        print("GPU not available.")
        return {"total": 0, "used": 0, "free": 0}

from PIL import ImageColor

def lighten_color(color, amount=0.2):
    if isinstance(color, str):
        color = ImageColor.getrgb(color)

    white = 255
    adjusted_color = tuple(
        int(round(old_value * (1 - amount) + white * amount))
        for old_value in color
    )
    return adjusted_color
