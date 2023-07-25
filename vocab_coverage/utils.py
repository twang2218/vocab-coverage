# -*- coding: utf-8 -*-

import os
import shutil
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

def release_resource(model_name:str = "", clear_cache=False):
    if len(model_name) > 0:
        label = f"[{model_name}]: "
    if torch.cuda.is_available():
        print(f"{label}releasing GPU memory...")
        torch.cuda.empty_cache()
        show_gpu_usage(model_name)
    if clear_cache:
        model_path = f"models--{model_name.replace('/', '--')}"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub", model_path)
        print(f"{label}clean up cache ({cache_dir})...")
        shutil.rmtree(cache_dir, ignore_errors=True)

