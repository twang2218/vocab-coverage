# -*- coding: utf-8 -*-

import datetime
import logging
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
        logger.debug(f"{name} GPU Memory usage: {memory_used:,.0f} MiB")
        logger.debug(f"{name} GPU Memory free: {memory_free:,.0f} MiB")
        return {"total": memory_total, "used": memory_used, "free": memory_free}
    else:
        logger.debug("GPU not available.")
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
        logger.debug(f"{label}releasing GPU memory...")
        torch.cuda.empty_cache()
        show_gpu_usage(model_name)
    if clear_cache:
        model_path = f"models--{model_name.replace('/', '--')}"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub", model_path)
        logger.debug(f"{label}clean up cache ({cache_dir})...")
        shutil.rmtree(cache_dir, ignore_errors=True)

def get_logger():
    logger = logging.getLogger(__package__)
    logger.setLevel(logging.DEBUG)
    # 创建StreamHandler处理器，用于输出到stderr
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 创建FileHandler处理器，用于输出到文件
    # 获取当前日期，生成日期字符串，用于日志文件名
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    log_file = f"{__package__}_{date_str}.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()
