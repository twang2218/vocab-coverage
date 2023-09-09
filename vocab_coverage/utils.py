# -*- coding: utf-8 -*-

import datetime
import logging
import os
import shutil
import torch
import inspect
import gc

def has_parameter(fn, parameter_name:str):
    try:
        sig = inspect.signature(fn)
        return parameter_name in sig.parameters
    except ValueError:  # Raised when a non-callable object is passed to `inspect.signature`
        return False

def show_gpu_usage(name:str = "", debug:bool=True):
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

        if debug:
            if len(name) > 0:
                name = f"[{name}]: "
            else:
                name = "> "
            logger.debug("%s GPU Memory total: %s MiB", name, f'{memory_total:,.0f}')
            logger.debug("%s GPU Memory usage: %s MiB", name, f'{memory_used:,.0f}')
            logger.debug("%s GPU Memory free: %s MiB", name, f'{memory_free:,.0f}')
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

def release_resource(model_name:str = "", clear_cache=False, debug:bool=True):
    gc.collect()
    if len(model_name) > 0:
        label = f"[{model_name}]: "
    if torch.cuda.is_available():
        if debug:
            logger.debug("%sreleasing GPU memory...", label)
        torch.cuda.empty_cache()
        show_gpu_usage(model_name, debug=debug)
    if clear_cache:
        model_path = f"models--{model_name.replace('/', '--')}"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub", model_path)
        if debug:
            logger.debug("%sclean up cache (%s)...", label, cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

def get_logger():
    l = logging.getLogger(__package__)
    l.setLevel(logging.DEBUG)
    # 创建StreamHandler处理器，用于输出到stderr
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    l.addHandler(ch)
    # 创建FileHandler处理器，用于输出到文件
    # 获取当前日期，生成日期字符串，用于日志文件名
    date_str = datetime.datetime.now().strftime('%Y-%m-%d.%H-%M')
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{__package__}_{date_str}.log"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)
    return l

logger = get_logger()

def _prepare_folder(filename:str):
    basedir = os.path.dirname(filename)
    if len(basedir) == 0:
        return
    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

def generate_model_path(model_name:str):
    basename = model_name.replace("/", "_")
    return basename

def generate_embedding_filename(model_name:str, granularity:str, position:str, postfix:str='', folder:str=''):
    # 根据参数生成文件基础文件名
    output_file = f'{generate_model_path(model_name)}.embedding.{granularity}.{position}.jpg'
    if len(postfix) > 0:
        output_file = output_file.replace('.jpg', f'.{postfix}.jpg')
    filename = os.path.join(folder, output_file)
    _prepare_folder(filename)
    return filename

def generate_coverage_filename(model_name:str, granularity:str, postfix:str='', folder:str=''):
    # 生成文件名
    filename = f'{generate_model_path(model_name)}.coverage.{granularity}.jpg'
    if len(postfix) > 0:
        filename = filename.replace('.jpg', f'.{postfix}.jpg')
    filename = os.path.join(folder, filename)
    _prepare_folder(filename)
    return filename

def generate_thumbnail_filename(filename:str, folder:str=''):
    if len(folder) > 0:
        # thumbnail is in a different folder
        basename = os.path.basename(filename)
        filename = os.path.join(folder, basename)
    filename = filename.replace(".jpg", ".thumbnail.jpg")
    _prepare_folder(filename)
    return filename

def is_match_patterns(model_name:str, patterns:list[str], ignore_case:bool=True) -> bool:
    if ignore_case:
        model_name = model_name.lower()
    for pattern in patterns:
        if ignore_case:
            pattern = pattern.lower()
        if pattern in model_name:
            return True
    return False

def get_cmap(name):
    import matplotlib as mpl
    import cmaps
    if 'cmaps:' in name:
        name = name.split(':')[1]
        return getattr(cmaps, name)
    else:
        return mpl.colormaps[name]

def get_colors(n, colormap='gist_rainbow'):
    cm = get_cmap(colormap)
    return [cm(i/n, bytes=True) for i in range(n)]

def get_colors_hex(n, colormap='gist_rainbow'):
    import matplotlib as mpl
    cm = get_cmap(colormap)
    return [mpl.colors.to_hex(cm(i/n)) for i in range(n)]
