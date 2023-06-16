# -*- coding: utf-8 -*-

import random
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont
default_palette = [
    '#B04759', '#E76161','#F99B7D',
    '#146C94', '#19A7CE', '#E893CF',
]
font_paths = [
    "STHeiti Light",
    "微软雅黑",
]
for font_path in font_paths:
    try:
        # 尝试加载字体文件
        zh_font = ImageFont.truetype(font_path, size=25)
        break  # 如果加载成功，跳出循环
    except IOError:
        continue  # 如果加载失败，继续尝试下一个字体

def draw_vocab_graph(model_name: str, charset_stats:dict, vocab_size:int, filename: str, width=100, height=100, cell_size=10, margin=40, palette=default_palette):
    total_chars = sum([s['total'] for s in charset_stats.values()])

    # 定义图像大小
    image_width = width * cell_size + 1 + margin * 2
    height = total_chars // width + 1
    image_height = height * cell_size + 1 + margin * 6

    # 创建新的空白图像
    image = Image.new("RGBA", (image_width, image_height), "#EEEEEE")

    # 获取图像的像素访问对象
    pixels = image.load()

    grid_color = (255,255,255, 40)

    # 根据map绘制栅格
    draw = ImageDraw.Draw(image)
    i = 0
    level = 0
    for name, stats in charset_stats.items():
        for j, m in enumerate(stats['map']):
            x = i % width
            y = i // width
            c = ImageColor.getrgb(palette[level])
            alpha = int(30 + m * 225)
            c = c[:3] + (alpha,)
            draw.rectangle((x * cell_size + 1 + margin,
                y * cell_size + 1 + margin,
                x * cell_size + (cell_size-1) + margin,
                y * cell_size + (cell_size-1) + margin
                ), fill=c)
            i += 1
        level += 1
    
    # 在图片左下角写入模型名称
    draw.text(
        (margin + 10, image_height - margin - 60),
        "[ {} ]".format(model_name),
        fill="#000000",
        align="right",
        font=ImageFont.truetype("Anonymous Pro", 30))
    # 在模型名称下方写入字表大小
    draw.text(
        (margin + 40, image_height - margin - 20),
        "vocab size: {:,} ".format(vocab_size),
        fill="#000000",
        align="right",
        font=ImageFont.truetype("Anonymous Pro", 20))

    # 在图片右下角写入字表统计信息
    stats_name = ""
    for name in charset_stats.keys():
        stats_name += "{}:\n".format(name)
    draw.text(
        (image_width - margin - 700, image_height - margin - 140),
        stats_name,
        fill="#000000",
        align="left",
        font=zh_font)
    
    stats_value = ""
    for s in charset_stats.values():
        stats_value += "{:4} / {:4}  ({:.2%})\n".format(s['known'], s['total'], float(s['known'])/s['total'])
    draw.text(
        (image_width - margin - 270, image_height - margin - 140),
        stats_value,
        fill="#000000",
        align="right",
        font=zh_font)

    # 保存图像
    image.save(filename)
