# -*- coding: utf-8 -*-

import random
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont
from sklearn.preprocessing import MinMaxScaler
from vocab_coverage.utils import lighten_color, logger

default_palette = [
    '#B04759', '#E76161','#F99B7D',
    '#146C94', '#19A7CE', '#E893CF',
]

# apt install fonts-noto-cjk fonts-anonymous-pro fonts-noto-color-emoji
def get_available_font_from_list(fonts: List[str], size=14):
    for font in fonts:
        try:
            # 尝试加载字体文件
            # 如果加载成功，跳出循环
            return ImageFont.truetype(font, size=size)
        except IOError:
            continue  # 如果加载失败，继续尝试下一个字体
    return None

def get_chinese_font(size=14):
    font_paths = [
        "NotoSansMonoCJKsc-Regular",
        "NotoSansCJKsc-Regular",
        "NotoSansCJK",
        "NotoSansCJK-Regular",
        "NotoSerifCJK",
        "NotoSerifCJK-Regular",
        "STHeiti Light",
        "微软雅黑",
    ]
    return get_available_font_from_list(font_paths, size=size)

def get_english_font(size=14):
    font_paths = [
        "Anonymous Pro",
        "DejaVuSansMono",
        "Arial",
    ]
    return get_available_font_from_list(font_paths, size=size)

def draw_vocab_graph(model_name: str, charset_stats:dict, vocab_size:int, width=100, height=100, cell_size=60, margin=300, palette=default_palette):
    total_chars = sum([s['total'] for s in charset_stats.values()])

    # 定义图像大小
    image_width = width * cell_size + 1 + margin * 2
    height = total_chars // width + 1
    image_height = height * cell_size + 1 + margin * 8

    # 创建新的空白图像
    image = Image.new("RGB", (image_width, image_height), "#FFFFFF")

    # 获取图像的像素访问对象
    pixels = image.load()

    grid_color = (255,255,255, 40)

    # 根据map绘制栅格
    draw = ImageDraw.Draw(image)
    # 画上底图
    draw.rectangle((margin, margin, width * cell_size + margin, height * cell_size + margin), fill='#EEEEEE')
    # 画每一个方块
    i = 0
    level = 0
    zh_font = get_chinese_font(int(cell_size*0.7))
    for name, stats in charset_stats.items():
        for j, m in enumerate(stats['map']):
            x = i % width
            y = i // width
            c = ImageColor.getrgb(palette[level])
            # 画方块
            draw.rectangle((x * cell_size + 1 + margin,
                y * cell_size + 1 + margin,
                x * cell_size + (cell_size-1) + margin,
                y * cell_size + (cell_size-1) + margin
                ), fill=lighten_color(c, 0.7-m))
            ch = stats['chars'][j]
            # 方块内填入文字
            draw.text(
                (x * cell_size + int(0.15*cell_size) + margin, y * cell_size - int(0.05*cell_size) + margin),
                ch,
                font=zh_font,
                fill=lighten_color(c, 0.5-m))
            i += 1
        level += 1
    
    # 在图片左下角写入模型名称
    draw.text(
        (margin + int(margin/4), image_height - margin - int(4.5*margin)),
        "[ {} ]".format(model_name),
        fill="#000000",
        align="right",
        font=get_english_font(int(margin*0.75)))
    # 在模型名称下方写入字表大小
    draw.text(
        (margin + int(0.3*margin), image_height - margin - int(3.5*margin)),
        "[ vocab size: {:,} ]".format(vocab_size),
        fill="#000000",
        align="right",
        font=get_english_font(int(margin/2)))

    # 在图片右下角写入字表统计信息
    zh_font = get_chinese_font(int(margin*0.5))

    # 画字符集图例方块
    legend_y = image_height - margin - int(3.5*margin)
    box_width = int(margin / 2)
    box_start_x = image_width - margin - int(15.5*margin)
    box_start_y = legend_y + int(0.1*margin)
    box_margin = int(0.2*box_width)
    for i, color in enumerate(palette):
        x = box_start_x
        y = box_start_y + i * (box_width + box_margin)
        draw.rectangle((x, y, x+box_width, y+box_width), fill=color)
    # 画字符集名称
    stats_name = ""
    for name in charset_stats.keys():
        stats_name += "{}:\n".format(name)
    draw.text(
        (image_width - margin - int(15*margin), legend_y),
        stats_name,
        fill="#000000",
        align="left",
        font=zh_font)
    # 画字符集统计信息
    stats_value = ""
    for s in charset_stats.values():
        stats_value += "{:4} / {:4}  ({:.2%})\n".format(s['known'], s['total'], float(s['known'])/s['total'])
    draw.text(
        (image_width - margin - 6*margin, legend_y),
        stats_value,
        fill="#000000",
        align="right",
        font=zh_font)

    return image


from vocab_coverage.charsets import CharsetClassifier
import math

# model: {model_name, model, tokenizer, vocab, embeddings, embeddings_2d}
# def draw_vocab_embeddings(model, charsets, width=8000, height=8000, is_detail=False, debug=False):
def draw_vocab_embeddings(model_name:str, embeddings_2d:List[List[float]], vocab:List[str], charsets:List[str],
                        embedding_type:str, width=8000, height=8000, is_detailed=False, debug=False):
    vocab_size = len(vocab)

    # calculate image size, margin, etc.
    margin = int(width / 20)
    image_width = width + margin * (2+2) # outer + inner margin
    image_height = height + margin * (2+2+3) # outer + inner margin + banner

    # normalize embeddings
    scaler = MinMaxScaler()
    embeddings_2d_norm = scaler.fit_transform(embeddings_2d)

    # draw embeddings
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width + (3*margin), height + (3*margin)), fill='#F0F0F0')

    # CharsetClassifier
    classifier = CharsetClassifier(charsets=charsets, is_detailed=is_detailed)
    word_type_count = {k: 0 for k in classifier.get_types()}
    palette = classifier.get_palette(with_prefix_palette=True)

    if debug:
        # logger.debug(f"palette: {palette}")
        logger.debug(f"[{model_name}]: draw embedding point: {vocab_size}")

    # draw embedding point
    # clip font size to [12, margin]
    font_size = int(margin * 2000 / vocab_size)
    font_size = int(min(max(font_size, 12), margin))
    zh_font = get_chinese_font(font_size)
    if debug:
        logger.debug(f"font size: {font_size}, font: {zh_font.getname()}")
    for i, (x, y) in enumerate(embeddings_2d_norm):
        word = vocab[i]
        word_type = classifier.get_word_type(word)
        word_type_count[word_type] += 1

        if word.startswith('##'):
            word_type = '##' + word_type

        c = palette[word_type]

        # draw text
        x = x * width + margin * 2
        y = y * height + margin * 2
        try:
            draw.text((x, y), word, fill=c, stroke_width=1, stroke_fill='#F0F0F0', font=zh_font)
        except Exception as e:
            logger.error(f"[{model_name}]: warning: draw text error: {e}")

    if debug:
        logger.debug(f"[{model_name}]: token type counts: {word_type_count}")
    # draw model name
    font_size = int(margin / 2)
    draw.text((margin, image_height - (3*margin)),
        f'[ {model_name} ]',
        fill='#000000',
        font=get_english_font(font_size))

    # draw embedding type
    draw.text((margin, image_height - int(2.3*margin)),
        f"[ {embedding_type} embeddings ]",
        fill='#000000',
        font=get_english_font(int(font_size/1.5)))

    # draw vocab size
    draw.text((margin, image_height - int(1.8*margin)),
        "[ vocab size: {:,} ]".format(vocab_size),
        fill='#000000',
        font=get_english_font(int(font_size/1.5)))

    # draw legend
    font_size = int(margin / 3)
    box_width = int(margin / 2)
    column = 4
    column_width = int(box_width * 6)
    column_size = math.ceil(len(word_type_count) / column)
    font_label = get_chinese_font(font_size)
    font_count = get_chinese_font(int(font_size/1.7))
    for i, word_type in enumerate(word_type_count.keys()):
        row = int(i % column_size)
        col = int(i / column_size)
        x = image_width - ((column)*column_width) + (col * column_width)
        y = image_height - ((column_size-row)*(box_width*1.4)) - (margin)
        color = palette[word_type]
        # logger.debug(i, row, col, x, y, word_type, color)

        # draw box
        draw.rectangle((x, y, x+box_width, y+box_width), fill=color)
        # logger.debug(word_type, color)

        # draw label
        draw.text((x+box_width*1.5, int(y - font_size*0.3)),
            f'{word_type}',
            fill='#000000',
            font=font_label)
        # draw count
        draw.text((x+box_width*1.5, int(y + font_size*1)),
            f'({word_type_count[word_type]})',
            fill='#000000',
            font=font_count)

    return image
