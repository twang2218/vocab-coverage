# -*- coding: utf-8 -*-

import math
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from vocab_coverage.lexicon import Lexicon
from vocab_coverage.utils import lighten_color, logger
from vocab_coverage import constants

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

def draw_coverage_graph(model_name:str, lexicon:Lexicon, vocab_size:int, width:int=8500, height:int=10000):
    # 计算各部分图像尺寸大小
    image_width = width
    image_height = height
    margin = image_width // 30

    # 创建新的空白图像
    image = Image.new("RGB", (image_width, image_height), constants.PALETTE_BACKGROUND)
    draw = ImageDraw.Draw(image)

    # 绘制栅格区域
    draw_coverage_region(draw,
                         region=(margin, margin, image_width - margin, image_width - margin),
                         lexicon=lexicon)

    # 绘制文字区域
    ## 标题
    title = f"[ {model_name} ]"
    ## 副标题
    if lexicon.classifier.granularity == constants.GRANULARITY_TOKEN:
        subtitle = "〔 词表 token 完整性覆盖图 〕"
    elif lexicon.classifier.granularity == constants.GRANULARITY_CHARACTER:
        subtitle = "〔 汉字完整性覆盖图 〕"
    elif lexicon.classifier.granularity == constants.GRANULARITY_WORD:
        subtitle = "〔 中文多字词完整性覆盖图 〕"
    else:
        raise ValueError(f"Unknown granularity: {lexicon.classifier.granularity}")
    ## 辅助信息
    total_items = lexicon.get_item_count()
    granularity = lexicon.classifier.granularity
    intact = 0
    for _, value in lexicon:
        intact += value['stats']['intact']
    completeness = float(intact / total_items if total_items > 0 else 0)
    auxiliary = [
        f"[ vocab size: {vocab_size:,} ]",          # 字表大小
        f"[ granularity: {granularity} ]",          # 颗粒度
        f"[ completeness: {completeness:.2%} ]"     # 完整覆盖率
    ]
    ## 图例
    legends = []
    for category, value in lexicon:
        intact = value['stats']['intact']
        total = value['stats']['total']
        completeness = f"({float(intact/total if total > 0 else 0):6.2%})"
        legend = {
            'color': value['color'],
            'label': f'{category}：',
            'value': f"{intact:6} / {total:6}  {completeness:>9}"
        }
        legends.append(legend)
    draw_text_region(draw,
                     (margin, image_width, image_width - margin, image_height - margin),
                     title, subtitle, auxiliary, legends)
    return image

def draw_coverage_region(draw:ImageDraw, region:Tuple[int, int, int, int], lexicon:Lexicon):
    ## 计算尺寸位置数值
    x, y, x2, y2 = region
    ### 计算最大正方形
    max_width = x2 - x
    max_height = y2 - y
    max_size = min(max_width, max_height)
    # region_height = y2 - y
    ### 计算行、列数
    total_items = lexicon.get_item_count()
    num_of_cell_per_row = math.ceil(math.sqrt(total_items))
    num_of_rows = math.ceil(total_items / num_of_cell_per_row)
    ### 计算每个方块的尺寸
    cell_size = (max_size) // num_of_cell_per_row
    ### 计算方块区域的尺寸
    region_width = cell_size * num_of_cell_per_row
    region_height = cell_size * num_of_rows
    ### 计算方块区域的位置
    region_x = x + (max_size - region_width) // 2
    region_y = y + (max_size - region_height) // 2
    region_x2 = region_x + region_width
    region_y2 = region_y + region_height

    ## 画上底图
    color_cell_region_background = constants.PALETTE_REGION_BACKGROUND
    draw.rectangle((region_x, region_y, region_x2, region_y2),
                    fill=color_cell_region_background)
    ## 画每一个方块
    i = 0
    font = get_chinese_font(int(cell_size*0.7))
    for category, value in lexicon:
        for item in value['items']:
            # 画方块
            if category in constants.CHARSET_CJK_SETS:
                ## CJK字符宽，因此只显示一个字符
                text = item['text'][0]
            else:
                text = item['text'][:2]
            cell_x = region_x + (i % num_of_cell_per_row) * cell_size
            cell_y = region_y + (i // num_of_cell_per_row) * cell_size
            draw_coverage_cell(draw, (cell_x, cell_y),
                               cell_size=cell_size,
                               completeness=item['completeness'],
                               color=value['color'],
                               text=text,
                               font=font)
            i += 1

def draw_text_region(draw:ImageDraw, region:Tuple[int, int, int, int], title:str, subtitle:str, auxiliary:List[str], legends:List[Dict[str, str]]):
    ## 计算文字区域相关数据
    x, y, x2, y2 = region
    width = x2 - x
    ## 计算文字尺寸
    margin = width // (30-2) # 给定的region去除了两侧margin
    text_size_large = int(0.75 * margin)
    text_size_small = int(0.5 * margin)
    text_size_xsmall = int(0.4 * margin)
    text_gap = int(0.1 * margin)

    ### 左上角写入模型名称
    draw_title(draw, (x, y), title, text_size_large)

    ### 在模型名称下方写入辅助信息（字表大小、颗粒度、完整覆盖率）
    auxiliary_x = x + text_gap
    auxiliary_y = y + text_size_large + text_gap + text_size_small + text_gap
    draw_auxiliary(draw, (auxiliary_x, auxiliary_y), auxiliary, text_size_small)

    ### 右上角写入副标题
    draw_subtitle(draw, (x2, y), subtitle, text_size_small)

    ## 在图片右下角画类别图例
    draw_legend(draw, (x2, y2), legends, text_size_xsmall)

def draw_coverage_cell(draw:ImageDraw, xy:Tuple[int, int], cell_size:int, completeness:float, color:str, text:str=None, font=None):
    x, y = xy
    draw.rectangle((x + 1, y + 1, x + (cell_size-1), y + (cell_size-1)),
        fill=lighten_color(color, 0.7-completeness))
    # 画方块内文字
    if cell_size > 40 and text is not None:
        # 如果方块尺寸太小，则不显示文字了
        # 方块内填入文字
        draw.text((x + int(0.15*cell_size), y - int(0.05*cell_size)),
            text,
            font=font,
            fill=lighten_color(color, 0.5-completeness))

def draw_title(draw:ImageDraw, xy:Tuple[int, int], title:str, text_size:int):
    if len(title) == 0:
        return
    x, y = xy
    font = get_english_font(text_size)
    draw.text(
        (x, y),
        title,
        fill=constants.PALETTE_TEXT,
        align="left",
        font=font)

def draw_subtitle(draw:ImageDraw, xy2:Tuple[int, int], subtitle:str, text_size:int):
    if len(subtitle) == 0:
        return
    x2, y2 = xy2 # 文本框右上角坐标，但视为右下角，因此副标题将在文本框右上角外部
    width = int(0.8 * (len(subtitle)+3) * text_size)
    font = get_chinese_font(text_size)
    draw.text(
        (x2 - width, y2 - text_size),
        subtitle,
        fill=constants.PALETTE_TEXT,
        align="right",
        font=font)

def draw_auxiliary(draw:ImageDraw, xy:Tuple[int, int], subtitles:List[str], text_size:int):
    if len(subtitles) == 0:
        return
    x, y = xy
    text_gap = int(0.1 * text_size)
    font = get_english_font(text_size)
    for i, subtitle in enumerate(subtitles):
        draw.text(
            (x, y + i * (text_size+text_gap)),
            subtitle,
            fill=constants.PALETTE_TEXT,
            align="left",
            font=font)

def draw_legend(draw:ImageDraw, xy2:Tuple[int, int], items:List[Dict[str, str]], text_size:int):
    if len(items) == 0:
        return

    if 'label' in items[0]:
        max_label_length = max(len(item['label']) for item in items)
        max_label_width = int(0.8*(max_label_length + 2) * text_size)
    else:
        max_label_width = 0

    if 'value' in items[0]:
        max_value_length = max(len(item['value']) for item in items)
        max_value_width = int(0.5*(max_value_length + 2) * text_size)
    else:
        max_value_width = 0

    if 'color' in items[0]:
        box_size = text_size
    else:
        box_size = 0

    x2, y2 = xy2 # 右下角坐标
    text_gap = int(0.2 * text_size)
    font_zh = get_chinese_font(text_size)
    MAX_ROWS_PER_COL = 6
    cols = math.ceil(len(items) / MAX_ROWS_PER_COL)
    rows_per_col = math.ceil(len(items) / cols)
    # 图例框（测试用）
    # draw.rectangle((x2 - (box_size + text_gap + max_label_width + text_gap + max_value_width),
    #                 y2 - (len(items) + 1) * (text_size + text_gap),
    #                 x2,
    #                 y2),
    #                 outline="#aaaaaa")

    for i, item in enumerate(items):
        current_col = i // rows_per_col
        # 画图例方块
        if 'color' in item:
            item_x = x2 - (box_size + text_gap + max_label_width + text_gap + max_value_width) * (cols - current_col)
            item_y = y2 - (rows_per_col - (i%rows_per_col) + 1) * (text_size + text_gap)
            box_x = item_x
            box_y = item_y + 2*text_gap
            draw.rectangle((box_x, box_y, box_x+box_size, box_y+box_size),
                        fill=item['color'])
        else:
            item_x = x2 - (text_gap + max_label_width + text_gap + max_value_width) * (cols - current_col)
            item_y = y2 - (rows_per_col - (i%rows_per_col) + 1) * (text_size + text_gap)
        # 画图例文字
        if 'label' in item:
            draw.text((item_x + box_size + 2*text_gap, item_y),
                    item['label'], fill=constants.PALETTE_TEXT, font=font_zh)
        # 画图例数值
        if 'value' in item:
            draw.text((item_x + box_size + text_gap + max_label_width, item_y),
                      item['value'], fill=constants.PALETTE_TEXT, font=font_zh)

def draw_embeddings_graph(model_name:str,
                          lexicon:Lexicon,
                          position:str,
                          width:int=8500,
                          height:int=10000,
                          debug=False):
    vocab_size = lexicon.get_item_count()
    granularity = lexicon.get_granularity()

    # 计算图片尺寸、边距等
    image_width = width
    image_height = height
    margin = image_width // 30

    # 创建画布
    image = Image.new('RGB', (image_width, image_height), constants.PALETTE_BACKGROUND)
    draw = ImageDraw.Draw(image)

    # 绘制向量分布区域
    draw_embedding_region(draw,
                          region=(margin, margin, image_width-margin, image_width-margin),
                          lexicon=lexicon,
                          debug=debug)

    # 绘制向量分布区周边，以覆盖向量分布区超出区域的部分
    draw.rectangle((image_width-margin, margin, image_width, image_width),
                     fill=constants.PALETTE_BACKGROUND)
    draw.rectangle((margin, image_width-margin, image_width, image_width),
                     fill=constants.PALETTE_BACKGROUND)
    # 绘制文字区域
    ## 标题
    title = f'[ {model_name} ]'
    ## 副标题
    position_name = {
        constants.EMBEDDING_POSITION_INPUT: '输入端',
        constants.EMBEDDING_POSITION_OUTPUT: '输出端',
    }
    if granularity == constants.GRANULARITY_TOKEN:
        subtitle = f"〔 词表向量{position_name[position]}分布图 〕"
        vocab_name = 'vocab'
    elif granularity == constants.GRANULARITY_CHARACTER:
        subtitle = f"〔 汉字向量{position_name[position]}分布图 〕"
        vocab_name = 'character'
    elif granularity == constants.GRANULARITY_WORD:
        subtitle = f"〔 中文多字词向量{position_name[position]}分布图 〕"
        vocab_name = 'word'
    else:
        raise ValueError(f"不支持的颗粒度：{granularity}")
    ## 辅助信息
    auxiliary = [
        f"[ {vocab_name} size: {vocab_size} ]",
        f"[ granularity: {granularity} ]",
        f"[ position: {position} ]",
    ]
    ## 图例
    legends = []
    for category, value in lexicon:
        legend = {
            'color': value['color'],
            'label': f'{category}：',
            'value': f"{len(value['items']):6}"
        }
        legends.append(legend)
    ## 绘制
    draw_text_region(draw,
                     (margin, image_width, image_width - margin, image_height - margin),
                     title, subtitle, auxiliary, legends)

    return image

def draw_embedding_region(draw:ImageDraw, region: Tuple[int, int, int, int], lexicon:Lexicon, debug:bool=False):
    ## 计算尺寸位置数值
    x, y, x2, y2 = region
    width = x2 - x
    height = y2 - y
    margin = width // 28
    # 画底图
    draw.rectangle((x, y, x2, y2), fill=constants.PALETTE_REGION_BACKGROUND)
    # 字体
    font_size = int(margin * 2000 / lexicon.get_item_count())
    font_size = int(min(max(font_size, 12), margin))
    font = get_chinese_font(font_size)
    # 画字
    if debug:
        logger.debug("> font size: %d, font: %s", font_size, font.getname())
    # 倒序类别画字，保证前面的类别不会被后面的类别覆盖
    for category, value in reversed(list(lexicon)):
        count = 0
        for item in value['items']:
            # 画字
            if 'embedding' not in item:
                logger.warning("> item has no embedding: %s", item)
                continue
            embedding = item['embedding']
            padding = margin // 4
            item_x = (x+padding) + embedding[0] * (width-font_size-padding*2)
            item_y = (x+padding) + embedding[1] * (height-font_size-padding*2)
            # if debug:
            #     logger.debug(">> %s: %s, %s", category, item['text'], embedding)
            color = value['color']
            if item['text'].startswith('##'):
                color = lighten_color(color, 0.4)
            if 'tokenized_text' in item:
                text = '/'.join(item['tokenized_text'])
            else:
                text = item['text']
            draw.text((item_x, item_y), text,
                      fill=color,
                      stroke_fill=constants.PALETTE_REGION_BACKGROUND, stroke_width=1,
                      font=font)
            count += 1
        if debug:
            logger.debug("> %s: %d", category, count)
