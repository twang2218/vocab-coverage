import random
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont

vocab_levels = [3500, 3000, 1605]
def vocab_index_to_level(index: int) -> int:
    if index < vocab_levels[0]:
        return 1
    elif index < vocab_levels[0] + vocab_levels[1]:
        return 2
    elif index < vocab_levels[0] + vocab_levels[1] + vocab_levels[2]:
        return 3
    else:
        return 0

def draw_vocab_graph(model_name: str, vocab: List, filename: str, width=100, height=100, cell_size=10, margin=20, palette=['#8BACAA', '#B04759', '#E76161','#F99B7D','#FEC89A','#F9F871','#C0D6DF','#E8E8E8','#FFFFFF']):
    # 定义图像大小
    image_width = width * cell_size + 1 + margin * 2
    height = len(vocab) // width + 1
    image_height = height * cell_size + 1 + margin * 5

    # 创建新的空白图像
    image = Image.new("RGBA", (image_width, image_height), "#EEEEEE")

    # 获取图像的像素访问对象
    pixels = image.load()

    grid_color = (255,255,255, 40)
    count_by_level = [0, 0, 0, 0]

    # 画出第一级字表的网格
    draw = ImageDraw.Draw(image)
    for i in range(0, len(vocab)):
        x = i % width
        y = i // width

        level = vocab_index_to_level(i)
        c = palette[level]        
        c = ImageColor.getrgb(c)
        if vocab[i] == 1:
            count_by_level[level] += 1
        ## vocab[i] = 0: transparent, 1: opaque
        alpha = int(30 + vocab[i] * 225)
        c = c[:3] + (alpha,)
        draw.rectangle((x * cell_size + 1 + margin,
                        y * cell_size + 1 + margin,
                        x * cell_size + (cell_size-1) + margin,
                        y * cell_size + (cell_size-1) + margin
                        ), fill=c)
    # 在图片左下角写入模型名称
    draw.text(
        (margin + 10, image_height - margin - 20),
        "[ {} ]".format(model_name),
        fill="#000000",
        align="right",
        font=ImageFont.truetype("Anonymous Pro", 20))

    # 在图片右下角写入字表统计信息
    stats = "I: {}/{} ({:.2%}), II: {}/{} ({:.2%}), III: {}/{} ({:.2%})".format(
        count_by_level[1], vocab_levels[0], count_by_level[1]/vocab_levels[0],
        count_by_level[2], vocab_levels[1], count_by_level[2]/vocab_levels[1],
        count_by_level[3], vocab_levels[2], count_by_level[3]/vocab_levels[2])
    draw.text(
        (image_width - margin - 500, image_height - margin - 50),
        stats,
        fill="#000000",
        align="left",
        font=ImageFont.truetype("Georgia", 15))    
    
    # 保存图像
    image.save(filename)

if __name__ == '__main__':
    print('Generating random vocab graph for testing...')
    vocab = [random.randint(0, 1) for _ in range(sum(vocab_levels))]
    draw_vocab_graph("LLaMA", vocab, 'grid_graph.png')
