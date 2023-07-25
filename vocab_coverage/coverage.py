# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import sys

import torch

from vocab_coverage.draw import draw_vocab_graph
from vocab_coverage.loader import load_tokenizer
from vocab_coverage.utils import release_resource

def coverage_analysis(model_name:str, charsets, output_dir:str=None, debug=False):
    print("检查模型 {} 的字表".format(model_name))
    tokenizer = load_tokenizer(model_name, debug=debug)

    charset_stats = {
        name: {
            'known': 0,
            'total': len(chars),
            'chars': chars,
            'map': [0 for _ in range(len(chars))]
        } for name, chars in charsets.items()
    }

    if debug:
        print(tokenizer)
        print('vocab size:', tokenizer.vocab_size)
        if hasattr(tokenizer, 'cls_token_id'):
            print('[Special Token ID] => cls: {}, sep: {}, pad: {}, unk: {}, mask: {}'.format(
                tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None,
                tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None,
                tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None,
                tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
                tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None
            ))

    try:
        # 对于 ChatGLM 等模型，有特殊的头部token，需要特殊处理
        prefix_token = tokenizer.convert_tokens_to_ids('▁')
    except:
        prefix_token = tokenizer.cls_token_id

    for name, chars in charsets.items():
        for i, c in enumerate(chars):
            # 编码
            try:
                tokens_ids = tokenizer.encode(c, add_special_tokens=False)
            except Exception as e:
                if "add_special_tokens" in e.args[0]:
                    tokens_ids = tokenizer.encode(c)
                else:
                    print("编码字 {} 失败：{}".format(c, e))
                    continue

            # 编码预处理
            tn = type(tokenizer).__name__
            if len(tokens_ids) > 0 and tokens_ids[0] == prefix_token:
                # 对有头部特殊token的编码，去掉头部特殊token
                tokens_ids = tokens_ids[1:]

            # 验证编码
            if len(tokens_ids) > 1:
                c2 = tokenizer.decode(tokens_ids)
                if c != c2:
                    print("[{}] 编码字 {}({}) 失败：{} != {}".format(tn, c, tokens_ids, c, c2))
                    
            # 识字程度判断
            if len(tokens_ids) < 1 or (len(tokens_ids) == 1 and hasattr(tokenizer, 'unk_token_id') and tokens_ids[0] == tokenizer.unk_token_id):
                # 未识别的字
                charset_stats[name]['map'][i] = 0
            elif len(tokens_ids) == 1:
                # 完全识别的字
                charset_stats[name]['map'][i] = 1
                charset_stats[name]['known'] += 1
            else: # len(tokens_ids) > 1
                # 一定程度上识别的字，并不计数，只计算识别程度
                charset_stats[name]['map'][i] = 1.0/len(tokens_ids) # 识别程度
                if debug:
                    print("[{}] 汉字({})被拆分了，编码为{}".format(tn, c, tokens_ids))

    # 统计显示
    for name, stats in charset_stats.items():
        print("字表{}：{}/{} ({:.2%})".format(name, stats['known'], stats['total'], float(stats['known'])/stats['total']))

    # 生成字表图
    image = draw_vocab_graph(model_name, charset_stats, tokenizer.vocab_size, width=150)

    # 生成文件名
    filename = model_name.replace('/', '_') + '.coverage.jpg'
    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, 'assets', 'coverage')
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)

    # 保存图像
    image.save(filename, quality=80, optimize=True)

    # clean up
    del tokenizer
    release_resource(model_name, clear_cache=False)

