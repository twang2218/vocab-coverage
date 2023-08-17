# -*- coding: utf-8 -*-

import os

from vocab_coverage.draw import draw_coverage_graph
from vocab_coverage.loader import load_tokenizer
from vocab_coverage.utils import release_resource, has_parameter, generate_coverage_filename, logger
from vocab_coverage.lexicon import Lexicon
from vocab_coverage import constants

def coverage_analysis(model_name:str,
                      lexicon:Lexicon,
                      granularity:str=constants.GRANULARITY_CHARACTER,
                      folder:str=None,
                      debug=False):
    logger.info("检查模型 %s 的字表", model_name)
    tokenizer = load_tokenizer(model_name, debug=debug)

    for _, value in lexicon:
        value['stats'] = {'intact': 0, 'total': len(value['items'])}

    if debug:
        logger.debug('[%s] tokenizer: %s', model_name, tokenizer)
        logger.debug('[%s] vocab size: %d', model_name, tokenizer.vocab_size)
        if hasattr(tokenizer, 'cls_token_id'):
            logger.debug('[Special Token ID] => cls: %s, sep: %s, pad: %s, unk: %s, mask: %s',
                tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None,
                tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None,
                tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None,
                tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
                tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None
            )

    try:
        # 对于 ChatGLM 等模型，有特殊的头部token，需要特殊处理
        special_prefix_id = tokenizer.convert_tokens_to_ids(constants.TEXT_LEADING_UNDERSCORE)
    # pylint: disable=broad-except
    except Exception:
        special_prefix_id = tokenizer.cls_token_id
    
    try:
        attached_prefix = '~'
        attached_prefix_id = tokenizer.convert_tokens_to_ids(attached_prefix)
    except Exception:
        attached_prefix_id = tokenizer.cls_token_id

    # 遍历词表
    for category, value in lexicon:
        for item in value['items']:
            split_count = get_token_split_count(tokenizer, item['text'],
                                                special_prefix_id=special_prefix_id,
                                                attached_prefix=attached_prefix,
                                                attached_prefix_id=attached_prefix_id)
            if split_count > 0:
                item['completeness'] = 1.0 / split_count
                if split_count == 1:
                    value['stats']['intact'] += 1
            else:
                item['completeness'] = 0.0
        intact = value['stats']['intact']
        total = value['stats']['total']
        logger.info("[%s] 字表 《%s》：字数：%d，完整：%d，完整率：%.2f%%",
                    model_name,
                    category,
                    total,
                    intact,
                    intact / total * 100 if total > 0 else 0)

    # 计算总字数和完整率
    total_intact = sum(value['stats']['intact'] for _, value in lexicon)
    total_items = sum(value['stats']['total'] for _, value in lexicon)
    logger.info("[%s] 总字数：%d，完整率：%.2f%%", model_name, total_items, total_intact / total_items * 100)

    # 生成字表图
    image = draw_coverage_graph(model_name, lexicon, tokenizer.vocab_size)

    # 保存图像
    filename = generate_coverage_filename(model_name,
                                          granularity=granularity,
                                          folder=folder)
    if image is None:
        logger.warning("[%s] 生成 %s 完整率覆盖图失败", model_name, granularity)
        return
    logger.info("[%s] 保存 %s 完整率覆盖图 (%s) ...", model_name, granularity, filename)
    image.save(filename, quality=80, optimize=True)

def get_token_split_count(tokenizer, text:str, special_prefix_id, attached_prefix, attached_prefix_id) -> int:
    # 编码
    kwargs = {}

    if has_parameter(tokenizer.encode, 'add_special_tokens'):
        kwargs['add_special_tokens'] = False

    has_attached_prefix = False

    if isinstance(text, bytes):
        # Qwen/Qwen-7B-Chat
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            return 0

    if len(text) > 2 and text.startswith('##') and text[2] != '#':
        text = text[2:]
        text = attached_prefix + text
        has_attached_prefix = True

    tokens_ids = tokenizer.encode(text, **kwargs)
    # 编码预处理
    tn = type(tokenizer).__name__
    if len(tokens_ids) > 0 and tokens_ids[0] == special_prefix_id:
        # 对有头部特殊token的编码，去掉头部特殊token
        tokens_ids = tokens_ids[1:]

    if len(tokens_ids) > 0 and tokens_ids[0] == attached_prefix_id:
        tokens_ids = tokens_ids[1:]

    if len(tokens_ids) == 1 and tokens_ids[0] == tokenizer.unk_token_id:
        # 未知token
        return 0

    # 验证编码
    if len(tokens_ids) > 1:
        decoded = tokenizer.decode(tokens_ids)
        if has_attached_prefix:
            text = text[1:]
        if text != decoded:
            logger.warning("[%s] 编码文本 %s(%s) 和解码结果（%s）不一致", tn, text, tokens_ids, decoded)

    # logger.debug("[%s] 编码文本 %s(%s) => %s", tn, text, tokens_ids, tokenizer.decode(tokens_ids))

    return len(tokens_ids)
