# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from vocab_coverage.utils import logger
from vocab_coverage import constants


# 获取《通用规范汉字表》中一级、二级、三级字表
# General Standard Chinese Characters (GSCC)
def fetch_gscc_charset() -> dict[str, dict]:
    links = [
        {'name': constants.CHARSET_CHINESE_GSCC_1,
         'color': constants.PALETTE_CHINESE_GSCC_1,
         'url':'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%80%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89'},
        {'name': constants.CHARSET_CHINESE_GSCC_2,
         'color': constants.PALETTE_CHINESE_GSCC_2,
         'url':'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%BA%8C%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89'},
        {'name': constants.CHARSET_CHINESE_GSCC_3,
         'color': constants.PALETTE_CHINESE_GSCC_3,
         'url':'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%89%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89'},
    ]
    charsets = {}
    for link in links:
        r = requests.get(link['url'], timeout=30)
        soup = BeautifulSoup(r.text, 'html.parser')
        chinese_chars = soup.css.select('#mw-content-text div p a')
        chars = []
        for c in chinese_chars:
            title = c.get('title')
            if title is not None:
                title = title.replace('（页面不存在）', '')
                chars.append(title)
        charsets[link['name']] = {
            'texts': chars,
            'color': link['color']
        }
    return charsets

# 获取《常用國字標準字體表》中常用字和次常用字字表
# Standard Forms of Common National Characters (SFCNC)
def fetch_sfcnc_charset() -> dict[str, dict]:
    links = [
        {'name': constants.CHARSET_CHINESE_SFCNC_A,
         'color': constants.PALETTE_CHINESE_SFCNC_A,
         'url':'https://raw.githubusercontent.com/ButTaiwan/cjktables/master/taiwan/edu_standard_1.txt'},
        {'name': constants.CHARSET_CHINESE_SFCNC_B,
         'color': constants.PALETTE_CHINESE_SFCNC_B,
         'url':'https://raw.githubusercontent.com/ButTaiwan/cjktables/master/taiwan/edu_standard_2.txt'},
    ]
    charsets = {}
    for link in links:
        chars = []
        r = requests.get(link['url'], timeout=30)
        lines = r.text.split('\n')
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                if l[0] == '#':
                    continue
                w = l.split('\t')
                if len(w) > 0:
                    chars.append(w[0])
        charsets[link['name']] = {
            'texts': chars,
            'color': link['color']
        }
    return charsets

# 生成Unicode中日韩统一表意文字
def fetch_unicode_charset() -> dict[str, dict]:
    # 中日韩统一表意文字区段包含了20,992个汉字，编码范围为U+4E00–U+9FFF，Unicode 14.0起已完整填满。 
    chars = [chr(i) for i in range(0x4E00, 0x9FFF+1)]
    return {constants.CHARSET_CHINESE_UNICODE: {
        'texts': chars,
        'color': constants.PALETTE_CHINESE_UNICODE
    }}

# 生成日文平假名和片假名
# 编码范围为U+3040–U+309F和U+30A0–U+30FF
def fetch_japanese_charset() -> dict[str, dict]:
    chars = [chr(i) for i in range(0x3040, 0x309F)]
    chars.extend([chr(i) for i in range(0x30A0, 0x30FF)])
    return {constants.CHARSET_JAPANESE: {
        'texts': chars,
        'color': constants.PALETTE_JAPANESE
    }}

# 生成韩文
# 编码范围为U+AC00–U+D7AF
def fetch_korean_charset() -> dict[str, dict]:
    chars = [chr(i) for i in range(0xAC00, 0xD7AF)]
    return {constants.CHARSET_KOREAN: {
        'texts': chars,
        'color': constants.PALETTE_KOREAN
    }}

def append_charset(base:dict[str, dict], new:dict[str, dict], overlap:bool=False, debug:bool=False) -> dict[str, dict]:
    if overlap:
        if debug:
            for new_charset, new_value in new.items():
                logger.info('%s \t字数：%d',
                    new_charset, len(new_value['texts']))
        base.update(new)
        return base

    for new_charset, new_value in new.items():
        additional_chars = []
        for ch in new_value['texts']:
            found = False
            for base_value in base.values():
                if ch in base_value['texts']:
                    found = True
                    break
            if not found:
                additional_chars.append(ch)
        new_charset = new_charset + constants.CHARSET_EXT
        base[new_charset] = {
            'texts': additional_chars,
            'color': new_value['color']
        }
        if debug:
            logger.info('%s \t字数：%d，新增 %d 字',
                        new_charset, len(new_value['texts']), len(additional_chars))
    return base

def get_chinese_charsets(overlap:bool=False, debug:bool=True) -> dict[str, dict]:
    # 获取《通用规范汉字表》中一级、二级、三级字表
    gscc = fetch_gscc_charset()
    charsets = gscc
    if debug:
        for charset, value in gscc.items():
            logger.info('%s \t字数：%d', charset, len(value['texts']))
            # logger.debug(value['texts'])
    # 获取《常用國字標準字體表》中常用字和次常用字字表
    sfcnc = fetch_sfcnc_charset()
    charsets = append_charset(charsets, sfcnc, overlap, debug)
    # 获取Unicode中日韩统一表意文字
    unicode = fetch_unicode_charset()
    charsets = append_charset(charsets, unicode, overlap, debug)
    # 总字数统计
    total = 0
    for value in charsets.values():
        total += len(value['texts'])
    logger.info('总字数：%d', total)
    return charsets

def simplify_chinese_charsets(charsets:dict[str, dict], debug:bool=False) -> dict[str, dict]:
    mapping = {
        constants.CHARSET_CHINESE_COMMON: {
            'color': constants.PALETTE_CHINESE_COMMON,
            'source': [
                constants.CHARSET_CHINESE_GSCC_1,
                constants.CHARSET_CHINESE_GSCC_2,
                constants.CHARSET_CHINESE_SFCNC_A_EXT]},
        constants.CHARSET_CHINESE_RARE: {
            'color': constants.PALETTE_CHINESE_RARE,
            'source': [
                constants.CHARSET_CHINESE_GSCC_3,
                constants.CHARSET_CHINESE_SFCNC_B_EXT,
                constants.CHARSET_CHINESE_UNICODE_EXT]},
    }

    new_charsets = {}
    for new_charset, value in mapping.items():
        chars = []
        for charset in value['source']:
            if charset in charsets:
                chars.extend(charsets[charset]['texts'])
        new_charsets[new_charset] = {
            'texts': chars,
            'color': value['color']
        }
        if debug:
            logger.info('%s \t字数：%d', new_charset, len(chars))

    return new_charsets

def get_token_charsets(debug:bool=False) -> dict[str, dict]:
    # 中文
    charsets = get_chinese_charsets(debug=False)
    charsets = simplify_chinese_charsets(charsets, debug=debug)
    # 日文
    jp = fetch_japanese_charset()
    charsets = append_charset(charsets, jp, overlap=True, debug=debug)
    # 韩文
    kr = fetch_korean_charset()
    charsets = append_charset(charsets, kr, overlap=True, debug=debug)
    # 英文等字母
    charsets[constants.CHARSET_ENGLISH] = {
        'texts': [],
        'color': constants.PALETTE_ENGLISH
    }
    logger.info('%s \t字数：%d', constants.CHARSET_ENGLISH,
                len(charsets[constants.CHARSET_ENGLISH]['texts']))
    # 数字
    charsets[constants.CHARSET_DIGIT] = {
        'texts': [],
        'color': constants.PALETTE_DIGIT
    }
    logger.info('%s \t字数：%d', constants.CHARSET_DIGIT,
                len(charsets[constants.CHARSET_DIGIT]['texts']))
    # 其他
    charsets[constants.CHARSET_OTHER] = {
        'texts': [],
        'color': constants.PALETTE_OTHER
    }
    logger.info('%s \t字数：%d', constants.CHARSET_OTHER,
                len(charsets[constants.CHARSET_OTHER]['texts']))
    total = 0
    for value in charsets.values():
        total += len(value['texts'])
    logger.info('总字数：%d', total)

    return charsets

def get_thuocl_dicts(debug:bool=False) -> dict[str, dict]:
    # http://thuocl.thunlp.org/
    # CSDN博客 时间：2014.07-2016.07 文档数：3785976
    # 新浪新闻 时间：2008.01-2016.11 文档数：8421097
    # 搜狗语料 文档数：729008561
    logger.info('获取清华大学中文词表...')

    corpus = {
        'CSDN博客': {'name': 'CSDN博客', 'num_of_documents': 3785976},
        '新浪新闻': {'name': '新浪新闻', 'num_of_documents': 8421097},
        '搜狗语料': {'name': '搜狗语料', 'num_of_documents': 729008561},
    }

    dicts = {
        'IT': {'url': 'http://thuocl.thunlp.org/source/THUOCL_it.txt', 'corpus': corpus['CSDN博客'], 'color': constants.PALETTE_WORD_IT},
        '财经': {'url': 'http://thuocl.thunlp.org/source/THUOCL_caijing.txt', 'corpus': corpus['新浪新闻'], 'color': constants.PALETTE_WORD_FINANCE},
        '成语': {'url': 'http://thuocl.thunlp.org/source/THUOCL_chengyu.txt', 'corpus': corpus['新浪新闻'], 'color': constants.PALETTE_WORD_IDIOM},
        '地名': {'url': 'http://thuocl.thunlp.org/source/THUOCL_diming.txt', 'corpus': corpus['搜狗语料'], 'color': constants.PALETTE_WORD_PLACE},
        '历史名人': {'url': 'http://thuocl.thunlp.org/source/THUOCL_lishimingren.txt', 'corpus': corpus['新浪新闻'], 'color': constants.PALETTE_WORD_PERSON},
        '诗词': {'url': 'http://thuocl.thunlp.org/source/THUOCL_poem.txt', 'corpus': corpus['新浪新闻'], 'color': constants.PALETTE_WORD_POEM},
        '医学': {'url': 'http://thuocl.thunlp.org/source/THUOCL_medical.txt', 'corpus': corpus['新浪新闻'], 'color': constants.PALETTE_WORD_MEDICAL},
        '饮食': {'url': 'http://thuocl.thunlp.org/source/THUOCL_food.txt', 'corpus': corpus['搜狗语料'], 'color': constants.PALETTE_WORD_FOOD},
        '法律': {'url': 'http://thuocl.thunlp.org/source/THUOCL_law.txt', 'corpus': corpus['搜狗语料'], 'color': constants.PALETTE_WORD_LAW},
        '汽车': {'url': 'http://thuocl.thunlp.org/source/THUOCL_car.txt', 'corpus': corpus['搜狗语料'], 'color': constants.PALETTE_WORD_CAR},
        '动物': {'url': 'http://thuocl.thunlp.org/source/THUOCL_animal.txt', 'corpus': corpus['搜狗语料'], 'color': constants.PALETTE_WORD_ANIMAL},
    }
    total = 0
    for name, value in dicts.items():
        r = requests.get(value['url'], timeout=30)
        r.encoding = r.apparent_encoding
        lines = r.text.split('\n')
        if len(lines) == 1:
            lines = r.text.split('\r')
        print('\n'.join(lines[:3]+['...']))
        value['items'] = []
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                if l[0] == '#':
                    continue
                w = l.split('\t')
                if len(w) > 0:
                    try:
                        freq = int(w[1])
                    except ValueError:
                        freq = 0
                    except IndexError:
                        freq = 0
                        logger.warning("IndexError: %s", l)
                    value['items'].append({
                        'text': w[0],
                        'frequency': freq,
                    })
        # 按照词频排序
        value['items'].sort(key=lambda x: x['frequency'], reverse=True)
        if debug:
            logger.debug('%s \t词数：%d', name, len(value['items']))
        total += len(value['items'])
    if debug:
        logger.debug('总词数：%d', total)
    return dicts

def get_chinese_word_dicts(debug:bool=False) -> dict[str, dict]:
    thuocl_dicts = get_thuocl_dicts(debug=debug)
    logger.info('精简词表...')
    threshold_per_category = 2000
    dicts = {}
    for name, value in thuocl_dicts.items():
        dicts[name] = {
            # 'items': value['items'][:threshold_per_category],
            'texts': [item['text'] for item in value['items'][:threshold_per_category]],
            'color': value['color']
        }
        if debug:
            logger.debug('%s \t词数：%d', name, len(dicts[name]['texts']))
    total = sum(len(value['texts']) for value in dicts.values())
    if debug:
        logger.debug('总词数：%d', total)
    return dicts