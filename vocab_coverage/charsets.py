# -*- coding: utf-8 -*-

import argparse
import json
from bs4 import BeautifulSoup
import requests
from vocab_coverage.utils import lighten_color

def get_chinese_chars():
    urls = [
        'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%80%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
        'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%BA%8C%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
        'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%89%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
    ]
    charset = []
    for url in urls:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        chinese_chars = soup.css.select('#mw-content-text div p a')
        chars = []
        for c in chinese_chars:
            title = c.get('title')
            if title is not None:
                title = title.replace('（页面不存在）', '')
                chars.append(title)
        charset.append(chars)
    return charset

def generate_unicode_chinese_chars():
    # 生成Unicode中日韩统一表意文字
    # 中日韩统一表意文字区段包含了20,992个汉字，编码范围为U+4E00–U+9FFF，Unicode 14.0起已完整填满。 
    chars = [chr(i) for i in range(0x4E00, 0x9FFF+1)]
    return chars

def generate_cnc_chars():
    # 获取《常用國字標準字體表》中常用字和次常用字字表
    urls = [
        'https://raw.githubusercontent.com/ButTaiwan/cjktables/master/taiwan/edu_standard_1.txt',
        'https://raw.githubusercontent.com/ButTaiwan/cjktables/master/taiwan/edu_standard_2.txt',
    ]
    charset = []
    for url in urls:
        chars = []
        r = requests.get(url)
        lines = r.text.split('\n')
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                if l[0] == '#':
                    continue
                w = l.split('\t')
                if len(w) > 0:
                    chars.append(w[0])
        charset.append(chars)
    return charset

def generate_charsets(filename:str):
    charset = {}
    # 获取《通用规范汉字表》中一级、二级、三级字表'
    s1 = get_chinese_chars()
    charset['《通用规范汉字表》一级汉字'] = s1[0]
    charset['《通用规范汉字表》二级汉字'] = s1[1]
    charset['《通用规范汉字表》三级汉字'] = s1[2]
    for k, v in charset.items():
        print(f"{k}共有{len(v)}个汉字")

    base_chars = set()
    for cc in charset.values():
        base_chars.update(cc)
    print(f"《通用规范汉字表》共有{len(base_chars)}个汉字")
    s2 = generate_cnc_chars()

    extra_chars = []
    for c in s2[0]:
        if c not in base_chars:
            extra_chars.append(c)
    charset['《常用國字標準字體表》甲表(增)'] = extra_chars
    print(f"《常用國字標準字體表》甲表共有{len(s2[0])}个汉字，其中{len(extra_chars)}个汉字不在《通用规范汉字表》中")
    for c in extra_chars:
        base_chars.update(c)

    extra_chars = []
    for c in s2[1]:
        if c not in base_chars:
            extra_chars.append(c)
    charset['《常用國字標準字體表》乙表(增)'] = extra_chars
    print(f"《常用國字標準字體表》乙表共有{len(s2[1])}个汉字，其中{len(extra_chars)}个汉字不在《通用规范汉字表》中")
    for c in extra_chars:
        base_chars.update(c)

    chars = generate_unicode_chinese_chars()
    extra_chars = []
    for c in chars:
        if c not in base_chars:
            extra_chars.append(c)
    print(f"《Unicode中日韩统一表意文字》共有{len(chars)}个汉字，其中{len(extra_chars)}个汉字不在《通用规范汉字表》和《常用國字標準字體表》中")
    charset['《Unicode中日韩统一表意文字》(增)'] = extra_chars

    print(f"共有{sum([len(c) for c in charset.values()])}个汉字")

    # 保存到JSON文件
    with open(filename, 'w') as f:
        json.dump(charset, f, ensure_ascii=False, indent=4)



class CharsetClassifier:
    def __init__(self, charsets:dict, is_detailed=False):
        self.charsets = charsets
        self.is_detailed = is_detailed
        self.palette = None

    @staticmethod
    def is_japanese_kana(word:str):
        word = word.strip()
        for character in word:
            unicode_value = ord(character)
            if (unicode_value >= 0x3040 and unicode_value <= 0x309F) or \
            (unicode_value >= 0x30A0 and unicode_value <= 0x30FF):
                # 只要出现平假名和片假名，就认为是日文，因为日文中有汉字
                return True
        return False

    @staticmethod
    def is_korean(word:str):
        word = word.strip()
        for character in word:
            unicode_value = ord(character)
            if unicode_value >= 0xAC00 and unicode_value <= 0xD7A3:
                # 只要出现韩文，就认为是韩文，因为韩文中有汉字
                return True
        return False

    @staticmethod
    def is_english(word:str):
        word = word.strip()
        for char in word:
            if not char.isalpha():
                return False
        return True

    @staticmethod
    def is_numeric(word:str):
        word = word.strip()
        for char in word:
            if not char.isdigit():
                return False
        return True

    def is_chinese(self, word:str, category:str):
        word = word.strip()
        if category == '《通用规范汉字表》一级汉字':
            return word[0] in self.charsets['《通用规范汉字表》一级汉字']
        elif category == '《通用规范汉字表》二级汉字':
            return word[0] in self.charsets['《通用规范汉字表》二级汉字']
        elif category == '《通用规范汉字表》三级汉字':
            return word[0] in self.charsets['《通用规范汉字表》三级汉字']
        elif category == '《常用國字標準字體表》甲表(增)':
            return word[0] in self.charsets['《常用國字標準字體表》甲表(增)']
        elif category == '《常用國字標準字體表》乙表(增)':
            return word[0] in self.charsets['《常用國字標準字體表》乙表(增)']
        elif category == '《Unicode中日韩统一表意文字》(增)':
            return word[0] in self.charsets['《Unicode中日韩统一表意文字》(增)']
        elif category == '汉字(常用字)':
            return word[0] in self.charsets['《通用规范汉字表》一级汉字'] or \
                word[0] in self.charsets['《通用规范汉字表》二级汉字'] or \
                word[0] in self.charsets['《常用國字標準字體表》甲表(增)']
        elif category == '汉字(生僻字)':
            return word[0] in self.charsets['《通用规范汉字表》三级汉字'] or \
                word[0] in self.charsets['《常用國字標準字體表》乙表(增)'] or \
                word[0] in self.charsets['《Unicode中日韩统一表意文字》(增)']
        elif category == '汉字':
            return self.is_chinese(word, '汉字(常用字)') or \
                self.is_chinese(word, '汉字(生僻字)')
        else:
            return False

    def get_word_type(self, word:str):
        word = word.strip()
        if len(word) == 0:
            return '其他'
        elif word.startswith('##'):
            return self.get_word_type(word[2:])
        elif word.startswith('▁'):
            return self.get_word_type(word[1:])
        elif word.endswith('</w>'):
            return self.get_word_type(word[:-4])
        elif self.is_numeric(word):
            return '数字'
        elif self.is_japanese_kana(word):
            return '日文'
        elif self.is_korean(word):
            return '韩文'
        else:
            if self.is_detailed:
                cats = [
                    '《通用规范汉字表》一级汉字',
                    '《通用规范汉字表》二级汉字',
                    '《通用规范汉字表》三级汉字',
                    '《常用國字標準字體表》甲表(增)',
                    '《常用國字標準字體表》乙表(增)',
                    '《Unicode中日韩统一表意文字》(增)',
                ]
            else:
                cats = ['汉字(常用字)', '汉字(生僻字)']
            for cat in cats:
                if self.is_chinese(word, cat):
                    return cat

            # 英文识别必须靠后，因为 isalpha() 会把中日韩文也识别为字母
            if self.is_english(word):
                return '英文'
            else:
                return '其他'

    def get_palette(self, with_prefix_palette=False):
        if self.palette is not None:
            return self.palette

        if with_prefix_palette:

            from PIL import ImageColor
            palette = self.get_palette(with_prefix_palette=False)
            prefix_palette = {}
            for k, v in palette.items():
                c = lighten_color(v, 0.7)
                prefix_palette['##'+k] = c
            self.palette = palette.copy()
            for k, v in self.palette.items():
                if isinstance(v, str):
                    self.palette[k] = ImageColor.getrgb(v)
            self.palette.update(prefix_palette)
            return self.palette

        if self.is_detailed:
            self.palette = {
                '《通用规范汉字表》一级汉字': '#B04759',
                '《通用规范汉字表》二级汉字': '#E76161',
                '《通用规范汉字表》三级汉字': '#F99B7D',
                '《常用國字標準字體表》甲表(增)': '#146C94',
                '《常用國字標準字體表》乙表(增)': '#19A7CE',
                '其他汉字': '#E893CF',
                '日文': '#827717',
                '韩文': '#FFA000',
                '英文': '#2E7D32',
                '数字': '#01579B',
                '其他': '#212121',
            }
        else:
            self.palette = {
                '汉字(常用字)': '#D50000',
                '汉字(生僻字)': '#7B1FA2',
                '日文': '#827717',
                '韩文': '#FFA000',
                '英文': '#2E7D32',
                '数字': '#01579B',
                '其他': '#212121',
            }
        return self.palette

    def get_types(self):
        if self.is_detailed:
            return [
                '《通用规范汉字表》一级汉字',
                '《通用规范汉字表》二级汉字',
                '《通用规范汉字表》三级汉字',
                '《常用國字標準字體表》甲表(增)',
                '《常用國字標準字體表》乙表(增)',
                '其他汉字',
                '日文',
                '韩文',
                '英文',
                '数字',
                '其他',
            ]
        else:
            return [
                '汉字(常用字)',
                '汉字(生僻字)',
                '日文',
                '韩文',
                '英文',
                '数字',
                '其他',
            ]
