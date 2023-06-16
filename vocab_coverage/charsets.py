# -*- coding: utf-8 -*-

import argparse
import json
from bs4 import BeautifulSoup
import pandas as pd
import requests

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--charset_file", type=str, default="charset.json", help="字表文件")
    args = parser.parse_args()
    generate_charsets(args.charset_file)