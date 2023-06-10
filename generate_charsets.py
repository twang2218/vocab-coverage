import json
from bs4 import BeautifulSoup
import pandas as pd
import requests

urls = [
    'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%80%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
    'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%BA%8C%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
    'https://www.zwbk2009.com/index.php?title=%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%EF%BC%88%E4%B8%89%E7%BA%A7%E5%AD%97%E8%A1%A8%EF%BC%89',
]
def get_chinese_chars(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    chinese_chars = soup.css.select('#mw-content-text div p a')
    charset = []
    for c in chinese_chars:
        title = c.get('title')
        if title is not None:
            title = title.replace('（页面不存在）', '')
            charset.append(title)
    return charset

def get_zdic_chars(filename="zdic.json"):
    zdic = json.load(open(filename, 'r'))
    chars = []
    for c in zdic:
        chars.append(c['Char'])
    return chars

if __name__ == '__main__':
    charset = []
    for i, url in enumerate(urls):
        chars = get_chinese_chars(url)
        # print(chars)
        charset.append(chars)
        print(f"第{i+1}级字表，共有{len(chars)}个汉字")

    chars = get_zdic_chars("zdic.2020-10-22.json")
    base_chars = set()
    for cc in charset:
        base_chars.update(cc)
    extra_chars = []
    print(f"通用规范汉字表共有{len(base_chars)}个汉字")
    for c in chars:
        if c not in base_chars:
            extra_chars.append(c)
    print(f"汉典共有{len(chars)}个汉字，其中{len(extra_chars)}个汉字不在通用规范汉字表中")
    charset.append(extra_chars)

    print(f"共有{sum([len(c) for c in charset])}个汉字")

    # 保存到JSON文件
    with open('charset.json', 'w') as f:
        json.dump(charset, f, ensure_ascii=False, indent=4)
