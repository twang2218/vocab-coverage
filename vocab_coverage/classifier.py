# -*- coding: utf-8 -*-
import json
import os
from vocab_coverage.utils import logger
from vocab_coverage import constants

class Classifier:
    def __init__(self, categories: dict[str, dict] = None, filename: str = None,
                 granularity:str=constants.GRANULARITY_CHARACTER):
        self.granularity = granularity
        if isinstance(categories, dict) and len(categories) > 0:
            self.categories = categories    # {category: {texts: [], color: ''}}
        elif isinstance(filename, str) and len(filename) > 0:
            self.load(filename)
        else:
            self.load(self._get_default_filename())
        self._text_category_index = self._create_text_category_index()

    def load(self, filename:str=None):
        if not(isinstance(filename, str) and len(filename) > 0):
            filename = self._get_default_filename()
        if filename.endswith('.json'):
            with open(filename, 'r', encoding='utf-8') as f:
                self.categories = json.load(f)
        else:
            raise ValueError(f'unsupported file type: "{filename}"')

    def save(self, filename:str=None, indent:int=None):
        if not(isinstance(filename, str) and len(filename) > 0):
            filename = self._get_default_filename()
        if not filename.endswith('.json'):
            raise ValueError(f'unsupported file type: "{filename}"')
        logger.info("保存分类器字典 '%s' ...", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, indent=indent, ensure_ascii=False)

    def classify(self, text:str) -> str:
        # 文本如果出现在类别的文本集中，则属于该类别
        if text in self._text_category_index:
            return self._text_category_index[text]
        return None

    def __getitem__(self, key:str) -> dict:
        return self.categories[key]

    def add_text(self, category:str, text:str):
        if category not in self.categories:
            self.categories[category] = {
                'texts': []
            }
        self.categories[category]['texts'].append(text)

    def get_categories(self) -> dict:
        return self.categories

    def _get_default_filename(self) -> str:
        basedir = os.path.dirname(os.path.abspath(__file__))
        filename = constants.FILE_CHARSET_DICT[self.granularity]
        filename = os.path.join(basedir, filename)
        return filename
    
    def _create_text_category_index(self) -> dict:
        index = {}
        for category, value in self.categories.items():
            for text in value['texts']:
                index[text] = category
        return index


class TokenClassifier(Classifier):
    def classify(self, text: str) -> dict:
        # 统计每个字符的类别归属
        candidates = {}

        # 处理字节流类型情况
        if isinstance(text, bytes):
            # Qwen/Qwen-7B-Chat
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                return constants.CHARSET_OTHER

        # 逐个字符判别类别
        for ch in text:
            category = super().classify(ch)
            # 对特殊类别（如英文、数字）进行特殊处理
            if category is None:
                if ch.isdigit() and self[constants.CHARSET_DIGIT] is not None:
                    category = constants.CHARSET_DIGIT
                elif ch.isalpha() and self[constants.CHARSET_ENGLISH] is not None:
                    category = constants.CHARSET_ENGLISH
            # 如果仍然无法判别，则归为其他
            if category is None:
                category = constants.CHARSET_OTHER
            # 统计类别
            if category in candidates:
                candidates[category] += 1
            else:
                candidates[category] = 1
        if len(candidates) == 0:
            return None
        # 根据类别归属统计判别文本类别
        ## 类别优先级
        category_priority = [
            constants.CHARSET_JAPANESE,           ## 如果字符中有日文假名，则判定为日文，而不因为有中文就判定为中文，因为日文中有汉字
            constants.CHARSET_KOREAN,             ## 如果字符中有韩文，则判定为韩文，而不因为有中文就判定为中文，因为韩文中有汉字
            constants.CHARSET_CHINESE_COMMON,     ## 如果字符中有汉字，则判定为汉字，不考虑是否包含英文、数字等
            constants.CHARSET_CHINESE_RARE
        ]
        final_category = None
        for category in category_priority:
            if category in candidates:
                final_category = category
                break
        if final_category is None:
            ## 选择出现次数最多的类别
            final_category = max(candidates, key=candidates.get)
        return final_category

def load_classifier(filename:str = '', granularity:str = constants.GRANULARITY_CHARACTER) -> Classifier:
    if granularity == constants.GRANULARITY_TOKEN:
        return TokenClassifier(filename=filename, granularity=granularity)
    return Classifier(filename=filename, granularity=granularity)
