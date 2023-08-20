# -*- coding: utf-8 -*-

from typing import List
from tqdm import tqdm

from vocab_coverage.classifier import Classifier, load_classifier
from vocab_coverage.loader import load_wordbook
from vocab_coverage.utils import logger
from vocab_coverage import constants

class Lexicon:
    def __init__(self, classifier:Classifier, wordbook:List[str]|List[dict]|dict[str, dict] = None):
        self.classifier = classifier
        # 词表结构
        # {
        #   category: {
        #       name: '',
        #       color: '',
        #       items: [
        #           {id: '', text: ''},
        #           {id: '', text: ''},
        #           ...
        #       ],
        # }
        self.lexicon = {}
        self.set_lexicon(wordbook)

    def __len__(self):
        return len(self.lexicon)

    def __iter__(self):
        for category, value in self.lexicon.items():
            yield category, value

    def set_lexicon(self, wordbook:List[str]|List[dict]|dict[str, dict]):
        if wordbook is None:
            self.lexicon = self._handle_categories(self.classifier.get_categories())
        elif isinstance(wordbook, list):
            if all(isinstance(item, str) for item in wordbook) or all(isinstance(item, bytes) for item in wordbook):
                self.lexicon = self._handle_list_str(wordbook)
            elif all(isinstance(item, dict) for item in wordbook):
                self.lexicon = self._handle_list_dict(wordbook)
            else:
                logger.debug(wordbook)
                raise ValueError('unsupported lexicon type')
        elif isinstance(wordbook, dict):
            self.lexicon = self._handle_dict(wordbook)
        else:
            raise ValueError(f'unsupported lexicon type. ({type(wordbook)})')
        logger.debug('Lexicon 词表加载完成 (%s)', self.get_item_count())

    def get_categories(self) -> List[str]:
        return list(self.classifier.get_categories().keys())

    def get_category(self, category:str) -> dict:
        return self.lexicon[category]

    def get_item_count(self) -> int:
        return sum(len(value['items']) for value in self.lexicon.values())

    def get_granularity(self) -> str:
        return self.classifier.granularity

    def _handle_categories(self, categories:dict[str, dict]) -> dict[str, list]:
        logger.debug('Lexicon 加载词表，词表为分类器的字典')
        lexicon = {}
        for category, value in categories.items():
            lexicon[category] = {
                'name': category,
                'color': value['color'],
                'items': [{'text': text} for text in value['texts']]
            }
        return lexicon

    def _handle_list_str(self, wordbook:List[str]) -> dict[str, list]:
        logger.debug('Lexicon 加载词表，词表为字符串列表')
        # 词表是一个字符串列表
        lexicon = {}
        for category in self.classifier.get_categories():
            lexicon[category] = {
                'name': category,
                'color': self.classifier[category]['color'],
                'items': []
            }
        # 将词表中的每个字符串归类到对应的类别中
        for text in wordbook:
            if len(text) > 0:
                # ziqingyang/chinese-llama-2-7b 会出现空字符串
                category = self.classifier.classify(text)
                if category is None:
                    raise ValueError(f'无法判别文本类别："{text}"')
                if category not in lexicon:
                    raise ValueError(f'词表中的类别({category})不在分类器中')
                lexicon[category]['items'].append({'text': text})
        # 删除空类别
        empty_categories = []
        for category, value in lexicon.items():
            if len(value['items']) == 0:
                empty_categories.append(category)
        for category in empty_categories:
            del lexicon[category]

        return lexicon

    def _handle_list_dict(self, wordbook:List[dict]) -> dict[str, list]:
        logger.debug('Lexicon 加载词表，词表为字典列表，每个字典包含 id 和 text 两个字段')
        # 词表是一个字典列表，每个字典包含 id 和 text 两个字段
        lexicon = {}
        for category in self.classifier.get_categories():
            lexicon[category] = {
                'name': category,
                'color': self.classifier[category]['color'],
                'items': []
            }
        # 将词表中的每个字符串归类到对应的类别中
        for item in tqdm(wordbook, desc='Lexicon 加载词表'):
            text = item['text']
            category = self.classifier.classify(text)
            if category is None:
                # 无需退出，忽略无法判别的文本即可
                logger.error('无法判别文本类别："%s"', text)
                continue
            if category not in lexicon:
                raise ValueError(f'词表中的类别({category})不在分类器中')
            lexicon[category]['items'].append({'id': item['id'], 'text': text})
        # 删除空类别
        empty_categories = []
        for category, value in lexicon.items():
            if len(value['items']) == 0:
                empty_categories.append(category)
        for category in empty_categories:
            del lexicon[category]
        return lexicon

    def _handle_dict(self, wordbook:dict[str, list]) -> dict[str, list]:
        logger.debug('Lexicon 加载词表，词表为字典，每个字典的 key 是类别，value 是该类别的语料')
        # 词表是一个字典，每个字典的 key 是类别，value 是该类别的语料
        # 确认lexicon中的类别是否在classifier中存在
        for category, value in wordbook.items():
            if category not in self.classifier.get_categories():
                classifier_cats = self.classifier.get_categories().keys()
                raise ValueError(f'词表中的类别({category})不在分类器中({classifier_cats})')
            value['name'] = category
            value['color'] = self.classifier[category]['color']
            for item in value['items']:
                if 'text' not in item:
                    raise ValueError(f'词表中的类别({category})缺少texts字段')
        return wordbook

def load_lexicon(model_name:str, granularity:str, debug:bool=False) -> Lexicon:
    cls = load_classifier(granularity=granularity)
    wordbook = load_wordbook(model_name, granularity, debug)
    lexicon = Lexicon(cls, wordbook)
    return lexicon
