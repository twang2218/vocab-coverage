# -*- coding: utf-8 -*-

import importlib
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from vocab_coverage.draw import draw_embeddings_graph
from vocab_coverage.loader import load_model, load_tokenizer
from vocab_coverage.utils import generate_embedding_filename, has_parameter, release_resource, logger
from vocab_coverage.reducer import reduce_to_2d
from vocab_coverage.lexicon import Lexicon
from vocab_coverage.cache import cache
from vocab_coverage import constants

def get_token_embeddings(model_name:str, model:PreTrainedModel, debug:bool=False):
    if not hasattr(model, 'get_input_embeddings'):
        raise ValueError(f"[{model_name}]: get_token_embeddings(): cannot find 'model.get_input_embeddings()'")
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, torch.nn.Embedding):
        raise ValueError(f"[{model_name}]: get_token_embeddings(): unknown input_embedding_func: {embedding}")
    if debug:
        logger.debug("[%s]: get_token_embeddings(): embedding:[weight.shape: %s, num_embeddings:%s]", model_name, embedding.weight.shape, embedding.num_embeddings)
    # embedding = embedding.weight
    token_ids = torch.tensor(np.arange(0, embedding.num_embeddings, 1)).to(model.device)
    embedding = embedding(token_ids)
    if embedding.is_cuda:
        embedding = embedding.cpu()
    return embedding.float().detach().numpy()

def get_sentences_embeddings(model_name:str, model:PreTrainedModel, tokenizer:PreTrainedTokenizer,
                           sentences:List[str]|List[int],
                           positions:List[str]=None,
                           max_length:int=256):
    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    embeddings = {position: [] for position in positions}
    # 检查给入ID是否超过词表大小
    if isinstance(sentences[0], int):
        max_id = max(sentences)
        max_id_index = sentences.index(max_id)
        num_embeddings = model.get_input_embeddings().num_embeddings
        if max_id >= num_embeddings:
            logger.error("[%s]: get_sentences_embeddings(): max_id: [%s]=%s >= num_embeddings %s", model_name, max_id_index, max_id, num_embeddings)

    inputs = _get_inputs(model_name, tokenizer, sentences, max_length)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # get attention_mask and token_embeddings
    attention_mask = _prepare_attention_mask(model_name=model_name, attention_mask=inputs['attention_mask'])

    for position in positions:
        if position == constants.EMBEDDING_POSITION_INPUT:
            token_embeddings = outputs.hidden_states[0].detach().clone()
        else:
            token_embeddings = outputs.hidden_states[-1].detach().clone()
        token_embeddings = _prepare_token_embeddings(model_name=model_name, token_embeddings=token_embeddings)
        sentence_embeddings = _calculate_sentence_embedding_mean_pooling(token_embeddings, attention_mask)

        embeddings[position] = sentence_embeddings

    return embeddings

def _get_inputs(model_name:str,
                tokenizer:PreTrainedTokenizer,
                sentences:List[str]|List[int],
                max_length:int=256) -> BatchEncoding:
    kwargs = {
        'max_length': max_length,
        'padding': True,
        'truncation': True,
        'add_special_tokens': False,
        'return_tensors': 'pt'
    }
    if not isinstance(sentences, list):
        raise TypeError(f"[{model_name}]: get_sentences_embedding(): sentences must be a list of str or int")
    if all(isinstance(s, int) for s in sentences):
        ## sentences is a list of token_ids
        placeholder = ['a'] * len(sentences)
        inputs = tokenizer(placeholder, **kwargs)
        for i, token_id in enumerate(sentences):
            # 替换占位符 'a' 的 token_id 为真实的 token_id
            inputs['input_ids'][i][-1] = token_id
    elif all(isinstance(s, str) for s in sentences):
        ## sentences is a list of str
        inputs = tokenizer(sentences, **kwargs)
    else:
        raise TypeError(f"[{model_name}]: get_sentences_embedding(): sentences must be a list of str or int")

    # post-processing
    if 'falcon' in model_name.lower():
        # tiiuae/falcon-7b-instruct
        # ValueError: Got unexpected arguments: {'token_type_ids': tensor([[0]])}
        del inputs['token_type_ids']
    elif 'bart' in model_name.lower():
        # fnlp/bart-base-chinese
        # BartEncoder.forward() got an unexpected keyword argument 'token_type_ids'
        del inputs['token_type_ids']
    return inputs

cache_model_fix_attention_mask = []
cache_model_fix_token_embeddings = []

def _prepare_attention_mask(model_name:str, attention_mask):
    model_name = model_name.lower()
    if 'chatglm-6b' in model_name:
        # THUDM/chatglm-6b
        #   attention_mask.shape: [50, 1, 4, 4] => [50, 4]
        old_shape = attention_mask.shape
        attention_mask = torch.where(attention_mask[:, 0, -1], torch.tensor(0), torch.tensor(1))
        if model_name not in cache_model_fix_attention_mask:
            cache_model_fix_attention_mask.append(model_name)
            logger.debug("[%s]: fix attention_mask: %s => %s",
                        model_name, old_shape, attention_mask.shape)
    return attention_mask

def _prepare_token_embeddings(model_name:str, token_embeddings):
    model_name = model_name.lower()
    if 'chatglm-6b' in model_name:
        # THUDM/chatglm-6b
        #   token_embeddings.shape: [4, 50, 4096] => [50, 4, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        if model_name not in cache_model_fix_token_embeddings:
            cache_model_fix_token_embeddings.append(model_name)
            logger.debug("[%s]: fix token_embeddings: %s => %s",
                        model_name, old_shape, token_embeddings.shape)
    elif 'chatglm2-6b' in model_name:
        # THUDM/chatglm2-6b
        #   attention_mask.shape: [50, 7]
        #   token_embeddings.shape: [7, 50, 4096] => [50, 7, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        if model_name not in cache_model_fix_token_embeddings:
            cache_model_fix_token_embeddings.append(model_name)
            logger.debug("[%s]: fix token_embeddings: %s => %s",
                        model_name, old_shape, token_embeddings.shape)
    return token_embeddings

def _calculate_sentence_embedding_mean_pooling(token_embeddings:torch.Tensor,
                                  attention_mask:torch.Tensor) -> np.ndarray:
    # Calculate of Sentences Embedding by the averaging the all token vectors
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Convert to numpy
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    embeddings = embeddings.detach().numpy()
    return embeddings

def get_output_embeddings_openai(model_name:str, vocab:List[str], batch=10, debug=False):
    Embedding = importlib.import_module('openai.Embedding')
    embeddings = []
    for i in range(0, len(vocab), batch):
        if debug:
            logger.debug("[%s]: get_output_embeddings_openai(): [%d:%d]", model_name, i, i+batch)
        batch_embeddings = Embedding.create(input = vocab[i:i+batch], model=model_name)['data']
        # ee = [e['embedding'] for e in ee]
        embeddings.extend(e['embedding'] for e in batch_embeddings)

    if debug:
        logger.debug("[%s] embeds: %d", model_name, len(embeddings))
    return np.array(embeddings)

def get_embeddings(model_name:str,
                   model:PreTrainedModel,
                   tokenizer:PreTrainedTokenizer,
                   lexicon:Lexicon,
                   granularity:str=constants.GRANULARITY_TOKEN,
                   positions:List[str]=None,
                   debug:bool=False):
    if debug:
        logger.debug("[%s]: get_embeddings(): granularity: %s, positions: %s",
                     model_name, granularity, positions)
    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    embeddings = {position: [] for position in positions}
    texts = []
    if granularity == constants.GRANULARITY_TOKEN:
        if constants.EMBEDDING_POSITION_INPUT in positions:
            embeddings[constants.EMBEDDING_POSITION_INPUT] = get_token_embeddings(model_name, model, debug=debug)
        if constants.EMBEDDING_POSITION_OUTPUT in positions:
            for _, value in lexicon:
                for i, item in enumerate(value['items']):
                    token_id = item['id']
                    if hasattr(model, 'get_input_embeddings') and hasattr(model.get_input_embeddings(), 'num_embeddings'):
                        # 检查给入ID是否超过词表大小
                        num_embeddings = model.get_input_embeddings().num_embeddings
                        if token_id >= num_embeddings:
                            logger.warning("[%s]: get_embeddings(): [%s]:'%s' id: %s >= num_embeddings: %s. use (num_embeddings - 1)=%s",
                                        model_name, i, item['text'], token_id, num_embeddings, num_embeddings-1)
                            token_id = num_embeddings - 1
                    texts.append(token_id)
    elif granularity == constants.GRANULARITY_CHARACTER:
        for _, value in lexicon:
            for item in value['items']:
                texts.append(item['text'])
    elif granularity == constants.GRANULARITY_WORD:
        for _, value in lexicon:
            for item in value['items']:
                texts.append(item['text'])
    # batch calculation
    batch_size = 100
    logger.debug("[%s]: get_embeddings(): batch_size: %d", model_name, batch_size)
    progress = tqdm(range(0, len(texts), batch_size))
    for i in progress:
        progress.set_description(f"get_sentences_embeddings({i}/{len(texts)})")
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = get_sentences_embeddings(model_name, model, tokenizer, batch_texts, positions)
        for position in positions:
            if i == 0:
                embeddings[position] = batch_embeddings[position]
            else:
                embeddings[position] = np.concatenate((embeddings[position], batch_embeddings[position]), axis=0)
            # if debug:
            #     logger.debug("[%s]: batch_embeddings: %s, embeddings: %s", model_name, batch_embeddings[position].shape, embeddings[position].shape)
        # if debug:
        #     for i, text in enumerate(batch_texts):
        #         logger.debug("[%s]: %s: %s", model_name, text, batch_embeddings[i][:5])
    return embeddings

def embedding_analysis(model_name:str,
                       lexicon:Lexicon,
                       positions=None,
                       granularity:str=constants.GRANULARITY_TOKEN,
                       reducer=constants.REDUCER_TSNE,
                       folder:str=constants.FOLDER_IMAGES,
                       postfix:str='',
                       override:bool=False,
                       debug=False):
    logger.info("对模型 %s 的 embedding 进行可视化...", model_name)

    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]

    if '/' in model_name:
        org, name = model_name.split('/')
        if org.lower() == 'openai' and name != 'text-embedding-ada-002':
            logger.warning("[%s] only 'text-embedding-ada-002' is supported, skip...", model_name)
            return

    # 生成文件名
    positions_candidates = []
    filenames = {}
    for position in positions:
        filename = generate_embedding_filename(model_name=model_name,
                                                granularity=granularity,
                                                position=position,
                                                postfix=postfix,
                                                folder=folder)
        ## 跳过已存在的文件
        if not override and os.path.exists(filename):
            logger.warning("[%s]: %s exists, skip...", model_name, filename)
        else:
            filenames[position] = filename
            positions_candidates.append(position)
    if len(positions_candidates) == 0:
        logger.warning("[%s]: no positions to process, skip...", model_name)
        return
    positions = positions_candidates

    # 获取向量
    tokenizer = load_tokenizer(model_name, debug=debug)
    has_cache = all(cache.has(cache.key(model_name, granularity, position, 'embeddings_2d')) for position in positions)
    if not has_cache:
        # 如果没有缓存，重新计算
        model = load_model(model_name, debug=debug)
        embeddings = get_embeddings(model_name, model, tokenizer, lexicon, granularity=granularity, positions=positions, debug=debug)    # 处理不同位置的向量
        del model
        release_resource(model_name, clear_cache=False)
    for position in positions:
        cache_key = cache.key(model_name, granularity, position, 'embeddings_2d')
        if cache.has(cache_key) or (embeddings[position] is not None and len(embeddings[position]) > 0):
            if cache.has(cache_key):
                embeddings_2d = cache.get(cache_key)
                logger.info("[%s]: 从缓存中获取 Embedding 2D 向量(%s)...", model_name, cache_key)
            else:
                # 检查
                ## 检查长度是否一致
                vocab_size = lexicon.get_item_count()
                if len(embeddings[position]) != vocab_size:
                    logger.warning("[%s]: %s_embeddings %s != vocab_size %s, skip...", model_name, position, len(embeddings[position]), vocab_size)
                    continue
                ## 检查是否有 nan
                i = 0
                for _, value in lexicon:
                    for item in value['items']:
                        if np.isnan(embeddings[position][i]).any():
                            logger.warning("[%s]: [%d]: '%s' embedding: (%s): %s", model_name, i, item['text'], np.shape(embeddings[position][i]), embeddings[position][i])
                        i += 1
                # 降维
                embeddings_2d = reduce_to_2d(embeddings[position], method=reducer, shuffle=True, debug=debug)
                # 缓存
                cache.set(cache_key, embeddings_2d)
            # 为 lexicon 添加 embedding 信息
            i = 0
            for _, value in lexicon:
                for item in value['items']:
                    item['embedding'] = embeddings_2d[i]
                    kwargs = {}
                    if has_parameter(tokenizer.tokenize, 'add_special_tokens'):
                        kwargs['add_special_tokens'] = False
                    text = item['text']
                    if isinstance(text, bytes):
                        # Qwen/Qwen-7B-Chat
                        text = text.decode('utf-8')
                    tokenized_text = tokenizer.tokenize(text, **kwargs)
                    item['tokenized_text'] = [t for t in tokenized_text if t != constants.TEXT_LEADING_UNDERSCORE]
                    i += 1
            # 绘制图像
            if debug:
                logger.debug("[%s]: draw %s_embeddings %s...",
                             model_name, position, embeddings_2d.shape)
            image = draw_embeddings_graph(
                model_name=model_name,
                lexicon=lexicon,
                position=position,
                debug=debug)

            # 保存图像到文件
            if image is None:
                logger.warning("[%s]: 生成 [%s] %s 向量图像失败...", model_name, granularity, position)
                continue
            
            filename = filenames[position]
            logger.info("[%s]: 保存 [%s] %s 向量图像 (%s)...", model_name, granularity, position, filename)
            image.save(filename, quality=80, optimize=True)
    return
