# -*- coding: utf-8 -*-

import importlib
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from vocab_coverage.draw import draw_embeddings_graph
from vocab_coverage.loader import load_model, load_tokenizer, is_bbpe_tokenizer
from vocab_coverage.utils import logger, generate_embedding_filename, has_parameter, release_resource, is_match_patterns
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
    model_patterns_token_type_ids = [
        # tiiuae/falcon-7b-instruct
        # ValueError: Got unexpected arguments: {'token_type_ids': tensor([[0]])}
        'falcon',
        # fnlp/bart-base-chinese
        # BartEncoder.forward() got an unexpected keyword argument 'token_type_ids'
        'bart',
        # xverse/XVERSE-13B-Chat
        # TypeError: XverseForCausalLM.forward() got an unexpected keyword argument 'token_type_ids'
        'xverse'
        ]
    if is_match_patterns(model_name, model_patterns_token_type_ids):
            del inputs['token_type_ids']
    return inputs

cache_model_fix_attention_mask = []
cache_model_fix_token_embeddings = []

def _prepare_attention_mask(model_name:str, attention_mask):
    if is_match_patterns(model_name, ['chatglm-6b']):
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
    if is_match_patterns(model_name, ['chatglm-6b']):
        # THUDM/chatglm-6b
        #   token_embeddings.shape: [4, 50, 4096] => [50, 4, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        if model_name not in cache_model_fix_token_embeddings:
            cache_model_fix_token_embeddings.append(model_name)
            logger.debug("[%s]: fix token_embeddings: %s => %s",
                        model_name, old_shape, token_embeddings.shape)
    elif is_match_patterns(model_name, ['chatglm2-6b']):
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

def get_output_embeddings_openai(model_name:str, lexicon:Lexicon, batch_size:int=1000, debug=False):
    Embedding = importlib.import_module('openai').Embedding
    embeddings = []
    texts = [item['text'] for _, value in lexicon for item in value['items']]
    with tqdm(total=len(texts), desc=f"get_output_embeddings_openai({model_name})") as pbar:
        for i in range(0, len(texts), batch_size):
            pbar.update(batch_size)
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = Embedding.create(input = batch_texts, model=model_name)['data']
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
                   batch_size:int=100,
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
    elif granularity in [constants.GRANULARITY_CHARACTER,
                         constants.GRANULARITY_WORD,
                         constants.GRANULARITY_SENTENCE,
                         constants.GRANULARITY_PARAGRAPH]:
        for _, value in lexicon:
            for item in value['items']:
                texts.append(item['text'])
    # batch calculation
    if batch_size is None:
        if granularity == constants.GRANULARITY_TOKEN:
            batch_size = 100
        elif granularity == constants.GRANULARITY_CHARACTER:
            batch_size = 100
        elif granularity == constants.GRANULARITY_WORD:
            batch_size = 100
        elif granularity == constants.GRANULARITY_SENTENCE:
            batch_size = 50
        elif granularity == constants.GRANULARITY_PARAGRAPH:
            batch_size = 20
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
                       no_cache:bool=False,
                       batch_size:int=100,
                       debug=False):
    logger.info("[%s] 对 [%s] @ %s embedding 进行可视化...", model_name, granularity, positions)

    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]

    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
        parts = model_name.lower().split('/')
        if len(parts) == 2:
            name = parts[-1]
            supported_models = [
                'text-similarity-babbage-001',
                'text-search-babbage-doc-001',
                'text-similarity-curie-001',
                'text-search-curie-doc-001',
                'text-embedding-ada-002',
            ]
            if not is_match_patterns(name, supported_models):
                logger.warning("[%s] only '%s' are supported, skip...", model_name, supported_models)
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
    if no_cache or not has_cache:
        # 如果没有缓存，重新计算
        if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
            cache_key = cache.key(model_name, granularity, constants.EMBEDDING_POSITION_OUTPUT, 'embeddings_2d')
            if not no_cache and cache.has(cache_key):
                # no need to calculate embeddings if we cached the embeddings_2d already, and there is no input embedding for OpenAI.
                embeddings = {}
            else:
                # call openai api to get embeddings
                openai_model_name = model_name.split('/')[-1]
                embeddings_openai = get_output_embeddings_openai(openai_model_name, lexicon, batch_size=batch_size, debug=debug)
                embeddings = { constants.EMBEDDING_POSITION_OUTPUT: embeddings_openai }
        else:
            model = load_model(model_name, debug=debug)
            embeddings = get_embeddings(model_name, model, tokenizer, lexicon, granularity=granularity, positions=positions, batch_size=batch_size, debug=debug)    # 处理不同位置的向量
            del model
            release_resource(model_name, clear_cache=False)
    for position in positions:
        cache_key = cache.key(model_name, granularity, position, 'embeddings_2d')
        if not no_cache and cache.has(cache_key):
            embeddings_2d = cache.get(cache_key)
            logger.info("[%s]: 从缓存中获取 Embedding 2D 向量(%s)...", model_name, cache_key)
        elif position in embeddings and embeddings[position] is not None and len(embeddings[position]) > 0:
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
        else:
            logger.warning("[%s]: %s_embeddings is None, skip...", model_name, position)
            continue
        # 为 lexicon 添加 tokenized_text 信息
        with tqdm(total=lexicon.get_item_count(), desc=f"为 lexicon 添加 tokenized_text 信息({position})") as pbar:
            if (granularity in [constants.GRANULARITY_TOKEN,
                                constants.GRANULARITY_CHARACTER,
                                constants.GRANULARITY_WORD]
                and not is_bbpe_tokenizer(model_name, tokenizer)):
                i = 0
                for _, value in lexicon:
                    for item in value['items']:
                        pbar.update(1)
                        text = item['text']
                        if hasattr(tokenizer, 'tokenize'):
                            kwargs = {}
                            if has_parameter(tokenizer.tokenize, 'add_special_tokens'):
                                kwargs['add_special_tokens'] = False
                            tokens = tokenizer.tokenize(text, **kwargs)
                        else:
                            # OpenAI - tiktoken
                            token_ids = tokenizer.encode(text)
                            tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
                        new_tokens = []
                        for token in tokens:
                            if token.startswith('Ġ'):
                                token = token[1:]
                            elif token.startswith('▁'):
                                token = token[1:]
                            if len(token) > 0:
                                new_tokens.append(token)
                        item['tokenized_text'] = new_tokens
        # 为 lexicon 添加 embedding 信息
        with tqdm(total=lexicon.get_item_count(), desc=f"为 lexicon 添加 embedding 信息({position})") as pbar:
            i = 0
            for _, value in lexicon:
                for item in value['items']:
                    pbar.update(1)
                    item['embedding'] = embeddings_2d[i]
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
