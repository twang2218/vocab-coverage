# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import traceback
from typing import List

import numpy as np
import torch

from vocab_coverage.draw import draw_vocab_embeddings
from vocab_coverage.loader import load_model, load_tokenizer
from vocab_coverage.utils import show_gpu_usage, release_resource, logger
from vocab_coverage.reducer import reduce_to_2d


EMBEDDING_POSITION_INPUT = 'input'
EMBEDDING_POSITION_OUTPUT = 'output'
EMBEDDING_POSITION_ALL = [
    EMBEDDING_POSITION_INPUT,
    EMBEDDING_POSITION_OUTPUT,
]

EMBEDDING_GRANULARITY_TOKEN = 'token'
EMBEDDING_GRANULARITY_CHAR = 'char'
EMBEDDING_GRANULARITY_WORD = 'word'
EMBEDDING_GRANULARITY_SENTENCE = 'sentence'
EMBEDDING_GRANULARITY_ALL = [
    EMBEDDING_GRANULARITY_TOKEN,
    EMBEDDING_GRANULARITY_CHAR,
    EMBEDDING_GRANULARITY_WORD,
    EMBEDDING_GRANULARITY_SENTENCE,
]

def get_vocab_token(model_name:str, tokenizer, debug=False):
    if "OpenAI" in model_name:
        model_name = model_name.split("/")[-1]
        return get_vocab_token_openai(model_name, debug=debug)

    vocab_size = max([id for id in tokenizer.get_vocab().values()]) + 1
    if debug:
        logger.debug(f"[{model_name}]: vocab_size: {vocab_size}")

    # get vocab
    vocab = [''] * (vocab_size)
    for k, v in tokenizer.get_vocab().items():
        if v >= vocab_size:
            # to avoid additional tokens which not in vocab
            logger.warning(f"[{model_name}] out of range: {k}, {v}")
            continue
        try:
            if isinstance(k, bytes):
                # Qwen/Qwen-7B-Chat
                vocab[v] = k
            elif hasattr(tokenizer, 'convert_tokens_to_string'):
                vocab[v] = tokenizer.convert_tokens_to_string([k])
            elif hasattr(tokenizer, 'text_tokenizer') and hasattr(tokenizer.text_tokenizer, 'convert_tokens_to_string'):
                # BAAI/aquila-7b
                vocab[v] = tokenizer.text_tokenizer.convert_tokens_to_string([k])
            else:
                vocab[v] = k
        except Exception as e:
            logger.error(f"[{model_name}]: convert_tokens_to_string({k}) failed: {e}")
            vocab[v] = k
    return vocab

def get_vocab_token_openai(model_name:str, debug=False):
    import tiktoken
    t = tiktoken.encoding_for_model('gpt-3.5-turbo')
    count_except = 0
    vocab = []
    for k in t._mergeable_ranks.keys():
        try:
            vocab.append(str(k, encoding='utf-8'))
        except:
            # logger.debug(str(k))
            count_except += 1
            vocab.append(str(k))
    if debug:
        logger.debug(f"[{model_name}]: vocab: {len(vocab)}")
        logger.debug(f"[{model_name}]: count_except: {count_except}")
    return vocab

def get_charset(file:str=''):
    if len(file) == 0:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charsets.json')
    charset = []
    with open(file, 'r') as f:
        charsets = json.load(f)
        for _, chars in charsets.items():
            charset.extend(chars)
        charset = list(set(charset))
    return charset

def get_vocab(model_name:str, model, tokenizer, granularity:str=EMBEDDING_GRANULARITY_TOKEN, debug=False):
    if granularity == EMBEDDING_GRANULARITY_TOKEN:
        vocab = get_vocab_token(model_name, tokenizer, debug=debug)
        # truncate vocab according to model
        if hasattr(model, 'get_input_embeddings') and hasattr(model.get_input_embeddings(), 'weight'):
            tokenizer_vocab_size = len(vocab)
            model_vocab_size = model.get_input_embeddings().weight.shape[0]
            if tokenizer_vocab_size > model_vocab_size:
                logger.warning(f"[{model_name}]: tokenizer_vocab_size({tokenizer_vocab_size}) > model_vocab_size({model_vocab_size}), will truncate the model vocab_size...")
                vocab = vocab[:model_vocab_size]
        return vocab
    elif granularity == EMBEDDING_GRANULARITY_CHAR:
        return get_charset()
    else:
        raise Exception(f"[{model_name}]: unknown granularity: {granularity}")

def get_token_embeddings(model_name, model, tokenizer, vocab, debug=False):
    input_embeddings = []
    try:
        if "OpenAI" in model_name:
            logger.error(f"[{model_name}]: Cannot retrieve input embeddings from OpenAI models.")
            return None

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'embedding') and hasattr(model.transformer.embedding, 'word_embeddings'):
            # THUDM/chatglm2-6b
            input_embedding_func = model.transformer.embedding.word_embeddings
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'get_input_embeddings'):
            # THUDM/chatglm-6b
            input_embedding_func = model.transformer.get_input_embeddings()
        elif hasattr(model, 'tok_embeddings'):
            # BAAI/aquila-7b
            input_embedding_func = model.tok_embeddings
        elif hasattr(model, 'get_input_embeddings'):
            # most Transformers
            input_embedding_func = model.get_input_embeddings()
        else:
            logger.error(f"[{model_name}]: cannot find 'model.get_input_embeddings()'")
            logger.debug(model)
            raise Exception(f"[{model_name}]: cannot find 'model.get_input_embeddings()'")

        if debug:
            logger.debug(f"[{model_name}]: get_token_embeddings(): {input_embedding_func}")

        vocab_size = len(vocab)
        if hasattr(input_embedding_func, 'weight'):
            # shibing624/prompt-t5-base-chinese: tokenizer.vocab_size=32228, get_input_embeddings().weight=(32128, 768)
            vocab_size = min(input_embedding_func.weight.shape[0], vocab_size)
        token_ids = torch.tensor(np.arange(0, vocab_size, 1)).to(model.device)
        input_embeddings = input_embedding_func(token_ids)
        if input_embeddings.is_cuda:
            input_embeddings = input_embeddings.cpu()
        # Qwen/Qwen-7B-Chat: Got unsupported ScalarType BFloat16
        input_embeddings = input_embeddings.float()
        input_embeddings = input_embeddings.detach().numpy()

        if debug:
            logger.debug(f"[{model_name}]: input_embeddings: {input_embeddings.shape}")
    except Exception as e:
        logger.error(f"[{model_name}]: get_token_embeddings failed: {e}")
        traceback.print_exc()
        logger.debug(model)
        raise e
    return input_embeddings

def do_sentences_embeddings(model_name, model, tokenizer,
                            sentences:List[str],
                            granularity=EMBEDDING_GRANULARITY_TOKEN,
                            position=EMBEDDING_POSITION_OUTPUT,
                            max_length=256):
    # from https://github.com/shibing624/text2vec/blob/master/text2vec/sentence_model.py#L96
    kwargs = {
        'max_length': max_length,
        'padding': True,
        'truncation': True,
        'add_special_tokens': False,
        'return_tensors': 'pt'
    }
    # To avoid token be split by tokenizer, we construct the inputs by using token ids directly
    # to do that, we generate the all the value by using unsplittable token, such as 'a',
    # then replace the id by the real token id
    if granularity == EMBEDDING_GRANULARITY_TOKEN:
        placeholder = ['a'] * len(sentences)
        inputs = tokenizer(placeholder, **kwargs)
        # if "qwen" in model_name.lower():
        #     sentences = [s.encode('UTF-8') for s in sentences]
        sentences_ids = tokenizer.convert_tokens_to_ids(sentences)

        for i, id in enumerate(sentences_ids):
            if id is None:
                logger.warning(f"[{model_name}]: [{i}] '{sentences[i]}' => {id}")
                continue
        for i, id in enumerate(sentences_ids):
            inputs['input_ids'][i][-1] = id
    else:
        inputs = tokenizer(sentences, **kwargs)

    # move inputs to device
    inputs = inputs.to(model.device)

    try:
        if "/falcon-" in model_name:
            # tiiuae/falcon-7b-instruct
            del inputs['token_type_ids']
        outputs = model(**inputs, output_hidden_states=True)
    except Exception as e:
        # google/flan-t5-base
        if hasattr(model, 'get_encoder'):
            outputs = model.get_encoder()(**inputs, output_hidden_states=True)
        else:
            logger.error(f"[{model_name}]: get_sentences_embeddings() failed: {e}")
            traceback.print_exc()
            logger.debug(model)
            raise e

    # get attention_mask and token_embeddings
    # logger.debug(f"[{model_name}]: input_ids: {inputs['input_ids'].shape}, attention_mask: {inputs['attention_mask'].shape}")
    attention_mask = inputs['attention_mask']
    del inputs
    all_hidden_states = []
    for hs in outputs.hidden_states:
        all_hidden_states.append(hs.detach().clone())
    
    hidden_state_index = -1
    if position == EMBEDDING_POSITION_INPUT:
        hidden_state_index = 0
    token_embeddings = outputs.hidden_states[hidden_state_index].detach().clone()
    del outputs

    if 'chatglm-6b' in model_name:
        # THUDM/chatglm-6b
        #   attention_mask.shape: [50, 1, 4, 4] => [50, 4]
        old_shape = attention_mask.shape
        attention_mask = torch.where(attention_mask[:, 0, -1], torch.tensor(0), torch.tensor(1))
        logger.debug(f"[{model_name}]: fix attention_mask: {old_shape} => {attention_mask.shape}")
        #   token_embeddings.shape: [4, 50, 4096] => [50, 4, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        logger.debug(f"[{model_name}]: fix token_embeddings: {old_shape} => {token_embeddings.shape}")
    elif 'chatglm2-6b' in model_name:
        # THUDM/chatglm2-6b
        #   attention_mask.shape: [50, 7]
        #   token_embeddings.shape: [7, 50, 4096] => [50, 7, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        logger.debug(f"[{model_name}]: fix token_embeddings: {old_shape} => {token_embeddings.shape}")

    # Calculate of Sentences Embedding by the averaging the all token vectors
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Convert to numpy
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    embeddings = embeddings.detach().numpy()

    for i, e in enumerate(embeddings):
        if np.isnan(e).any():
            logger.warning(f"[{model_name}]: embeddings for {i}:'{sentences[i]}' contains NaN")
            # logger.debug(f"[{model_name}]: > attention_mask({attention_mask[i].shape}):  {attention_mask[i]}")
            # logger.debug(f"[{model_name}]: > token_embeddings({token_embeddings[i].shape}): {token_embeddings[i]}")
            logger.debug(f"[{model_name}]: > all_hidden_states: {len(all_hidden_states)}")
            for j, hs in enumerate(all_hidden_states):
                if np.isnan(hs[i]).any():
                    logger.debug(f"[{model_name}]: > all_hidden_states[{j}]({hs[i].shape}): {hs[i]}")
            # logger.debug(f"[{model_name}]: > input_mask_expanded({input_mask_expanded[i].shape}): {input_mask_expanded[i]}")
    return embeddings

def get_sentence_embeddings(model_name, model, tokenizer, vocab,
                            granularity=EMBEDDING_GRANULARITY_TOKEN,
                            position=EMBEDDING_POSITION_OUTPUT,
                            batch_size=0,
                            debug=False):
    embeddings = []
    try:
        if "OpenAI" in model_name:
            model_name = model_name.split("/")[-1]
            return get_output_embeddings_openai(model_name, vocab, batch=2000, debug=debug)

        memory = show_gpu_usage(model_name)
        if batch_size == 0:
            ## 根据 GPU 内存大小自动调整 batch_size，（粗略计算，不一定充分利用 GPU 内存）
            if memory['total'] > 0:
                batch_size = round((memory['free']//18)/200) * 200
            else:
                batch_size = 100
        logger.info(f"[{model_name}]: batch_size: {batch_size}")

        # get sentence embedding in batch
        for i in range(0, len(vocab), batch_size):
            batch_sentences = vocab[i:i+batch_size]
            batch_embeddings = do_sentences_embeddings(model_name, model, tokenizer, 
                                                       sentences=batch_sentences, 
                                                       granularity=granularity, 
                                                       position=position,
                                                       max_length=5)
            if i == 0:
                embeddings = batch_embeddings
            else:
                embeddings = np.concatenate((embeddings, batch_embeddings))
            logger.info(f"[{model_name}]: batch_embeddings: {batch_embeddings.shape}, embeddings: {embeddings.shape}")

    except Exception as e:
        logger.error(f"[{model_name}]: get_output_embedding failed: {e}")
        traceback.print_exc()
        logger.debug(model)
        raise e
    return embeddings

def get_output_embeddings_openai(model_name:str, vocab:List[str], batch=10, debug=False):
    import openai
    embeds = []
    for i in range(0, len(vocab), batch):
        if debug:
            logger.debug(f"[{model_name}]: get_output_embeddings_openai(): {i}")
        ee = openai.Embedding.create(input = vocab[i:i+batch], model=model_name)['data']
        ee = [e['embedding'] for e in ee]
        if debug:
            logger.debug(f"[{model_name}]: Retrieved {len(ee)} embeddings for {vocab[i:i+batch]}")
        embeds.extend(ee)

    if debug:
        logger.debug(f"embeds: {len(embeds)}")
    return np.array(embeds)

def get_embeddings(model_name:str, model, tokenizer, vocab, 
                   granularity=EMBEDDING_GRANULARITY_TOKEN,
                   position=EMBEDDING_POSITION_INPUT,
                   debug=False):
    if position == EMBEDDING_POSITION_INPUT:
        if granularity == EMBEDDING_GRANULARITY_TOKEN:
            return get_token_embeddings(model_name, model, tokenizer, vocab, debug=debug)
        else:
            return get_sentence_embeddings(model_name, model, tokenizer, vocab, granularity=granularity, position=position, debug=debug)
    elif position == EMBEDDING_POSITION_OUTPUT:
        return get_sentence_embeddings(model_name, model, tokenizer, vocab, granularity=granularity, position=position, debug=debug)
    else:
        logger.error(f"[{model_name}]: unknown embedding_position: {position}")
        return None

def do_embedding_analysis(model_name:str, embeddings, vocab, charsets:dict, is_detailed=False, folder=None, position=EMBEDDING_POSITION_INPUT, reducer_method='tsne', debug=False):
    if debug:
        logger.debug(f"[{model_name}]: reducing the dimension of '{position}_embeddings' {embeddings.shape} to 2D by {reducer_method}...")
    embeddings_2d = reduce_to_2d(embeddings, method=reducer_method, debug=debug)
    if debug:
        logger.debug(f"[{model_name}]: draw {position}_embeddings {embeddings_2d.shape}...")
    image = draw_vocab_embeddings(
        model_name=model_name,
        embeddings_2d=embeddings_2d,
        vocab=vocab,
        charsets=charsets,
        position=position,
        width=8000,
        height=8000,
        is_detailed=is_detailed,
        debug=debug)

    return image

def generate_embedding_filename(model_name:str, granularity:str, position:str, postfix:str='', flat=False, output_dir:str=''):
    # 根据参数生成文件基础文件名
    output_file = model_name.replace('/', '_') + f'.embeddings.{granularity}.{position}.jpg'
    if len(postfix) > 0:
        output_file = output_file.replace('.jpg', f'.{postfix}.jpg')
    # 根据参数生成文件所在路径
    if len(output_dir) == 0:
        output_dir = 'images'
    if flat:
        workdir = output_dir
    else:
        workdir = os.path.join(output_dir, 'assets', 'embeddings')
    # 确保目录存在
    os.makedirs(workdir, exist_ok=True)
    return os.path.join(workdir, output_file)

def embedding_analysis(model_name:str, charsets:dict, output_dir:str, positions=[EMBEDDING_POSITION_INPUT], granularity:str=EMBEDDING_GRANULARITY_TOKEN, is_detailed=False, debug=False, reducer_method='tsne', clear_cache=False, postfix:str='', flat:bool=False, override:bool=False):
    logger.info("对模型 {} 的 embedding 进行可视化...".format(model_name))

    if '/' in model_name:
        org, name = model_name.split('/')
        if org.lower() == 'openai' and name != 'text-embedding-ada-002':
            logger.warning(f"Skip {model_name}, only 'text-embedding-ada-002' is supported...")
            return

    tokenizer = load_tokenizer(model_name, debug=debug)
    model = load_model(model_name, debug=debug)
    vocab = get_vocab(model_name, model, tokenizer, granularity=granularity, debug=debug)

    for position in positions:
        # 生成文件名
        filename = generate_embedding_filename(model_name=model_name,
                                               granularity=granularity,
                                               position=position,
                                               postfix=postfix,
                                               flat=flat,
                                               output_dir=output_dir)
        ## 跳过已存在的文件
        if not override and os.path.exists(filename):
            logger.warning(f"[{model_name}]: {filename} exists, skip...")
            continue

        # 获取词向量
        embeddings = get_embeddings(model_name, model, tokenizer, vocab, granularity=granularity, position=position, debug=debug)
        not_non_embeddings = []
        not_non_vocab = []
        for i, e in enumerate(embeddings):
            if np.isnan(e).any():
                logger.warning(f"[{model_name}]: [{i}]: '{vocab[i]}' embeddings: ({np.shape(e)}): {e}")
            else:
                not_non_embeddings.append(e)
                not_non_vocab.append(vocab[i])
        embeddings = np.array(not_non_embeddings)
        vocab = not_non_vocab

        # Qwen/Qwen-7B-Chat
        ## 其 vocab 中为 bytes，需要转换为方便可视化的 string
        if 'qwen' in model_name.lower():
            new_vocab = []
            for k in vocab:
                try:
                    k = bytearray(k).decode('utf-8')
                except:
                    # cannot decode in utf-8, so use "b'...'"
                    k = str(bytes(k))
                new_vocab.append(k)
            vocab = new_vocab

        release_resource(model_name, clear_cache=clear_cache)

        if embeddings is not None and len(embeddings) > 0:
            # 生成图像
            image = do_embedding_analysis(
                model_name=model_name,
                embeddings=embeddings,
                vocab=vocab,
                charsets=charsets,
                is_detailed=is_detailed,
                position=position,
                reducer_method=reducer_method,
                debug=debug)
            # 保存图像到文件
            if image is not None:
                if debug:
                    logger.debug(f"[{model_name}]: save {position}_embeddings to {filename}...")
                image.save(filename, quality=80, optimize=True)
            else:
                logger.warning(f"[{model_name}]: image is None, skip save to {filename}...")

    # clean up
    del tokenizer
    del model

    release_resource(model_name, clear_cache=clear_cache)

    return
