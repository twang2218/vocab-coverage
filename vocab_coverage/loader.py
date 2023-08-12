# -*- coding: utf-8 -*-

import os
from typing import List
from importlib import import_module
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaTokenizerFast
)
import torch
import tiktoken

from vocab_coverage.utils import show_gpu_usage, logger
from vocab_coverage import constants

def load_tokenizer(model_name:str, debug:bool=False):
    try:
        kwargs = {}
        kwargs['trust_remote_code'] = True
        if 'llama' in model_name.lower() or 'vicuna' in model_name.lower() or 'alpaca' in model_name.lower():
            # https://github.com/LianjiaTech/BELLE/issues/242#issuecomment-1514330432
            # Avoid LlamaTokenizerFast conversion
            # lmsys/vicuna-7b-v1.3
            tokenizer = LlamaTokenizer.from_pretrained(model_name, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    # pylint: disable=invalid-name,broad-exception-caught
    except Exception as e:
        if "aquila" in e.args[0]:
            Tokenizer = import_module('flagai.data.tokenizer.Tokenizer')
            name = 'aquila-7b'
            cache_dir = os.path.join('./model', name)
            tokenizer = Tokenizer.from_pretrained(name, cache_dir=cache_dir)
            tokenizer.cls_token_id = tokenizer.token_start_id
            tokenizer.sep_token_id = tokenizer.token_end_id
            tokenizer.unk_token_id = tokenizer.get('token_unk_id', None)
            tokenizer.pad_token_id = tokenizer.get('token_pad_id', None)
            tokenizer.mask_token_id = tokenizer.get('token_mask_id', None)
            tokenizer.vocab_size = tokenizer.num_tokens
        elif "OpenAI" in e.args[0]:
            name = model_name.split("/")[-1]
            tokenizer = tiktoken.encoding_for_model(name)
            tokenizer.vocab_size = tokenizer.n_vocab
            tokenizer.cls_token_id = tokenizer.encode_single_token('<|endoftext|>')
            if debug:
                # pylint: disable=protected-access
                logger.debug(tokenizer._special_tokens)
        else:
            logger.error("加载模型 %s 失败：%s", model_name, e)
            raise e

    # https://github.com/huggingface/transformers/issues/24514
    if isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # https://github.com/huggingface/transformers/issues/22312
    if not tokenizer.pad_token:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            logger.warning("[%s]: 'tokenizer.pad_token' is None, set tokenizer.pad_token = tokenizer.eos_token (%s))",
                           model_name, tokenizer.eos_token)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("[%s]: 'tokenizer.pad_token' and 'tokenizer.eos_token' are None, set tokenizer.pad_token = '</s>'", model_name)
            tokenizer.bos_token = '<s>'
            tokenizer.eos_token = '</s>'
            tokenizer.unk_token = '<unk>'
            tokenizer.pad_token = tokenizer.eos_token

    if debug:
        logger.debug(tokenizer)

    return tokenizer

def _generate_model_kwargs(model_name:str):
    model_name_lower = model_name.lower()
    if 'openai' in model_name_lower:
        raise ValueError(f"不支持的模型：{model_name}")
    models_kwargs = {
        'chatglm-6b': {
            # THUDM/chatglm-6b
            'torch_dtype': torch.half,
            'trust_remote_code': True,
        },
        'chatglm2': {
            # THUDM/chatglm2-6b
            'trust_remote_code': True,
        },
        'falcon-7b': {
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'trust_remote_code': True,
        },
        'mpt-7b': {
            # tiiuae/falcon-7b-instruct
            # mosaicml/mpt-7b-instruct
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'trust_remote_code': True,
        },
        'qwen': {
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'trust_remote_code': True,
        },
        'oasst': {
            'load_in_4bit': True,
            'device_map': 'auto',
            'trust_remote_code': True,
        },
    }
    large_model_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    large_model_pattern = ['6b', '7b', '12b', '13b', 'llama', 'gpt', 'aquila', 'moss']
    for pattern, kwargs in models_kwargs.items():
        if pattern in model_name_lower:
            return kwargs
    for pattern in large_model_pattern:
        if pattern in model_name_lower:
            return large_model_kwargs
    return {}

def _get_model_class(model_name:str):
    models_classes = {
        'falcon': AutoModelForCausalLM,
    }
    for pattern, model_class in models_classes.items():
        if pattern in model_name.lower():
            if isinstance(model_class, str):
                model_class = import_module(model_class)
            return model_class
    return AutoModel

def load_model(model_name:str, debug:bool=False):
    if "OpenAI" in model_name:
        raise ValueError(f"不支持加载的模型：{model_name}")

    try:
        kwargs = _generate_model_kwargs(model_name)
        if debug:
            logger.debug("[%s]: AutoModel.from_pretrained(model_name=%s, kwargs=%s)",
                         model_name, model_name, kwargs)
        model_class = _get_model_class(model_name)
        model = model_class.from_pretrained(model_name, **kwargs)

        ## postprocess
        if hasattr(model, 'get_encoder'):
            # google/flan-t5-base
            ## for encoder-decoder model, we only need the encoder
            model = model.get_encoder()

    # pylint: disable=invalid-name,broad-exception-caught
    except Exception as e:
        logger.warning("[%s]: AutoModel.from_pretrained(model_name=%s, kwargs=%s) failed: %s, args: %s",
                       model_name, model_name, kwargs, e, e.args)
        if isinstance(e.args, (list, tuple)) and isinstance(e.args[0], str) and "AutoModel" in e.args[0]:
            logger.debug("[%s]: 试图加载 AutoModelForCausalLM", model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        elif isinstance(e.args, (list, tuple)) and isinstance(e.args[0], str) and "aquila" in e.args[0]:
            logger.debug("[%s]: 试图加载 flagai.model.aquila_model.AQUILAModel", model_name)
            # pylint: disable=invalid-name
            AQUILAModel = import_module('flagai.model.aquila_model.AQUILAModel')
            # cache_dir = os.path.join('./model', 'aquila-7b')
            # logger.debug(f"cache_dir: {os.path.abspath(cache_dir)}")
            model = AQUILAModel.from_pretrain(model_name='aquila-7b', download_path='./model')
        else:
            logger.error("加载 AutoModel 模型 %s 失败：%s", model_name, e)
            raise e

    if debug:
        num_parameters = f'{model.num_parameters():,}' if hasattr(model, 'num_parameters') else 'N/A'
        logger.debug("[%s]: num_parameters: %s", model_name, num_parameters)

    # ValueError: `.to` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    #   fnlp/moss-moon-003-sft-int4
    #   OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
    # if not should_use_4bit:
    #     if hasattr(torch, 'device'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("[%s]: %s model loaded on device: %s",
                model_name, model.__class__.__name__, model.device)

    if debug:
        show_gpu_usage(model_name)

    return model

def load_vocab(model_name:str, tokenizer, debug=False) -> List[str]:
    if "OpenAI" in model_name:
        model_name = model_name.split("/")[-1]
        return load_vocab_openai(model_name, debug=debug)

    vocab_size = max(id for id in tokenizer.get_vocab().values()) + 1
    if debug:
        logger.debug("[%s]: vocab_size: %s", model_name, vocab_size)

    # get vocab
    vocab = [''] * (vocab_size)
    for k, v in tokenizer.get_vocab().items():
        if v >= vocab_size:
            # to avoid additional tokens which not in vocab
            logger.warning("[%s]: out of range: %s, %s", model_name, k, v)
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
        # pylint: disable=broad-except,invalid-name
        except Exception as e:
            logger.error("[%s]: convert_tokens_to_string(%s) failed: %s", model_name, k, e)
            vocab[v] = k
    return vocab

def load_vocab_openai(model_name:str, debug=False):
    t = tiktoken.encoding_for_model('gpt-3.5-turbo')
    count_except = 0
    vocab = []
    # pylint: disable=protected-access
    for k in t._mergeable_ranks.keys():
        try:
            vocab.append(str(k, encoding='utf-8'))
        # pylint: disable=broad-except
        except Exception:
            # logger.debug(str(k))
            count_except += 1
            vocab.append(str(k))
    if debug:
        logger.debug("[%s]: vocab: %s", model_name, len(vocab))
        logger.debug("[%s]: count_except: %s", model_name, count_except)
    return vocab

def load_wordbook(model_name:str, granularity:str=constants.GRANULARITY_TOKEN, debug:bool=False) -> List[str]:
    if granularity == constants.GRANULARITY_TOKEN:
        tokenizer = load_tokenizer(model_name, debug=debug)
        vocab = load_vocab(model_name, tokenizer, debug=debug)
        wordbook = [{'id': i, 'text': v} for i, v in enumerate(vocab)]
        return wordbook
    elif granularity == constants.GRANULARITY_CHARACTER:
        # 返回 None，表示将使用 classifier 的内容做语料
        return None
    else:
        raise ValueError(f"[{model_name}]: unknown granularity: {granularity}")
