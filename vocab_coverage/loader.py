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
    LlamaTokenizerFast,
    PreTrainedTokenizerBase,
)
import torch
import tiktoken
from tqdm import tqdm

from vocab_coverage.utils import logger, show_gpu_usage, is_match_patterns
from vocab_coverage import constants

def load_tokenizer_openai(model_name:str, debug:bool=False):
    name = model_name.split("/")[-1]
    tokenizer = tiktoken.encoding_for_model(name)
    tokenizer.vocab_size = tokenizer.n_vocab
    eos_token_id = tokenizer.encode_single_token(constants.TEXT_OPENAI_END_OF_TEXT)
    tokenizer.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else eos_token_id
    tokenizer.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else eos_token_id
    tokenizer.unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else eos_token_id
    tokenizer.pad_token = constants.TEXT_OPENAI_END_OF_TEXT
    if debug:
        # pylint: disable=protected-access
        logger.debug(tokenizer._special_tokens)
    return tokenizer

def load_tokenizer(model_name:str, debug:bool=False):
    try:
        model_name, revision = _split_revision(model_name)
        kwargs = {}
        kwargs['trust_remote_code'] = True
        if revision:
            kwargs['revision'] = revision
        if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_LLAMA):
            # https://github.com/LianjiaTech/BELLE/issues/242#issuecomment-1514330432
            # Avoid LlamaTokenizerFast conversion
            # lmsys/vicuna-7b-v1.3
            tokenizer = LlamaTokenizer.from_pretrained(model_name, **kwargs)
        elif is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
            tokenizer = load_tokenizer_openai(model_name, debug=debug)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    # pylint: disable=invalid-name,broad-exception-caught
    except Exception as e:
        logger.error("加载模型 %s 失败：%s", model_name, e)
        raise e

    # https://github.com/huggingface/transformers/issues/24514
    if isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # special cases
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_QWEN):
        eos_token_id = tokenizer.convert_tokens_to_ids(constants.TEXT_OPENAI_END_OF_TEXT)
        tokenizer.pad_token = constants.TEXT_OPENAI_END_OF_TEXT
        tokenizer.eos_token_id = eos_token_id
        tokenizer.pad_token_id = eos_token_id

    # https://github.com/huggingface/transformers/issues/22312
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            logger.warning("[%s]: 'tokenizer.pad_token' is None, set tokenizer.pad_token = tokenizer.eos_token (%s))",
                           model_name, tokenizer.eos_token)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("[%s]: both 'tokenizer.pad_token' and 'tokenizer.eos_token' are None, set tokenizer.pad_token = '</s>'", model_name)
            tokenizer.bos_token = '<s>'
            tokenizer.eos_token = '</s>'
            tokenizer.unk_token = '<unk>'
            tokenizer.pad_token = tokenizer.eos_token

    if debug:
        logger.debug(tokenizer)

    return tokenizer

def is_bbpe_tokenizer(model_name:str, tokenizer:PreTrainedTokenizerBase):
    logger.debug("[%s]: %s", model_name, tokenizer)
    return is_match_patterns(str(type(tokenizer)), constants.PATTERN_TOKENIZER_BBPE) or \
            is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_BBPE)

def _generate_model_kwargs(model_name:str):
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
        raise ValueError(f"不支持的模型：{model_name}")
    # 判断是否是需要用 int8 加载的模型
    int8_kwargs = {
        'load_in_8bit': True,
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_INT8):
        return int8_kwargs

    # 判断是否是需要 fp16 加载的模型
    fp16_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_FP16):
        return fp16_kwargs

    # 其他返回空
    return {}

def _get_model_class(model_name:str):
    # AutoModelForCausalLM
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_CAUSALLM):
        return AutoModelForCausalLM
    # default to AutoModel
    return AutoModel

def _split_revision(model_name:str):
    if '@' in model_name:
        return model_name.split('@')
    return model_name, None

def load_model(model_name:str, debug:bool=False):
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
        raise ValueError(f"不支持加载的模型：{model_name}")

    try:
        model_name, revision = _split_revision(model_name)
        kwargs = _generate_model_kwargs(model_name)
        if revision:
            kwargs['revision'] = revision
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
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            logger.error("加载 AutoModel 模型 %s (kwargs=%s) 失败：%s", model_name, kwargs, e)
            raise e

    if debug:
        num_parameters = f'{model.num_parameters():,}' if hasattr(model, 'num_parameters') else 'N/A'
        logger.debug("[%s]: num_parameters: %s", model_name, num_parameters)

    # ValueError: `.to` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    #   fnlp/moss-moon-003-sft-int4
    #   OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
    # if not should_use_4bit:
    #     if hasattr(torch, 'device'):
    if not getattr(model, 'is_quantized', False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()

    logger.info("[%s]: %s model loaded on device: %s",
                model_name, model.__class__.__name__, model.device)

    if debug:
        show_gpu_usage(model_name)

    return model

def load_vocab(model_name:str, tokenizer, debug=False) -> List[str]:
    if is_match_patterns(model_name, constants.PATTERN_MODEL_NAME_OPENAI):
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
                try:
                    vocab[v] = k.decode('utf-8')
                except UnicodeDecodeError:
                    hex_representation = k.hex()
                    vocab[v] = ''.join(['\\x' + hex_representation[i:i+2] for i in range(0, len(hex_representation), 2)])
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
    t = tiktoken.encoding_for_model(model_name)
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
    # debug with small amount of vocab
    # vocab = vocab[:4000]
    return vocab

def load_wordbook(model_name:str, granularity:str=constants.GRANULARITY_TOKEN, debug:bool=False) -> List[str]:
    if granularity == constants.GRANULARITY_TOKEN:
        tokenizer = load_tokenizer(model_name, debug=debug)
        vocab = load_vocab(model_name, tokenizer, debug=debug)
        wordbook = [{'id': i, 'text': v} for i, v in enumerate(vocab)]
        return wordbook
    elif granularity in [constants.GRANULARITY_CHARACTER,
                         constants.GRANULARITY_WORD,
                         constants.GRANULARITY_SENTENCE,
                         constants.GRANULARITY_PARAGRAPH]:
        # 返回 None，表示将使用 classifier 的内容做语料
        return None
    else:
        raise ValueError(f"[{model_name}]: unknown granularity: {granularity}")
