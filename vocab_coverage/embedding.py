# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from vocab_coverage.draw import draw_vocab_embeddings

EMBEDDING_TYPE_INPUT = 'input'
EMBEDDING_TYPE_OUTPUT = 'output'

def load_tokenizer(model_name:str, debug:bool=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        if "LLaMATokenizer" in e.args[0]:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
        elif "aquila" in e.args[0]:
            from flagai.data.tokenizer import Tokenizer
            name = 'aquila-7b'
            cache_dir = os.path.join('./model', name)
            tokenizer = Tokenizer.from_pretrained(name, cache_dir=cache_dir)
            tokenizer.cls_token_id = tokenizer.token_start_id
            tokenizer.sep_token_id = tokenizer.token_end_id
            tokenizer.unk_token_id = tokenizer.token_unk_id if hasattr(tokenizer, 'token_unk_id') else None
            tokenizer.pad_token_id = tokenizer.token_pad_id if hasattr(tokenizer, 'token_pad_id') else None
            tokenizer.mask_token_id = tokenizer.token_mask_id if hasattr(tokenizer, 'token_mask_id') else None
            tokenizer.vocab_size = tokenizer.num_tokens
        elif "OpenAI" in e.args[0]:
            import tiktoken
            name = model_name.split("/")[-1]
            tokenizer = tiktoken.encoding_for_model(name)
            tokenizer.vocab_size = tokenizer.n_vocab
            tokenizer.cls_token_id = tokenizer.encode_single_token('<|endoftext|>')
            if debug:
                print(tokenizer._special_tokens)
        else:
            print("加载模型 {} 失败：{}".format(model_name, e))
            exit(1)
    return tokenizer

def load_model(model_name:str, debug:bool=False):
    if "OpenAI" in model_name:
        print(f"[{model_name}]: OpenAI don't support model loading.")
        return None

    # 加载预训练模型
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        if isinstance(e.args, (list, tuple)) and "AutoModel" in e.args[0]:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        elif isinstance(e.args, (list, tuple)) and "aquila" in e.args[0]:
            from flagai.model.aquila_model import AQUILAModel
            # cache_dir = os.path.join('./model', 'aquila-7b')
            # print(f"cache_dir: {os.path.abspath(cache_dir)}")
            model = AQUILAModel.from_pretrain(model_name='aquila-7b', download_path='./model')
        else:
            print("加载 AutoModel 模型 {} 失败：{}".format(model_name, e))
            exit(1)
    model.eval()
    return model

def get_vocab(model_name:str, debug=False):
    if "OpenAI" in model_name:
        model_name = model_name.split("/")[-1]
        return get_vocab_openai(model_name, debug=debug)

    tokenizer = load_tokenizer(model_name)

    vocab_size = max([id for id in tokenizer.get_vocab().values()]) + 1
    if debug:
        print(f"[{model_name}] vocab_size: {vocab_size}")

    # get vocab
    vocab = [''] * (vocab_size)
    for k, v in tokenizer.get_vocab().items():
        if v >= vocab_size:
            print(f"[{model_name}] out of range: {k}, {v}")
            continue
        try:
            if hasattr(tokenizer, 'convert_tokens_to_string'):
                vocab[v] = tokenizer.convert_tokens_to_string([k])
            elif hasattr(tokenizer, 'text_tokenizer') and hasattr(tokenizer.text_tokenizer, 'convert_tokens_to_string'):
                # BAAI/aquila-7b
                vocab[v] = tokenizer.text_tokenizer.convert_tokens_to_string([k])
            else:
                vocab[v] = k
        except Exception as e:
            print(f"[{model_name}]: convert_tokens_to_string({k}) failed: {e}")
            vocab[v] = k
    return vocab

def get_vocab_openai(model_name:str, debug=False):
    import tiktoken
    t = tiktoken.encoding_for_model('gpt-3.5-turbo')
    count_except = 0
    vocab = []
    for k in t._mergeable_ranks.keys():
        try:
            vocab.append(str(k, encoding='utf-8'))
        except:
            # print(str(k))
            count_except += 1
            vocab.append(str(k))
    if debug:
        print(f"[{model_name}]: vocab: {len(vocab)}")
        print(f"[{model_name}]: count_except: {count_except}")
    return vocab

def get_input_embeddings(model_name, model, tokenizer, vocab, debug=False):
    input_embeddings = []
    try:
        if "OpenAI" in model_name:
            print(f"[{model_name}]: Cannot retrieve input embeddings from OpenAI models.")
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
            print(f"[{model_name}]: cannot find 'model.get_input_embeddings()'")
            print(model)
            exit(1)
        if debug:
            print(f"[{model_name}]: get_input_embeddings(): {input_embedding_func}")
        token_ids = torch.tensor(np.arange(0, len(vocab), 1)).to(model.device)
        input_embeddings = input_embedding_func(token_ids)
        if debug:
            print(f"[{model_name}]: input_embeddings: {input_embeddings.shape}")
    except Exception as e:
        print(f"[{model_name}]: get_input_embeddings failed: {e}")
        print(model)
        exit(1)
    return input_embeddings

def get_output_embeddings(model_name, model, tokenizer, vocab, debug=False):
    output_embeddings = []
    try:
        if "OpenAI" in model_name:
            model_name = model_name.split("/")[-1]
            return get_output_embeddings_openai(model_name, vocab, batch=2000, debug=debug)

        if hasattr(model, 'get_output_embeddings'):
            print(f"[{model_name}]: get_output_embeddings(): {model.get_output_embeddings()}")
            # THUDM/chatglm-6b
            vocab_embedding_func = model.get_output_embeddings()
            if vocab_embedding_func is None:
                print(f"[{model_name}]: 'model.get_output_embeddings()' is None")
            else:
                token_ids = torch.tensor(np.arange(0, len(vocab), 1)).to(model.device)
                output_embeddings = vocab_embedding_func(token_ids)

        if len(output_embeddings) == 0:
            # BERT-like models
            from text2vec import SentenceModel
            sm = SentenceModel('bert-base-chinese') # TODO: this model will be replaced by real model
            sm.model_name_or_path = model_name
            sm.tokenizer = tokenizer
            sm.bert = model
            sm.bert.to(sm.device)
            if debug:
                print(f"[{model_name}]: get_output_embeddings(): {sm.encode}")
            output_embeddings = sm.encode(vocab, batch_size=1000)
            if debug:
                print(f"[{model_name}]: output_embeddings: {np.shape(output_embeddings)}")
    except Exception as e:
        print(f"[{model_name}]: get_output_embedding failed: {e}")
        print(model)
        exit(1)
    return output_embeddings

def get_output_embeddings_openai(model_name:str, vocab:List[str], batch=10, debug=False):
    import openai
    embeds = []
    for i in range(0, len(vocab), batch):
        if debug:
            print(f"[{model_name}]: get_output_embeddings_openai(): {i}")
        ee = openai.Embedding.create(input = vocab[i:i+batch], model=model_name)['data']
        ee = [e['embedding'] for e in ee]
        if debug:
            print(f"[{model_name}]: Retrieved {len(ee)} embeddings for {vocab[i:i+batch]}")
        embeds.extend(ee)

    if debug:
        print(f"embeds: {len(embeds)}")
    return np.array(embeds)

def get_embeddings(model_name:str, model, tokenizer, vocab, embedding_type=EMBEDDING_TYPE_INPUT, debug=False):
    if embedding_type == EMBEDDING_TYPE_INPUT:
        return get_input_embeddings(model_name, model, tokenizer, vocab, debug=debug)
    elif embedding_type == EMBEDDING_TYPE_OUTPUT:
        return get_output_embeddings(model_name, model, tokenizer, vocab, debug=debug)
    else:
        print(f"[{model_name}]: unknown embedding_type: {embedding_type}")
        return None

def reduce_to_2d_tsne(embeddings, debug=False):
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2,
        early_exaggeration=12,
        metric='cosine',
        init='pca',
        verbose=2 if debug else 0,
        n_iter=1000,
        random_state=42,
        n_jobs=-1)
    embeddings_2d = tsne_model.fit_transform(embeddings)

    return embeddings_2d

def reduce_to_2d_umap(embeddings, debug=False):
    import warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    import umap

    umap_model = umap.UMAP(n_components=2,
        n_neighbors=40,
        min_dist=7,
        spread=7,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_2d = umap_model.fit_transform(embeddings)
    return embeddings_2d

def do_embedding_analysis(model_name:str, embeddings, vocab, charsets:dict, is_detailed=False, folder=None, embedding_type=EMBEDDING_TYPE_INPUT, debug=False):
    if debug:
        print(f"[{model_name}]: reducing the dimension of '{embedding_type}_embeddings' {embeddings.shape} to 2D...")
    embeddings_2d = reduce_to_2d_tsne(embeddings, debug=debug)
    if debug:
        print(f"[{model_name}]: draw {embedding_type}_embeddings {embeddings_2d.shape}...")
    image = draw_vocab_embeddings(
        model_name=model_name,
        embeddings_2d=embeddings_2d,
        vocab=vocab,
        charsets=charsets,
        embedding_type=embedding_type,
        width=8000,
        height=8000,
        is_detailed=is_detailed,
        debug=debug)

    # 生成文件名
    filename = model_name.replace('/', '_') + f'.{embedding_type}.jpg'
    filename = 'embeddings.' + filename
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    if debug:
        print(f"[{model_name}]: save {embedding_type}_embeddings to {filename}...")
    image.save(filename, quality=80, optimize=True, progressive=True)


def embedding_analysis(model_name:str, charsets:dict, output_dir:str, embedding_type=[EMBEDDING_TYPE_INPUT], is_detailed=False, debug=False):
    print("对模型 {} 的 embedding 进行可视化...".format(model_name))

    workdir = os.path.join(output_dir, 'embeddings')

    for etype in embedding_type:
        tokenizer = load_tokenizer(model_name, debug=debug)
        model = load_model(model_name, debug=debug)
        vocab = get_vocab(model_name, debug=debug)
        embeddings = get_embeddings(model_name, model, tokenizer, vocab, embedding_type=etype, debug=debug)
        if embeddings is not None and len(embeddings) > 0:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().numpy()
            # vocab = vocab[:1000]
            # embeddings = embeddings[:1000]
            do_embedding_analysis(
                model_name=model_name,
                embeddings=embeddings,
                vocab=vocab,
                charsets=charsets,
                is_detailed=is_detailed,
                folder=workdir,
                embedding_type=etype,
                debug=debug)

    return
