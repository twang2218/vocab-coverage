# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from vocab_coverage.draw import draw_vocab_embeddings

def load_tokenizer(model_name):
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

def load_model(model_name):
    # 加载预训练模型
    tokenizer = load_tokenizer(model_name)
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
            print("加载模型 {} 失败：{}".format(model_name, e))
            exit(1)
    model.eval()
    return model, tokenizer

def get_vocab_embeddings(model_name:str, debug=False):
    if "OpenAI" in model_name:
        model_name = model_name.split("/")[-1]
        vocab = get_openai_vocab(model_name, debug=debug)
        vocab_embeddings = get_openai_vocab_embeddings(vocab, model_name, batch=2000, debug=debug)
        return vocab, vocab_embeddings

    model, tokenizer = load_model(model_name)
    # get embeddings
    vocab_index = np.arange(0, model.config.vocab_size, 1)
    if hasattr(model, 'get_input_embeddings'):
        vocab_embedding_func = model.get_input_embeddings()
    elif hasattr(model, 'tok_embeddings'):
        # BAAI/aquila-7b
        vocab_embedding_func = model.tok_embeddings
    vocab_embeddings = vocab_embedding_func(torch.tensor(vocab_index)).detach().numpy()


    if debug:
        if hasattr(tokenizer, 'convert_tokens_to_string'):
            print(f"tokenizer {tokenizer.__class__.__name__} has convert_tokens_to_string()")
        if hasattr(tokenizer, 'text_tokenizer') and hasattr(tokenizer.text_tokenizer, 'convert_tokens_to_string'):
            print(f"tokenizer {tokenizer.__class__.__name__} has text_tokenizer.convert_tokens_to_string()")

    # get vocab
    vocab = [''] * (model.config.vocab_size)
    for k, v in tokenizer.get_vocab().items():
        if v >= model.config.vocab_size:
            print(f"out of range: {k}, {v}")
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
            print(f"convert_tokens_to_string({k}) failed: {e}")
            vocab[v] = k

    return vocab, vocab_embeddings

def get_openai_vocab(model_name:str, debug=False):
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
        print(f"vocab: {len(vocab)}")
        print(f"count_except: {count_except}")
    return vocab

def get_openai_vocab_embeddings(vocab:list, model='text-embedding-ada-002', batch=10, debug=False):
    import openai
    embeds = []
    for i in range(0, len(vocab), batch):
        if debug:
            print(f"get_openai_vocab_embeddings(): {i}")
        ee = openai.Embedding.create(input = vocab[i:i+batch], model=model)['data']
        ee = [e['embedding'] for e in ee]
        if debug:
            print(f"Retrieved {len(ee)} embeddings for {vocab[i:i+batch]}")
        embeds.extend(ee)

    if debug:
        print(f"embeds: {len(embeds)}")
    return np.array(embeds)

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

def embedding_analysis(model_name:str, charsets:dict, output_dir:str, is_detail=False, debug=False):
    print("对模型 {} 的 embedding 进行可视化...".format(model_name))

    model = {
        'model_name': model_name,
    }
    vocab, embeddings = get_vocab_embeddings(model_name, debug)
    model['vocab'] = vocab
    model['embeddings'] = embeddings
    if debug:
        print(f"reducing embeddings {embeddings.shape} to 2D...")
    model['embeddings_2d'] = reduce_to_2d_tsne(embeddings, debug=debug)
    if debug:
        print(f"draw embeddings {model['embeddings_2d'].shape}...")
    image = draw_vocab_embeddings(model, charsets, width=8000, height=8000, is_detail=is_detail, debug=debug)

    # 生成文件名
    filename = model_name.replace('/', '_') + '.jpg'
    filename = 'embeddings_' + filename
    output_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)

    model['filename'] = filename

    # save to file
    if debug:
        print(f"save to {filename}...")
    image.save(filename, quality=80, optimize=True, progressive=True)

    return model
