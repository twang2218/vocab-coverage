# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import traceback
from typing import List

import numpy as np
import torch
import shutil

from transformers import AutoTokenizer, AutoModel
from vocab_coverage.draw import draw_vocab_embeddings

EMBEDDING_TYPE_INPUT = 'input'
EMBEDDING_TYPE_OUTPUT = 'output'

def load_tokenizer(model_name:str, debug:bool=False):
    try:
        kwargs = {}
        kwargs['trust_remote_code'] = True
        if 'llama' in model_name.lower() or 'vicuna' in model_name.lower():
            # https://github.com/LianjiaTech/BELLE/issues/242#issuecomment-1514330432
            # Avoid LlamaTokenizerFast conversion
            # lmsys/vicuna-7b-v1.3
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception as e:
        if "aquila" in e.args[0]:
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
            raise e

    # https://github.com/huggingface/transformers/issues/24514
    from transformers import LlamaTokenizerFast
    if isinstance(tokenizer, LlamaTokenizerFast):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    # https://github.com/huggingface/transformers/issues/22312
    if (not hasattr(tokenizer, 'pad_token')) or (tokenizer.pad_token is None) or (len(tokenizer.pad_token) == 0):
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None and len(tokenizer.eos_token) > 0:
            print(f"[{model_name}]: 'tokenizer.pad_token' is None, set tokenizer.pad_token = tokenizer.eos_token ({tokenizer.eos_token}))")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print(f"[{model_name}]: 'tokenizer.pad_token' and 'tokenizer.eos_token' are None, set tokenizer.pad_token = '</s>'")
            tokenizer.bos_token = '<s>'
            tokenizer.eos_token = '</s>'
            tokenizer.unk_token = '<unk>'
            tokenizer.pad_token = tokenizer.eos_token

    if debug:
        print(tokenizer)

    return tokenizer

def load_model(model_name:str, debug:bool=False):
    if "OpenAI" in model_name:
        print(f"[{model_name}]: OpenAI don't support model loading.")
        return None

    # 加载预训练模型

    # 判断是否是大模型
    is_large_model = False
    for large_model in ['6b', '7b', '12b', '13b', 'llama', 'gpt', 'aquila', 'moss']:
        # print(f"[{model_name}]: large_model: {large_model} in {model_name.lower()}? {large_model in model_name.lower()}")
        if large_model in model_name.lower():
            is_large_model = True
            break

    # 判断是否应以 4bit 模型加载
    should_use_4bit = False
    for large_model in ['oasst', 'int4']:
        if 'chatglm' in model_name.lower():
            # THUDM/chatglm-6b-int4
            # THUDM/chatglm2-6b-int4
            break
        if large_model in model_name.lower():
            should_use_4bit = True
            break
    
    try:
        kwargs = {}
        if is_large_model and not should_use_4bit:
            if "chatglm-6b" in model_name:
                # THUDM/chatglm-6b
                kwargs['torch_dtype'] = torch.half
                # kwargs['device_map'] = "auto"
            elif "chatglm2" in model_name:
                # THUDM/chatglm2-6b
                pass
            elif "falcon-7b" in model_name or "mpt-7b" in model_name:
                # tiiuae/falcon-7b-instruct
                # mosaicml/mpt-7b-instruct
                kwargs['torch_dtype'] = torch.bfloat16
                kwargs['device_map'] = "auto"
            else:
                kwargs['torch_dtype'] = torch.float16
                kwargs['device_map'] = "auto"

        if should_use_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['device_map'] = "auto"

        if debug:
            print(f"[{model_name}]: AutoModel.from_pretrained(model_name={model_name}, kwargs={kwargs})")

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    except Exception as e:
        if debug:
            print(f"[{model_name}]: AutoModel.from_pretrained(model_name={model_name}, kwargs={kwargs}) failed: {e}, args: {e.args}")
        if isinstance(e.args, (list, tuple)) and isinstance(e.args[0], str) and "AutoModel" in e.args[0]:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        elif isinstance(e.args, (list, tuple)) and isinstance(e.args[0], str) and "aquila" in e.args[0]:
            from flagai.model.aquila_model import AQUILAModel
            # cache_dir = os.path.join('./model', 'aquila-7b')
            # print(f"cache_dir: {os.path.abspath(cache_dir)}")
            model = AQUILAModel.from_pretrain(model_name='aquila-7b', download_path='./model')
        else:
            print("加载 AutoModel 模型 {} 失败：{}".format(model_name, e))
            raise e

    if debug:
        print(f"[{model_name}]: num_parameters: {model.num_parameters():,}")

    # ValueError: `.to` is not supported for `4-bit` or `8-bit` models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    #   fnlp/moss-moon-003-sft-int4
    #   OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
    if not should_use_4bit:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()

    print(f"[{model_name}]: {model.__class__.__name__} model loaded on device: {model.device}")

    if debug:
        show_gpu_usage(model_name)

    return model

def show_gpu_usage(name:str = ""):
    if torch.cuda.is_available():
        # 获取当前设备
        device = torch.cuda.current_device()
        # 获取显存信息
        memory_info = torch.cuda.memory_stats(device=device)
        # 获取显存使用量
        memory_used = memory_info["allocated_bytes.all.current"] / 1024 ** 2
        # 获取总共显存量
        memory_total = torch.cuda.get_device_properties(device=device).total_memory / 1024 ** 2
        # 获取剩余可用显存
        memory_free = memory_total - memory_used
        if len(name) > 0:
            name = f"[{name}]: "
        else:
            name = "> "
        print(f"{name} GPU Memory usage: {memory_used:,.0f} MiB")
        print(f"{name} GPU Memory free: {memory_free:,.0f} MiB")
        return {"total": memory_total, "used": memory_used, "free": memory_free}
    else:
        print("GPU not available.")
        return {"total": 0, "used": 0, "free": 0}

def get_vocab(model_name:str, tokenizer, debug=False):
    if "OpenAI" in model_name:
        model_name = model_name.split("/")[-1]
        return get_vocab_openai(model_name, debug=debug)

    vocab_size = max([id for id in tokenizer.get_vocab().values()]) + 1
    if debug:
        print(f"[{model_name}]: vocab_size: {vocab_size}")

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
            raise Exception(f"[{model_name}]: cannot find 'model.get_input_embeddings()'")

        if debug:
            print(f"[{model_name}]: get_input_embeddings(): {input_embedding_func}")

        vocab_size = len(vocab)
        if hasattr(input_embedding_func, 'weight'):
            # shibing624/prompt-t5-base-chinese: tokenizer.vocab_size=32228, get_input_embeddings().weight=(32128, 768)
            vocab_size = min(input_embedding_func.weight.shape[0], vocab_size)
        token_ids = torch.tensor(np.arange(0, vocab_size, 1)).to(model.device)
        input_embeddings = input_embedding_func(token_ids)
        if input_embeddings.is_cuda:
            input_embeddings = input_embeddings.cpu()
        input_embeddings = input_embeddings.detach().numpy()

        if debug:
            print(f"[{model_name}]: input_embeddings: {input_embeddings.shape}")
    except Exception as e:
        print(f"[{model_name}]: get_input_embeddings failed: {e}")
        traceback.print_exc()
        print(model)
        raise e
    return input_embeddings

def get_sentences_embeddings(model_name, model, tokenizer, sentences:List[str], use_token_id=False, max_length=256):
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
    if use_token_id:
        placeholder = ['a'] * len(sentences)
        inputs = tokenizer(placeholder, **kwargs)
        sentences_ids = tokenizer.convert_tokens_to_ids(sentences)
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
            print(f"[{model_name}]: get_sentences_embeddings() failed: {e}")
            traceback.print_exc()
            print(model)
            raise e

    # get attention_mask and token_embeddings
    # print(f"[{model_name}]: input_ids: {inputs['input_ids'].shape}, attention_mask: {inputs['attention_mask'].shape}")
    attention_mask = inputs['attention_mask']
    del inputs
    all_hidden_states = []
    for hs in outputs.hidden_states:
        all_hidden_states.append(hs.detach().clone())
    token_embeddings = outputs.hidden_states[-1].detach().clone()
    del outputs

    if 'chatglm-6b' in model_name:
        # THUDM/chatglm-6b
        #   attention_mask.shape: [50, 1, 4, 4] => [50, 4]
        old_shape = attention_mask.shape
        attention_mask = torch.where(attention_mask[:, 0, -1], torch.tensor(0), torch.tensor(1))
        print(f"[{model_name}]: fix attention_mask: {old_shape} => {attention_mask.shape}")
        #   token_embeddings.shape: [4, 50, 4096] => [50, 4, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        print(f"[{model_name}]: fix token_embeddings: {old_shape} => {token_embeddings.shape}")
    elif 'chatglm2-6b' in model_name:
        # THUDM/chatglm2-6b
        #   attention_mask.shape: [50, 7]
        #   token_embeddings.shape: [7, 50, 4096] => [50, 7, 4096]
        old_shape = token_embeddings.shape
        token_embeddings = token_embeddings.permute(1, 0, 2)
        print(f"[{model_name}]: fix token_embeddings: {old_shape} => {token_embeddings.shape}")

    # Calculate of Sentences Embedding by the averaging the all token vectors
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Convert to numpy
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    embeddings = embeddings.detach().numpy()

    for i, e in enumerate(embeddings):
        if np.isnan(e).any():
            print(f"[{model_name}]: embeddings for {i}:'{sentences[i]}' contains NaN")
            # print(f"[{model_name}]: > attention_mask({attention_mask[i].shape}):  {attention_mask[i]}")
            # print(f"[{model_name}]: > token_embeddings({token_embeddings[i].shape}): {token_embeddings[i]}")
            print(f"[{model_name}]: > all_hidden_states: {len(all_hidden_states)}")
            for j, hs in enumerate(all_hidden_states):
                if np.isnan(hs[i]).any():
                    print(f"[{model_name}]: > all_hidden_states[{j}]({hs[i].shape}): {hs[i]}")
            # print(f"[{model_name}]: > input_mask_expanded({input_mask_expanded[i].shape}): {input_mask_expanded[i]}")
    return embeddings

def get_sentences_embedding_in_batch(model_name, model, tokenizer, sentences:List[str], batch_size=32, use_token_id=False, max_length=256):
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_embeddings = get_sentences_embeddings(model_name, model, tokenizer, batch_sentences, use_token_id=use_token_id, max_length=max_length)
        if i == 0:
            embeddings = batch_embeddings
        else:
            embeddings = np.concatenate((embeddings, batch_embeddings))
        print(f"[{model_name}]: batch_embeddings: {batch_embeddings.shape}, embeddings: {embeddings.shape}")
    return embeddings

def get_output_embeddings(model_name, model, tokenizer, vocab, debug=False):
    output_embeddings = []
    try:
        if "OpenAI" in model_name:
            model_name = model_name.split("/")[-1]
            return get_output_embeddings_openai(model_name, vocab, batch=2000, debug=debug)

        memory = show_gpu_usage(model_name)
        if memory['total'] > 0:
            batch_size = round((memory['free']//12)/200) * 200
        else:
            batch_size = 100
        print(f"[{model_name}]: batch_size: {batch_size}")

        output_embeddings = get_sentences_embedding_in_batch(model_name, model, tokenizer, vocab, batch_size=batch_size, use_token_id=True, max_length=5)

    except Exception as e:
        print(f"[{model_name}]: get_output_embedding failed: {e}")
        traceback.print_exc()
        print(model)
        raise e
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
        # early_exaggeration=12,
        learning_rate='auto',
        metric='cosine',
        init='pca',
        verbose=2 if debug else 0,
        n_iter=1000,
        random_state=42,
        method='barnes_hut',
        n_jobs=-1)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    return embeddings_2d

def reduce_to_2d_tsne_cuml(embeddings, debug=False):
    from cuml.manifold import TSNE
    tsne_model = TSNE(n_components=2,
        # early_exaggeration=12,
        learning_rate_method='adaptive',
        metric='cosine',
        # init='pca',?
        verbose=debug,
        n_iter=1000,
        random_state=42,
        method='barnes_hut')
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
    filename = model_name.replace('/', '_') + f'.embeddings.{embedding_type}.jpg'
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    if debug:
        print(f"[{model_name}]: save {embedding_type}_embeddings to {filename}...")
    image.save(filename, quality=80, optimize=True, progressive=True)


def embedding_analysis(model_name:str, charsets:dict, output_dir:str, embedding_type=[EMBEDDING_TYPE_INPUT], is_detailed=False, debug=False, clear_cache=False):
    print("对模型 {} 的 embedding 进行可视化...".format(model_name))

    if '/' in model_name:
        org, name = model_name.split('/')
        if org.lower() == 'openai' and name != 'text-embedding-ada-002':
            print(f"Skip {model_name}, only 'text-embedding-ada-002' is supported...")
            return

    workdir = os.path.join(output_dir, 'embeddings')

    tokenizer = load_tokenizer(model_name, debug=debug)
    model = load_model(model_name, debug=debug)
    vocab = get_vocab(model_name, tokenizer=tokenizer, debug=debug)

    if hasattr(model, 'get_input_embeddings') and hasattr(model.get_input_embeddings(), 'weight'):
        tokenizer_vocab_size = len(vocab)
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size > model_vocab_size:
            print(f"[{model_name}]: tokenizer_vocab_size({tokenizer_vocab_size}) > model_vocab_size({model_vocab_size}), will truncate the model vocab_size...")
            vocab = vocab[:model_vocab_size]

    for etype in embedding_type:
        embeddings = get_embeddings(model_name, model, tokenizer, vocab, embedding_type=etype, debug=debug)
        not_non_embeddings = []
        not_non_vocab = []
        for i, e in enumerate(embeddings):
            if np.isnan(e).any():
                print(f"[{model_name}]: [{i}]: '{vocab[i]}' embeddings: ({np.shape(e)}): {e}")
            else:
                not_non_embeddings.append(e)
                not_non_vocab.append(vocab[i])
        embeddings = np.array(not_non_embeddings)
        vocab = not_non_vocab
        if embeddings is not None and len(embeddings) > 0:
            do_embedding_analysis(
                model_name=model_name,
                embeddings=embeddings,
                vocab=vocab,
                charsets=charsets,
                is_detailed=is_detailed,
                folder=workdir,
                embedding_type=etype,
                debug=debug)

    # clean up
    del tokenizer
    del model
    gc.collect()

    if torch.cuda.is_available():
        print(f"[{model_name}]: releasing GPU memory...")
        torch.cuda.empty_cache()
        show_gpu_usage(model_name)
    if clear_cache:
        model_path = f"models--{model_name.replace('/', '--')}"
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub", model_path)
        print(f"[{model_name}]: clean up cache ({cache_dir})...")
        shutil.rmtree(cache_dir, ignore_errors=True)

    return
