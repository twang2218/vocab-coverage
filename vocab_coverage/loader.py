# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import os
from vocab_coverage.utils import show_gpu_usage

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
