import json
import os
from transformers import AutoTokenizer
from grid_graph import draw_vocab_graph
import argparse

# 加载字表
charset = json.load(open('charset.json', 'r'))

def zh_vocab_check(model_name:str, debug=False):
    print("检查模型 {} 的字表".format(model_name))
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
            tokenizer.unk_token_id = 0
            tokenizer.vocab_size = tokenizer.num_tokens
        elif "OpenAI" in e.args[0]:
            import tiktoken
            name = model_name.split("/")[-1]
            tokenizer = tiktoken.encoding_for_model(name)
            tokenizer.vocab_size = tokenizer.n_vocab
            if debug:
                print(tokenizer._special_tokens)
        else:
            print("加载模型 {} 失败：{}".format(model_name, e))
            return

    if debug:
        print(tokenizer)
        print(tokenizer.vocab_size)

    tokenizers_with_warp_token = [
        "BertTokenizer",
        "BertTokenizerFast",
        "RobertaTokenizer",
        "RobertaTokenizerFast",
        "ElectraTokenizer",
        "ElectraTokenizerFast",
        "T5Tokenizer",
        "T5TokenizerFast",
        "MPNetTokenizer",
        "MPNetTokenizerFast",
        "DistilBertTokenizer",
        "DistilBertTokenizerFast",
        "XLMRobertaTokenizer",
        "XLMRobertaTokenizerFast",
        "XLNetTokenizer",
        "XLNetTokenizerFast",
        "AlbertTokenizer",
        "AlbertTokenizerFast",
    ]

    charset_stats = {
        name: {
            'known': 0,
            'total': len(chars),
            'chars': chars,
            'map': [0 for _ in range(len(chars))]
        } for name, chars in charset.items()
    }

    if debug:
        if hasattr(tokenizer, 'cls_token_id'):
            print('[Special Token ID] => cls: {}, sep: {}, pad: {}, unk: {}, mask: {}'.format(
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
                tokenizer.mask_token_id
            ))

    for name, chars in charset.items():
        for i, c in enumerate(chars):
            # 编码
            tokens_ids = tokenizer.encode(c)

            # 编码预处理
            tn = type(tokenizer).__name__
            if tn in tokenizers_with_warp_token:
                # 对有头尾token的编码，去掉头尾token
                tokens_ids = tokens_ids[1:-1]
                if len(tokens_ids) > 0 and tokens_ids[0] in [6, 13]:
                    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
                    # 模型的tokenizer会在开始前缀一个'6'，后面的id已经足够可以解码为给定汉字了，因此去除前缀
                    # albert-base-v2 会在开始前缀一个'13'
                    tokens_ids = tokens_ids[1:]
                if len(tokens_ids) > 0 and tokens_ids[-1] == tokenizer.sep_token_id:
                    # 对有sep_token的编码，去掉尾部的sep_token
                    tokens_ids = tokens_ids[:-1]
            elif tn == "ChatGLMTokenizer":
                tokens_ids = tokens_ids[:-2]
                if len(tokens_ids) > 0 and tokens_ids[0] == 5:
                    # 有时候会在开始前缀一个'5'
                    tokens_ids = tokens_ids[1:]
            elif tn == "LlamaTokenizer" or tn == "LlamaTokenizerFast":
                # TODO: 不使用 hardcode 的数值
                # 汉字(一)被拆分了，编码为[1, 29871, 30287]
                # 汉字(溻)被拆分了，编码为[0, 29871, 233, 189, 190]
                # 汉字(一)被拆分了，编码为[1, 31822, 231, 187, 131]
                if tokens_ids[0] in [0,1] and tokens_ids[1] in [29871, 31822]:
                    tokens_ids = tokens_ids[2:]

            # 识字程度判断
            if len(tokens_ids) < 1 or (len(tokens_ids) == 1 and hasattr(tokenizer, 'unk_token_id') and tokens_ids[0] == tokenizer.unk_token_id):
                # 未识别的字
                charset_stats[name]['map'][i] = 0
            elif len(tokens_ids) == 1:
                # 完全识别的字
                charset_stats[name]['map'][i] = 1
                charset_stats[name]['known'] += 1
            else: # len(tokens_ids) > 1
                # 一定程度上识别的字，并不计数，只计算识别程度
                charset_stats[name]['map'][i] = 1.0/len(tokens_ids) # 识别程度
                if debug:
                    print("[{}] 汉字({})被拆分了，编码为{}".format(tn, c, tokens_ids))

    # 统计显示
    for name, stats in charset_stats.items():
        print("字表{}：{}/{} ({:.2%})".format(name, stats['known'], stats['total'], float(stats['known'])/stats['total']))

    # 生成字表图
    filename = model_name.replace("/", "_") + ".png"
    filename = os.path.join("images", filename)
    draw_vocab_graph(model_name, charset_stats, tokenizer.vocab_size, filename, width=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    zh_vocab_check(args.model_name, args.debug)
