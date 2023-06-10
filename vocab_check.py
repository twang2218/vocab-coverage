import json
import os
from transformers import AutoTokenizer
from grid_graph import draw_vocab_graph, vocab_index_to_level
import argparse
from transformers import LlamaTokenizer as LLaMATokenizer

# 加载字表
charset_table = json.load(open('charset.json', 'r'))
# 由于汉典的汉字太多了，8万汉字，我们暂时不考虑汉典其它汉字，将来如果找到一个2万汉字的字表，再考虑
charset_table = charset_table[:3]
all_chars = []
for c in charset_table:
    all_chars.extend(c)
# print("共有{}个汉字".format(len(all_chars)))
charset_map = [1 for _ in range(len(all_chars))]


def zh_vocab_check(model_name:str, debug=False):
    print("检查模型 {} 的字表".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        if "LLaMATokenizer" in e.args[0]:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            print("加载模型 {} 失败：{}".format(model_name, e))
            return

    if debug:
        print(tokenizer)

    tokenizers_with_warp_token = [
        "BertTokenizer",
        "BertTokenizerFast",
        "RobertaTokenizer",
        "RobertaTokenizerFast",
        "ElectraTokenizer",
        "ElectraTokenizerFast",
        "T5Tokenizer",
        "T5TokenizerFast",
    ]
    count_by_level = [0, 0, 0, 0]
    for i, c in enumerate(all_chars):
        tokens_ids = tokenizer.encode(c)
        tn = type(tokenizer).__name__
        if tn in tokenizers_with_warp_token:
            tokens_ids = tokens_ids[1:-1]
        elif tn == "ChatGLMTokenizer":
            tokens_ids = tokens_ids[:-2]
            if tokens_ids[0] == 5:
                tokens_ids = tokens_ids[1:]
        elif tn == "LlamaTokenizer" or tn == "LlamaTokenizerFast":
            # 汉字(一)被拆分了，编码为[1, 29871, 30287]
            # 汉字(溻)被拆分了，编码为[0, 29871, 233, 189, 190]
            if tokens_ids[0] in [0,1] and tokens_ids[1] == 29871:
                tokens_ids = tokens_ids[2:]
        if len(tokens_ids) < 1 or (len(tokens_ids) == 1 and tokens_ids[0] == tokenizer.unk_token_id):
            charset_map[i] = 0
        else:
            level = vocab_index_to_level(i)
            charset_map[i] = 1.0/len(tokens_ids)
            if len(tokens_ids) > 1:
                # 说明可能是BBPE编码导致汉字被拆分了，这里我们不认为覆盖了该汉字
                count_by_level[level] += 0
                if debug:
                    print("汉字({})被拆分了，编码为{}".format(c, tokens_ids))
            else:
                count_by_level[level] += 1


    for i in range(len(charset_table)):
        print("第{}级字表：{}/{} ({:.2%})".format(i+1, count_by_level[i+1], len(charset_table[i]), float(count_by_level[i+1])/len(charset_table[i])))
    print("总计：{}/{} ({:.2%})".format(sum(count_by_level), len(all_chars), float(sum(count_by_level))/len(all_chars)))
    filename = model_name.replace("/", "_") + ".png"
    filename = os.path.join("images", filename)
    draw_vocab_graph(model_name, charset_map, filename, width=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    zh_vocab_check(args.model_name, args.debug)