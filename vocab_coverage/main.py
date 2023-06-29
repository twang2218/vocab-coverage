import os
import sys
import argparse
import json
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocab_coverage import model_check, embedding_analysis

def main():
    parser = argparse.ArgumentParser()

    subcommands = parser.add_subparsers(dest='command')

    cmdModel = subcommands.add_parser('model', help='模型汉字识字率分析')
    cmdModel.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese", help="模型在 HuggingFace Hub 上的名称（默认为 shibing624/text2vec-base-chinese）")
    cmdModel.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdModel.add_argument("--output_dir", type=str, default="images", help="生成的图像文件的输出目录（默认为 images）")
    cmdModel.add_argument("--debug", action='store_true', help="是否打印调试信息")

    cmdEmbedding = subcommands.add_parser('embedding', help='词向量可视化分析')
    cmdEmbedding.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese", help="模型在 HuggingFace Hub 上的名称（默认为 shibing624/text2vec-base-chinese）")
    cmdEmbedding.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdEmbedding.add_argument("--output_dir", type=str, default="images", help="生成的图像文件的输出目录（默认为 images）")
    cmdEmbedding.add_argument("--is_detailed", action='store_true', help="是否对汉字进行详细分类（默认为 False）")
    cmdEmbedding.add_argument("--debug", action='store_true', help="是否打印调试信息（默认为 False）")
    cmdEmbedding.add_argument("--skip_input_embeddings", action='store_true', help="不计算输入层的词向量")
    cmdEmbedding.add_argument("--output_embeddings", action='store_true', help="计算输出层的词向量")

    cmdCharset = subcommands.add_parser('charset', help='生成用以统计识字率的字表文件')
    cmdCharset.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")

    args = parser.parse_args()

    if args.command == 'charset':
        from vocab_coverage import generate_charsets
        generate_charsets(args.charset_file)
        return
    elif args.command == 'model':
        charsets = json.load(open(args.charset_file, 'r'))
        model_check(args.model_name, charsets, args.output_dir, args.debug)
        return
    elif args.command == 'embedding':
        charsets = json.load(open(args.charset_file, 'r'))
        etypes = []
        if not args.skip_input_embeddings:
            etypes.append('input')
        if args.output_embeddings:
            etypes.append('output')
        embedding_analysis(
            model_name=args.model_name,
            charsets=charsets,
            output_dir=args.output_dir,
            embedding_type=etypes,
            is_detailed=args.is_detailed,
            debug=args.debug)
    else:
        parser.print_help()
        return

if __name__ == "__main__":
    load_dotenv()
    main()
