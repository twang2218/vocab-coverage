import os
import sys
import argparse
import json

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocab_coverage import model_check

def main():
    parser = argparse.ArgumentParser()

    subcommands = parser.add_subparsers(dest='command')

    cmdModel = subcommands.add_parser('model', help='模型汉字识字率分析')
    cmdModel.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese", help="模型在 HuggingFace Hub 上的名称（默认为 shibing624/text2vec-base-chinese）")
    cmdModel.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdModel.add_argument("--output_dir", type=str, default="images", help="生成的图像文件的输出目录（默认为 images）")
    cmdModel.add_argument("--debug", action='store_true', help="是否打印调试信息")

    cmdCharset = subcommands.add_parser('charset', help='生成用以统计识字率的字表文件')
    cmdCharset.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")

    args = parser.parse_args()

    if args.command == 'charset':
        from vocab_coverage import generate_charsets
        generate_charsets(args.charset_file)
        return
    elif args.command == 'model':
        charsets = json.load(open(args.charset_file, 'r'))
        model_check(charsets, args.model_name, args.output_dir, args.debug)
        return
    else:
        parser.print_help()
        return

if __name__ == "__main__":
    main()
