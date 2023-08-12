# -*- coding: utf-8 -*-

import os
import sys
import argparse
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from vocab_coverage import coverage_analysis, embedding_analysis
from vocab_coverage.crawler import get_chinese_charsets, get_token_charsets
from vocab_coverage.utils import logger
from vocab_coverage.classifier import Classifier
from vocab_coverage.lexicon import load_lexicon
from vocab_coverage import constants


def main():
    parser = argparse.ArgumentParser()

    subcommands = parser.add_subparsers(dest='command')

    cmd_coverage = subcommands.add_parser('coverage', help='模型汉字识字率分析')
    cmd_coverage.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese", help="模型在 HuggingFace Hub 上的名称（默认为 shibing624/text2vec-base-chinese）")
    cmd_coverage.add_argument("--granularity", type=str, default="char", help="字表类型，可选值为 token, char（默认为 char）")
    cmd_coverage.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_ASSETS_COVERAGE, help=f"生成的图像文件的输出目录（默认为 {constants.FOLDER_IMAGES_ASSETS_COVERAGE}）")
    cmd_coverage.add_argument("--debug", action='store_true', help="是否打印调试信息")

    cmd_embedding = subcommands.add_parser('embedding', help='词向量可视化分析')
    cmd_embedding.add_argument("--model_name", type=str, default="shibing624/text2vec-base-chinese", help="模型在 HuggingFace Hub 上的名称（默认为 shibing624/text2vec-base-chinese）")
    cmd_embedding.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_ASSETS_EMBEDDING, help=f"生成的图像文件的输出目录（默认为 {constants.FOLDER_IMAGES_ASSETS_EMBEDDING}）")
    cmd_embedding.add_argument("--postfix", type=str, default='', help="图像文件名可选后缀，用以控制生成的文件名")
    cmd_embedding.add_argument("--override", action='store_true', help="是否覆盖已存在的图像文件（默认为 False）")
    cmd_embedding.add_argument("--debug", action='store_true', help="是否打印调试信息（默认为 False）")
    cmd_embedding.add_argument("--position", type=str, default='input', help="词向量的位置。可选项为 'input', 'output'，多选用逗号分隔，如：'input,output',（默认为 'input'）")
    cmd_embedding.add_argument("--granularity", type=str, default='token', help="向量分析的颗粒度，可以为 'token'、'char'(汉字)、'word'(词)、'sentence'(句)，（默认为 'token'）")
    cmd_embedding.add_argument("--reducer_method", type=str, default="tsne", help="降维算法（默认为 tsne），可选值为 tsne, umap, tsne_cuml, umap_cuml, umap_tsne, umap_tsne_cuml")

    cmd_crawler = subcommands.add_parser('crawler', help='爬取用以统计识字率的字表文件')
    cmd_crawler.add_argument("--granularity", type=str, default="char", help="爬取的字表类型，可选值为 token, char（默认为 char）")
    cmd_crawler.add_argument("-f", "--file", type=str, default="", help="用以统计识字率的字表文件（默认为内置字符集文件）")
    # cmd_crawler.add_argument("--debug", action='store_true', help="是否打印调试信息（默认为 False）")
    cmd_crawler.add_argument("--indent", type=int, default=None, help="输出 JSON 文件时的缩进量（默认为 None）")
    args = parser.parse_args()

    # if hasattr(args, "debug") and args.debug:
    #     logger.setLevel(logging.DEBUG)

    if args.command == 'coverage':
        lexicon = load_lexicon(args.model_name, granularity=args.granularity, debug=args.debug)
        coverage_analysis(args.model_name,
                          lexicon=lexicon,
                          granularity=args.granularity,
                          folder=args.folder,
                          debug=args.debug)
    elif args.command == 'embedding':
        lexicon = load_lexicon(args.model_name, granularity=args.granularity, debug=args.debug)
        positions = args.position.split(',')
        embedding_analysis(
            model_name=args.model_name,
            lexicon=lexicon,
            folder=args.folder,
            postfix=args.postfix,
            override=args.override,
            positions=positions,
            granularity=args.granularity,
            reducer=args.reducer_method,
            debug=args.debug)
    elif args.command == 'crawler':
        # 爬取用以统计识字率的字表文件
        if args.granularity == constants.GRANULARITY_CHARACTER:
            charsets = get_chinese_charsets(debug=True)
        elif args.granularity == constants.GRANULARITY_TOKEN:
            charsets = get_token_charsets(debug=True)
        else:
            logger.error('不支持的字表类型：%s', args.granularity)
            sys.exit(1)
        # 处理缩进
        if isinstance(args.indent, int):
            if args.indent > 4:
                args.indent = 4
            elif args.indent < 0:
                args.indent = 0
        # 保存到文件
        Classifier(charsets, granularity=args.granularity).save(args.file, args.indent)
    else:
        parser.print_help()

if __name__ == "__main__":
    load_dotenv()
    main()
