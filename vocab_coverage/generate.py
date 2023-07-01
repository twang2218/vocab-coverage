# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import traceback
from typing import List

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocab_coverage import coverage_analysis, embedding_analysis

DEFAULT_IMAGE_FOLDER = "images"

def load_model_list(filename:str="models.json") -> List[dict]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, filename)
    try:
        with open(filename, "r") as f:
            models = json.load(f)
        return models
    except Exception as e:
        print(f"Cannot load model list from {filename}")
        traceback.print_exc()
        exit(1)

def find_coverage_file(model_name:str, folder:str=DEFAULT_IMAGE_FOLDER):
    folder = os.path.join(folder, "coverage")
    basename = model_name.replace("/", "_")
    candidates = [
        f"{folder}/{basename}.png",
        f"{folder}/{basename}_coverage.png",
        f"{folder}/coverage.{basename}.png",
        f"{folder}/{basename}.coverage.png",
    ]
    for filename in candidates:
        # print(filename)
        if os.path.exists(filename):
            return filename
    return None

def find_embedding_file(model_name:str, embedding_type:str, folder:str=DEFAULT_IMAGE_FOLDER):
    folder = os.path.join(folder, "embeddings")
    basename = model_name.replace("/", "_")
    candidates = [
        f"{folder}/embeddings_{basename}.{embedding_type}.jpg",
        f"{folder}/embeddings.{basename}.{embedding_type}.jpg",
    ]
    if embedding_type == "input":
        candidates.append(f"{folder}/embeddings_{basename}.jpg")
        candidates.append(f"{folder}/embeddings.{basename}.jpg")
    for filename in candidates:
        # print(filename)
        if os.path.exists(filename):
            return filename
    return None

def get_thumbnail_filename(filename:str):
    basedir = os.path.dirname(filename)
    basename = os.path.basename(filename)
    thumbnail = os.path.join(basedir, "thumbnails", basename)
    return thumbnail

def find_thumbnail_file(filename:str):
    thumbnail = get_thumbnail_filename(filename)
    if os.path.exists(thumbnail):
        return thumbnail
    return None

def generate_markdown(models:List[dict], output:str="graphs.md"):
    with open(output, "w") as f:
        for section in models:
            f.write(f"## {section['name']}\n\n")
            # Table header
            f.write("| 名称| ![](images/empty.png) 中文覆盖率 | ![](images/empty.png) 输入词向量分布 | ![](images/empty.png) 输出词向量分布 |\n")
            f.write("| :---: | :---: | :---: | :---: |\n")
            # Table body
            for model_name in section["models"]:
                basename = model_name.replace("/", "_")
                coverage = find_coverage_file(model_name)
                if coverage is None:
                    print(f"Cannot find coverage file for {model_name}")
                input_embedding = find_embedding_file(model_name, "input")
                if input_embedding is None and not 'openai' in model_name.lower():
                    print(f"Cannot find input embedding file for {model_name}")
                output_embedding = find_embedding_file(model_name, "output")
                if output_embedding is None and not 'openai' in model_name.lower():
                    print(f"Cannot find output embedding file for {model_name}")
                if coverage is None and input_embedding is None and output_embedding is None:
                    print(f"Cannot find any file for {model_name}")
                    continue
                else:
                    # Name
                    # model_name = f'<b style="display: inline-block; transform: rotate(-90deg);">{model_name}</b>'
                    if "/" in model_name:
                        org, name = model_name.split("/")
                        model_name = f'<p>{org}</p><p>/</p><p>{name}</p>'
                    model_name = f'<b>{model_name}</b>'
                    # Coverage
                    if coverage is None or len(coverage) == 0:
                        coverage = " "
                    else:
                        coverage = f"![Vocab Coverage for {model_name}]({coverage})"
                    # Input Embedding
                    if input_embedding is None or len(input_embedding) == 0:
                        input_embedding = " "
                    else:
                        input_embedding_thumbnail = find_thumbnail_file(input_embedding)
                        if input_embedding_thumbnail is None:
                            print(f"Cannot find thumbnail file for {input_embedding}, use the full image instead.")
                            input_embedding_thumbnail = input_embedding
                        input_embedding = f"[![input embedding image for {model_name}]({input_embedding_thumbnail})]({input_embedding})"
                    # Output Embedding
                    if output_embedding is None or len(output_embedding) == 0:
                        output_embedding = " "
                    else:
                        output_embedding_thumbnail = find_thumbnail_file(output_embedding)
                        if output_embedding_thumbnail is None:
                            print(f"Cannot find thumbnail file for {output_embedding}, use the full image instead.")
                            output_embedding_thumbnail = output_embedding
                        output_embedding = f"[![output embedding image for {model_name}]({output_embedding_thumbnail})]({output_embedding})"
                    f.write(f"| {model_name} | {coverage} | {input_embedding} | {output_embedding} |\n")
                # print(f"* {model_name}")
                # print("\n")
            f.write("\n\n")

def generate_coverage(models:List[dict], charsets:dict, group:str='', folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        if group != '' and section['group'] != group:
            if debug:
                print(f"Skip group {section['group']}")
            continue
        for model_name in section["models"]:
            try:
                coverage = find_coverage_file(model_name)
                if coverage is not None:
                    if debug:
                        print(f"Nothing to generate for {model_name} coverage. ({coverage}).")
                    continue
                # generate coverage
                coverage_analysis(model_name, charsets, folder, debug)
                print(f"Generated coverage for {model_name}")
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()
                continue

def generate_embedding(models:List[dict], charsets:dict, group:str='', folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        if group != '' and section['group'] != group:
            if debug:
                print(f"Skip group {section['group']}")
            continue
        for model_name in section["models"]:
            try:
                embedding_types = []
                input_embedding = find_embedding_file(model_name, "input")
                if input_embedding is None and (not 'openai' in model_name.lower()):
                    embedding_types.append("input")
                output_embedding = find_embedding_file(model_name, "output")
                if output_embedding is None:
                    embedding_types.append("output")
                if len(embedding_types) == 0:
                    if debug:
                        print(f"Nothing to generate for {model_name} embedding. ({input_embedding}, {output_embedding}))")
                    continue
                if "openai" not in model_name.lower() or "/text-embedding-ada-002" not in model_name.lower():
                    if debug:
                        print(f"Do not support embedding analysis for {model_name}")
                    continue
                if 'openai' in model_name.lower():
                    print(f'embedding_types: {embedding_types}')
                # generate embedding
                embedding_analysis(model_name=model_name,
                                charsets=charsets,
                                output_dir=folder,
                                embedding_type=embedding_types,
                                debug=debug)
                print(f"Generated embedding for {model_name}")
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()


def generate_embedding_thumbnails(models:List[dict], folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        for model_name in section["models"]:
            try:
                input_embedding = find_embedding_file(model_name, "input", folder=folder)
                if debug:
                    print(f"model_name: {model_name}, input_embedding: {input_embedding}")
                if input_embedding is not None:
                    input_embedding_thumbnail = get_thumbnail_filename(input_embedding)
                    if not os.path.exists(input_embedding_thumbnail):
                        print(f"Creating thumbnail for {input_embedding}")
                        os.system(f"convert {input_embedding} -quality 20 -resize 30% {input_embedding_thumbnail}")
                output_embedding = find_embedding_file(model_name, "output", folder=folder)
                if debug:
                    print(f"model_name: {model_name}, output_embedding: {output_embedding}")
                if output_embedding is not None:
                    output_embedding_thumbnail = get_thumbnail_filename(output_embedding)
                    if not os.path.exists(output_embedding_thumbnail):
                        print(f"Creating thumbnail for {output_embedding}")
                        os.system(f"convert {output_embedding} -quality 20 -resize 30% {output_embedding_thumbnail}")
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()
                continue

def main():
    models = load_model_list()

    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest='command')

    cmdMarkdown = subcommands.add_parser('markdown', help='Generate markdown file for graphs')
    cmdMarkdown.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdMarkdown.add_argument("--markdown", type=str, default="graphs.md")

    cmdCoverage = subcommands.add_parser('coverage', help='Generate coverage graphs')
    cmdCoverage.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmdCoverage.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdCoverage.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmdCoverage.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")

    cmdEmbedding = subcommands.add_parser('embedding', help='Generate embedding graphs')
    cmdEmbedding.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmdEmbedding.add_argument("--charset_file", type=str, default="charset.json", help="用以统计识字率的字表文件（默认为 charset.json）")
    cmdEmbedding.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")
    cmdEmbedding.add_argument("--debug", action="store_true", help="是否输出调试信息")

    cmdThumbnails = subcommands.add_parser('thumbnails', help='Generate thumbnails for embedding graphs')
    cmdThumbnails.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")
    cmdThumbnails.add_argument("--debug", action="store_true", help="是否输出调试信息")

    args = parser.parse_args()

    if args.command == "coverage":
        charsets = json.load(open(args.charset_file, 'r'))
        generate_coverage(models, charsets, group=args.group, folder=args.folder, debug=args.debug)
    elif args.command == "embedding":
        charsets = json.load(open(args.charset_file, 'r'))
        generate_embedding(models, charsets, group=args.group, folder=args.folder, debug=args.debug)
    elif args.command == "thumbnails":
        generate_embedding_thumbnails(models, folder=args.folder, debug=args.debug)
    elif args.command == "markdown":
        generate_markdown(models, output=args.markdown)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()