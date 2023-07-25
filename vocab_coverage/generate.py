# -*- coding: utf-8 -*-

import argparse
import io
import json
import os
import sys
import traceback
from typing import List
import yaml

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocab_coverage import coverage_analysis, embedding_analysis

DEFAULT_IMAGE_FOLDER = "images"

def get_model_basename(model_name:str):
    basename = model_name.replace("/", "_")
    return basename

def get_standard_coverage_filename(model_name:str, folder:str=DEFAULT_IMAGE_FOLDER):
    folder = os.path.join(folder, "assets", "coverage")
    basename = get_model_basename(model_name)
    filename = f"{folder}/{basename}.coverage.jpg"
    return filename

def get_standard_embedding_filename(model_name:str, embedding_type:str, folder:str=DEFAULT_IMAGE_FOLDER):
    folder = os.path.join(folder, "assets", "embeddings")
    basename = get_model_basename(model_name)
    filename = f"{folder}/{basename}.embeddings.{embedding_type}.jpg"
    return filename

def find_coverage_file(model_name:str, folder:str=DEFAULT_IMAGE_FOLDER):
    folder = os.path.join(folder, "assets", "coverage")
    basename = get_model_basename(model_name)
    candidates = [
        get_standard_coverage_filename(model_name),
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
    folder = os.path.join(folder, "assets", "embeddings")
    basename = get_model_basename(model_name)
    candidates = [
        get_standard_embedding_filename(model_name, embedding_type),
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

def get_thumbnail_filename(filename:str, folder:str=DEFAULT_IMAGE_FOLDER):
    basename = os.path.basename(filename)
    basename = basename.replace(".jpg", ".thumbnail.jpg")
    thumbnail = os.path.join(folder, "thumbnails", basename)
    return thumbnail

def find_thumbnail_file(filename:str):
    thumbnail = get_thumbnail_filename(filename)
    if os.path.exists(thumbnail):
        return thumbnail
    return None

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
                    # fix coverage filename
                    standard_coverage = get_standard_coverage_filename(model_name)
                    if standard_coverage != coverage:
                        print(f"Renaming {coverage} => {standard_coverage}")
                        os.rename(coverage, standard_coverage)
                        coverage = standard_coverage
                    if debug:
                        print(f"Nothing to generate for {model_name} coverage. ({coverage}).")
                    continue
                # generate coverage
                coverage_analysis(model_name, charsets, folder, debug)
                print(f"Generated coverage for {model_name}")
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()

def generate_embedding(models:List[dict], charsets:dict, group:str='', folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        if group != '' and section['group'] != group:
            if debug:
                print(f"Skip group {section['group']}")
            continue
        for model_name in section["models"]:
            try:
                embedding_types = []
                # Input Embedding
                input_embedding = find_embedding_file(model_name, "input")
                if input_embedding is None and (not 'openai' in model_name.lower()):
                    embedding_types.append("input")
                # fix the embedding filename
                standard_input_embedding = get_standard_embedding_filename(model_name, "input")
                if input_embedding is not None and input_embedding != standard_input_embedding:
                    print(f"Renaming {input_embedding} => {standard_input_embedding}")
                    os.rename(input_embedding, standard_input_embedding)
                    input_embedding = standard_input_embedding
                # Output Embedding
                output_embedding = find_embedding_file(model_name, "output")
                if output_embedding is None:
                    embedding_types.append("output")
                # fix the embedding filename
                standard_output_embedding = get_standard_embedding_filename(model_name, "output")
                if output_embedding is not None and output_embedding != standard_output_embedding:
                    print(f"Renaming {output_embedding} => {standard_output_embedding}")
                    os.rename(output_embedding, standard_output_embedding)
                    output_embedding = standard_output_embedding
                # check if we need to generate embedding
                if len(embedding_types) == 0:
                    if debug:
                        print(f"Nothing to generate for {model_name} embedding. ({input_embedding}, {output_embedding}))")
                    continue
                # check special case for OpenAI
                if "openai" in model_name.lower() and "text-embedding-ada-002" not in model_name.lower():
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
                                clear_cache=True,
                                debug=debug)
                print(f"Generated embedding for {model_name}")
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()

def generate_thumbnail(filename:str, folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    if filename is None or len(filename) == 0 or not os.path.exists(filename):
        print(f"Cannot find file {filename}")
        return
    thumbnail_filename = get_thumbnail_filename(filename, folder=folder)
    if not os.path.exists(thumbnail_filename):
        basedir = os.path.dirname(thumbnail_filename)
        if not os.path.exists(basedir):
            os.makedirs(basedir, exist_ok=True)
        print(f"Creating thumbnail for {filename}")
        ret = os.system(f"convert {filename} -quality 50 -resize 20% {thumbnail_filename}")
        if ret != 0:
            print(f"Failed to create thumbnail for {filename}")
            raise Exception(f"Failed to create thumbnail for {filename}")

def generate_embedding_thumbnails(models:List[dict], folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        for model_name in section["models"]:
            try:
                input_embedding = find_embedding_file(model_name, "input", folder=folder)
                if input_embedding is not None:
                    generate_thumbnail(input_embedding, folder=folder, debug=debug)
                output_embedding = find_embedding_file(model_name, "output", folder=folder)
                if output_embedding is not None:
                    generate_thumbnail(output_embedding, folder=folder, debug=debug)
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()

def generate_coverage_thumbnails(models:List[dict], folder=DEFAULT_IMAGE_FOLDER, debug:bool=False):
    for section in models:
        for model_name in section["models"]:
            try:
                coverage = find_coverage_file(model_name, folder=folder)
                if coverage is not None:
                    generate_thumbnail(coverage, folder=folder, debug=debug)
            except Exception as e:
                print(f"Error in {model_name}")
                traceback.print_exc()

def get_oss_url(image) -> str:
    # base_url = "https://lab99-syd-pub.oss-accelerate.aliyuncs.com/vocab-coverage/"
    base_url = "https://lab99-syd-pub.oss-ap-southeast-2.aliyuncs.com/vocab-coverage/"
    if image.startswith("images/assets/"):
        image = image.replace("images/assets/", "")
    return base_url + image

def generate_markdown_for_model(model_name:str) -> str:
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
        return ""
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
            coverage_thumbnail = find_thumbnail_file(coverage)
            if coverage_thumbnail is None:
                print(f"Cannot find thumbnail file for {coverage}, use the full image instead.")
                coverage_thumbnail = coverage
            coverage = f"[![Vocab Coverage for {model_name}]({coverage_thumbnail})]({get_oss_url(coverage)})"
        # Input Embedding
        if input_embedding is None or len(input_embedding) == 0:
            input_embedding = " "
        else:
            input_embedding_thumbnail = find_thumbnail_file(input_embedding)
            if input_embedding_thumbnail is None:
                print(f"Cannot find thumbnail file for {input_embedding}, use the full image instead.")
                input_embedding_thumbnail = input_embedding
            input_embedding = f"[![input embedding image for {model_name}]({input_embedding_thumbnail})]({get_oss_url(input_embedding)})"
        # Output Embedding
        if output_embedding is None or len(output_embedding) == 0:
            output_embedding = " "
        else:
            output_embedding_thumbnail = find_thumbnail_file(output_embedding)
            if output_embedding_thumbnail is None:
                print(f"Cannot find thumbnail file for {output_embedding}, use the full image instead.")
                output_embedding_thumbnail = output_embedding
            output_embedding = f"[![output embedding image for {model_name}]({output_embedding_thumbnail})]({get_oss_url(output_embedding)})"
        return f"| {model_name} | {coverage} | {input_embedding} | {output_embedding} |\n"

def generate_markdown_for_models(models:List[str]) -> str:
    content = ""
    with io.StringIO() as f:
        f.write("| 名称| 汉字覆盖率分析 | 输入词向量分布 | 输出词向量分布 |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        # Table body
        for model_name in models:
            f.write(generate_markdown_for_model(model_name))
        content = f.getvalue()
    return content

def generate_markdown_for_all(models:List[dict], output:str="graphs.md", section_level:int=2):
    with open(output, "w") as f:
        section_mark = "#" * section_level
        for section in models:
            f.write(f"{section_mark} {section['name']}\n\n")
            f.write(generate_markdown_for_models(section["models"]))
            f.write("\n\n")

def generate_markdown_from_template(template_file:str, model_list_file:str, output:str="README.md"):
    with open(template_file, "r") as f:
        template = f.read()
    with open(model_list_file, "r") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    for tag, section in models.items():
        content = generate_markdown_for_models(section["models"])
        if len(content) > 0:
            template = template.replace(f"{{{tag}}}", content)
    with open(output, "w") as f:
        f.write(template)

def main():
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest='command')

    cmdCoverage = subcommands.add_parser('coverage', help='Generate coverage graphs')
    cmdCoverage.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmdCoverage.add_argument("--charset_file", type=str, default="", help="用以统计识字率的字表文件（默认为使用内置字符集文件）")
    cmdCoverage.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmdCoverage.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")

    cmdEmbedding = subcommands.add_parser('embedding', help='Generate embedding graphs')
    cmdEmbedding.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmdEmbedding.add_argument("--charset_file", type=str, default="", help="用以统计识字率的字表文件（默认为使用内置字符集文件）")
    cmdEmbedding.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")
    cmdEmbedding.add_argument("--debug", action="store_true", help="是否输出调试信息")

    cmdThumbnails = subcommands.add_parser('thumbnails', help='Generate thumbnails for embedding graphs')
    cmdThumbnails.add_argument("--folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="输出文件夹（默认为 images）")
    cmdThumbnails.add_argument("--debug", action="store_true", help="是否输出调试信息")

    cmdMarkdown = subcommands.add_parser('markdown', help='Generate markdown file for graphs')
    cmdMarkdown.add_argument("--charset_file", type=str, default="", help="用以统计识字率的字表文件（默认为使用内置字符集文件）")
    cmdMarkdown.add_argument("--markdown", type=str, default="graphs.md")

    args = parser.parse_args()

    if hasattr(args, "charset_file") and len(args.charset_file) == 0:
        # 使用内置字符集文件
        args.charset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charsets.json')

    models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yaml')
    with open(models_file, "r") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    if args.command == "coverage":
        charsets = json.load(open(args.charset_file, 'r'))
        generate_coverage(models, charsets, group=args.group, folder=args.folder, debug=args.debug)
    elif args.command == "embedding":
        charsets = json.load(open(args.charset_file, 'r'))
        generate_embedding(models, charsets, group=args.group, folder=args.folder, debug=args.debug)
    elif args.command == "thumbnails":
        generate_coverage_thumbnails(models, folder=args.folder, debug=args.debug)
        generate_embedding_thumbnails(models, folder=args.folder, debug=args.debug)
    elif args.command == "markdown":
        # markdown for all models
        print(f"Generating markdown for all models to {args.markdown}")
        generate_markdown_for_all(models, output=args.markdown)
        # markdown for readme
        template_file = os.path.join('docs', 'README.template.md')
        models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_readme.yaml')
        print(f"Generating markdown for readme by {template_file} and {models_file}")
        generate_markdown_from_template(template_file=template_file, model_list_file=models_file, output='README.md')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()