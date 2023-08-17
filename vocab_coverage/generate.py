# -*- coding: utf-8 -*-

import argparse
import io
import json
import logging
import os
import sys
import traceback
from typing import List
import yaml

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: disable=wrong-import-position
from vocab_coverage.coverage import coverage_analysis
from vocab_coverage.embedding import embedding_analysis
from vocab_coverage.lexicon import load_lexicon
from vocab_coverage.utils import logger, generate_coverage_filename, generate_embedding_filename, generate_thumbnail_filename, generate_model_path, release_resource
from vocab_coverage import constants

def find_coverage_file(model_name:str, granularity:str, postfix:str='', folder:str=constants.FOLDER_IMAGES):
    basename = generate_model_path(model_name)
    candidates = [
        generate_coverage_filename(model_name, granularity=granularity, postfix=postfix, folder=folder),
        f"{folder}/{basename}.png",
        f"{folder}/{basename}_coverage.png",
        f"{folder}/coverage.{basename}.png",
        f"{folder}/{basename}.coverage.png",
    ]
    for filename in candidates:
        # logger.debug(filename)
        if os.path.exists(filename):
            return filename
    return None

def find_embedding_file(model_name:str, granularity:str, position:str, postfix:str='', folder:str=constants.FOLDER_IMAGES):
    basename = generate_model_path(model_name)
    candidates = [
        generate_embedding_filename(model_name,
                                    granularity=granularity, position=position, postfix=postfix,
                                    folder=folder),
        f"{folder}/embeddings_{basename}.{position}.jpg",
        f"{folder}/embeddings.{basename}.{position}.jpg",
    ]
    if position == constants.EMBEDDING_POSITION_INPUT:
        candidates.append(f"{folder}/embeddings_{basename}.jpg")
        candidates.append(f"{folder}/embeddings.{basename}.jpg")
    for filename in candidates:
        # logger.debug(filename)
        if os.path.exists(filename):
            return filename
    return None

def find_thumbnail_file(filename:str, folder:str=constants.FOLDER_IMAGES):
    thumbnail = generate_thumbnail_filename(filename, folder=folder)
    if os.path.exists(thumbnail):
        return thumbnail
    return None

def generate_coverage(models:List[dict],
                      group:str='',
                      granularity:str=constants.GRANULARITY_CHARACTER,
                      folder=constants.FOLDER_IMAGES,
                      debug:bool=False):
    for section in models:
        if group not in ('', section['group']):
            continue
        for model_name in section["models"]:
            try:
                coverage = find_coverage_file(model_name, granularity=granularity, folder=folder)
                if coverage is not None:
                    # fix coverage filename
                    standard_coverage = generate_coverage_filename(model_name, granularity=granularity, folder=folder)
                    if coverage and standard_coverage != coverage:
                        logger.info("Renaming %s => %s", coverage, standard_coverage)
                        os.rename(coverage, standard_coverage)
                        coverage = standard_coverage
                    if debug:
                        logger.debug("[%s] No [%s] coverage is required to be generated.", model_name, granularity)
                    continue
                # generate coverage
                lexicon = load_lexicon(model_name, granularity=granularity, debug=debug)
                coverage_analysis(model_name,
                                  lexicon=lexicon,
                                  granularity=granularity,
                                  folder=folder,
                                  debug=debug)
                logger.info("[%s] Generated [%s] coverage.", model_name, granularity)
            except Exception as ex:
                logger.error("[%s] coverage_analysis() failed. [%s]", model_name, ex)
                traceback.print_exc()

def generate_embedding(models:List[dict],
                       group:str='',
                       granularity:str=constants.GRANULARITY_TOKEN,
                       positions:List[str]=None,
                       reducer:str=constants.REDUCER_TSNE,
                       folder=constants.FOLDER_IMAGES,
                       cleanup:bool=True,
                       debug:bool=False):
    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]

    for section in models:
        if group not in ('', section['group']):
            continue
        for model_name in section["models"]:
            try:
                # check embedding files
                position_candidates = []
                for position in positions:
                    embedding_file = find_embedding_file(model_name, granularity=granularity, position=position, folder=folder)
                    if embedding_file is None:
                        position_candidates.append(position)
                    standard_file = generate_embedding_filename(model_name,
                                                                granularity=granularity,
                                                                position=position,
                                                                folder=folder)
                    if embedding_file and standard_file != embedding_file:
                        logger.info("Renaming %s => %s", embedding_file, standard_file)
                        os.rename(embedding_file, standard_file)
                        embedding_file = standard_file
                
                if len(position_candidates) == 0:
                    logger.debug("[%s] No [%s] embedding at %s is required to be generated.", model_name, granularity, positions)
                    continue

                # release cache (Only for GPU)
                release_resource(model_name, clear_cache=False)

                # generate embedding
                lexicon = load_lexicon(model_name, granularity=granularity, debug=debug)
                embedding_analysis(model_name,
                                   lexicon=lexicon,
                                   granularity=granularity,
                                   positions=position_candidates,
                                   reducer=reducer,
                                   folder=folder,
                                   debug=debug)
                logger.info("[%s] Generated [%s] embedding at %s", model_name, granularity, position_candidates)
            except Exception as ex:
                logger.error("[%s] embedding_analysis() failed. [%s]", model_name, ex)
                traceback.print_exc()
            finally:
                # release cache (Both for GPU and Storage)
                release_resource(model_name, clear_cache=cleanup)


def generate_thumbnail(filename:str, folder=constants.FOLDER_IMAGES):
    if not(filename and os.path.exists(filename)):
        raise ValueError(f"Cannot find file {filename}")
    thumbnail_filename = generate_thumbnail_filename(filename, folder=folder)
    if os.path.exists(thumbnail_filename):
        logger.debug("Thumbnail for %s already exists.", filename)
        return
    logger.info("Creating thumbnail for %s", filename)
    ret = os.system(f"convert {filename} -quality 50 -resize 20% {thumbnail_filename}")
    if ret != 0:
        raise ValueError(f"Failed to create thumbnail for {filename}. ({ret})")

def generate_coverage_thumbnails(models:List[dict], 
                                 granularity:str=constants.GRANULARITY_CHARACTER,
                                 folder=constants.FOLDER_IMAGES):
    for section in models:
        for model_name in section["models"]:
            coverage = find_coverage_file(model_name,
                                          granularity=granularity,
                                          folder=folder)
            if coverage is not None:
                generate_thumbnail(coverage, folder=folder)

def generate_embedding_thumbnails(models:List[dict],
                                  granularity:str=constants.GRANULARITY_TOKEN,
                                  positions:List[str]=None,
                                  folder=constants.FOLDER_IMAGES):
    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    for section in models:
        for model_name in section["models"]:
            for position in positions:
                embedding = find_embedding_file(model_name,
                                                granularity=granularity,
                                                position=position,
                                                folder=folder)
                if embedding is not None:
                    generate_thumbnail(embedding, folder=folder)

def get_oss_url(image) -> str:
    # base_url = "https://lab99-syd-pub.oss-accelerate.aliyuncs.com/vocab-coverage/"
    # base_url = "https://lab99-syd-pub.oss-ap-southeast-2.aliyuncs.com/vocab-coverage/"
    base_url = "http://syd.jiefu.org/vocab-coverage/"
    if image.startswith("images/assets/"):
        image = image.replace("images/assets/", "")
    elif image.startswith("images/"):
        image = image.replace("images/", "")
    return base_url + image

def generate_markdown_for_model_graph(image_file) -> str:
    content = " "
    if image_file is None or len(image_file) == 0:
        content = " "
    else:
        thumbnail_file = find_thumbnail_file(image_file)
        if thumbnail_file is None:
            logger.debug("Cannot find thumbnail file for %s, use the full size instead.", image_file)
            thumbnail_file = image_file
        # use OSS for image
        image_file = get_oss_url(image_file)
        thumbnail_file = get_oss_url(thumbnail_file)
        content = f"[![]({thumbnail_file})]({image_file})"
    return content

def generate_markdown_for_model(model_name:str, folder:str=constants.FOLDER_IMAGES) -> str:
    granularities = [constants.GRANULARITY_CHARACTER, constants.GRANULARITY_TOKEN]
    coverages = {granularity: find_coverage_file(model_name, granularity=granularity, folder=folder) for granularity in granularities}
    positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    embeddings = {granularity:
                  {position:
                    find_embedding_file(model_name, granularity=granularity, position=position, folder=folder)
                    for position in positions}
                    for granularity in granularities}
    if all(coverage is None for coverage in coverages.values()) and all(embedding is None for embedding in embeddings.values()):
        logger.error("[%s] cannot find any coverage or embedding file.", model_name)
        return ""

    # construct the markdown
    if "/" in model_name:
        org, name = model_name.split("/")
        model_name = f'<p>{org}</p><p>/</p><p>{name}</p>'
    model_name = f'<b>{model_name}</b>'
    token_content = "| {model_name} | {coverage} | {input_embedding} | {output_embedding} |\n".format(
        model_name=model_name,
        coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_TOKEN]),
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_INPUT]),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_OUTPUT])
    )
    character_content = "| {model_name} | {coverage} | {input_embedding} | {output_embedding} |\n".format(
        model_name='', # 同一个模型只显示一次
        coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_CHARACTER]),
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_INPUT]),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_OUTPUT])
    )
    return token_content + character_content

def generate_markdown_for_models(models:List[str]) -> str:
    content = ""
    with io.StringIO() as f:
        f.write("| 名称| 完整覆盖率分析 | 输入向量分布 | 输出向量分布 |\n")
        f.write("| :---: | :---: | :---: | :---: |\n")
        # Table body
        for model_name in models:
            f.write(generate_markdown_for_model(model_name))
        content = f.getvalue()
    return content

def generate_markdown_for_all(models:List[dict], output:str="graphs.md", section_level:int=2):
    with open(output, "w", encoding='utf-8') as f:
        section_mark = "#" * section_level
        for section in models:
            f.write(f"{section_mark} {section['name']}\n\n")
            f.write(generate_markdown_for_models(section["models"]))
            f.write("\n\n")

def generate_markdown_from_template(template_file:str, model_list_file:str, output:str="README.md"):
    with open(template_file, "r", encoding='utf-8') as f:
        template = f.read()
    with open(model_list_file, "r", encoding='utf-8') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    for tag, section in models.items():
        content = generate_markdown_for_models(section["models"])
        if len(content) > 0:
            template = template.replace(f"{{{tag}}}", content)
    with open(output, "w", encoding='utf-8') as f:
        f.write(template)

def main():
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest='command')

    cmd_coverage = subcommands.add_parser('coverage', help='Generate coverage graphs')
    cmd_coverage.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmd_coverage.add_argument("--granularity", type=str, default="char", help="统计颗粒度，可选值为 token 或 char（默认为 char）")
    cmd_coverage.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmd_coverage.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_ASSETS_COVERAGE, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_ASSETS_COVERAGE}）")

    cmd_embedding = subcommands.add_parser('embedding', help='Generate embedding graphs')
    cmd_embedding.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmd_embedding.add_argument("--granularity", type=str, default="token", help="统计颗粒度，可选值为 token 或 char（默认为 token）")
    cmd_embedding.add_argument("--position", type=str, default="input,output", help="向量位置，可选值为 input, output 或 input,output（默认为 input,output）")
    cmd_embedding.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_ASSETS_EMBEDDING, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_ASSETS_EMBEDDING}）")
    cmd_embedding.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmd_embedding.add_argument("--no_cleanup", action="store_true", help="是否保留模型缓存文件")
    cmd_embedding.add_argument("--reducer", type=str, default="tsne", help="降维算法，可选值为 tsne 或 umap（默认为 tsne）")
    cmd_thumbnail = subcommands.add_parser('thumbnail', help='Generate thumbnails for embedding graphs')
    cmd_thumbnail.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_THUMBNAIL, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_THUMBNAIL}）")
    cmd_thumbnail.add_argument("--debug", action="store_true", help="是否输出调试信息")

    cmd_markdown = subcommands.add_parser('markdown', help='Generate markdown file for graphs')
    cmd_markdown.add_argument("--markdown", type=str, default="graphs.md")

    args = parser.parse_args()

    if hasattr(args, "debug") and args.debug:
        logger.setLevel(logging.DEBUG)

    models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yaml')
    with open(models_file, "r", encoding='utf-8') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    if args.command == "coverage":
        generate_coverage(models,
                          group=args.group,
                          granularity=args.granularity,
                          folder=args.folder,
                          debug=args.debug)
    elif args.command == "embedding":
        generate_embedding(models,
                           group=args.group,
                           granularity=args.granularity,
                           positions=args.position.split(','),
                           reducer=args.reducer,
                           folder=args.folder,
                           cleanup=not args.no_cleanup,
                           debug=args.debug)
    elif args.command == "thumbnails":
        generate_coverage_thumbnails(models, folder=args.folder)
        generate_embedding_thumbnails(models, folder=args.folder)
    elif args.command == "markdown":
        # markdown for all models
        logger.info("Generating markdown for all models to %s", args.markdown)
        generate_markdown_for_all(models, output=args.markdown)
        # markdown for readme
        template_file = os.path.join('docs', 'README.template.md')
        models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_readme.yaml')
        logger.info("Generating markdown for readme by %s and %s", template_file, models_file)
        generate_markdown_from_template(template_file=template_file, model_list_file=models_file, output='README.md')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
