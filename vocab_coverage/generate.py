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

def find_coverage_file(model_name:str, granularity:str, postfix:str='', folder:str=constants.FOLDER_IMAGES, debug:bool=False):
    basename = generate_model_path(model_name)
    candidates = [
        generate_coverage_filename(model_name, granularity=granularity, postfix=postfix, folder=folder),
        f"{folder}/{basename}.png",
        f"{folder}/{basename}_coverage.png",
        f"{folder}/coverage.{basename}.png",
        f"{folder}/{basename}.coverage.png",
    ]
    # if debug:
    #     logger.debug('> candidates: %s', candidates)
    for filename in candidates:
        # logger.debug(filename)
        if os.path.exists(filename):
            return filename
    return None

def find_embedding_file(model_name:str, granularity:str, position:str, postfix:str='', folder:str=constants.FOLDER_IMAGES, debug:bool=False):
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
    # if debug:
    #     logger.debug('> candidates: %s', candidates)
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
            # pylint: disable=broad-except
            except Exception as ex:
                logger.error("[%s] coverage_analysis() failed. [%s]", model_name, ex)
                traceback.print_exc()

def generate_embedding(models:List[dict],
                       group:str='',
                       granularities:List[str]=None,
                       positions:List[str]=None,
                       reducer:str=constants.REDUCER_TSNE,
                       folder=constants.FOLDER_IMAGES,
                       cleanup:bool=True,
                       debug:bool=False):
    if granularities is None:
        granularities = [constants.GRANULARITY_TOKEN]

    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]

    for section in models:
        if group not in ('', section['group']):
            continue
        for model_name in section["models"]:
            try:
                # check embedding files
                position_candidates = {}
                for granularity in granularities:
                    position_candidates[granularity] = []
                    for position in positions:
                        embedding_file = find_embedding_file(model_name, granularity=granularity, position=position, folder=folder, debug=debug)
                        if embedding_file is None:
                            position_candidates[granularity].append(position)
                        standard_file = generate_embedding_filename(model_name,
                                                                    granularity=granularity,
                                                                    position=position,
                                                                    folder=folder)
                        if embedding_file and standard_file != embedding_file:
                            logger.info("Renaming %s => %s", embedding_file, standard_file)
                            os.rename(embedding_file, standard_file)
                            embedding_file = standard_file
                
                if sum(len(position_candidates[granularity]) for granularity in granularities) == 0:
                    logger.debug("[%s] Skip. Exists all [%s] embedding for [%s]", model_name, positions, granularities)
                    continue

                # generate embedding
                for granularity in granularities:
                    # release cache (Only for GPU)
                    release_resource(model_name, clear_cache=False)
                    # analysis
                    lexicon = load_lexicon(model_name, granularity=granularity, debug=debug)
                    embedding_analysis(model_name,
                                    lexicon=lexicon,
                                    granularity=granularity,
                                    positions=position_candidates[granularity],
                                    reducer=reducer,
                                    folder=folder,
                                    debug=debug)
                    logger.info("[%s] Generated [%s] embedding at %s", model_name, granularity, position_candidates[granularity])
            # pylint: disable=broad-except
            except Exception as ex:
                logger.error("[%s] embedding_analysis() failed. [%s]", model_name, ex)
                traceback.print_exc()
            finally:
                # release cache (Both for GPU and Storage)
                release_resource(model_name, clear_cache=cleanup)


def generate_thumbnail(filename:str, folder=constants.FOLDER_IMAGES, debug:bool=False):
    if not(filename and os.path.exists(filename)):
        raise ValueError(f"Cannot find file {filename}")
    thumbnail_filename = generate_thumbnail_filename(filename, folder=folder)
    if os.path.exists(thumbnail_filename):
        logger.debug("Thumbnail for %s already exists.", filename)
        return
    logger.info("Creating thumbnail for %s", filename)
    ret = os.system(f"convert {filename} -quality 50 -resize 10% {thumbnail_filename}")
    if ret != 0:
        raise ValueError(f"Failed to create thumbnail for {filename}. ({ret})")

def generate_coverage_thumbnails(models:List[dict], 
                                 granularities:List[str]=None,
                                 input:str=constants.FOLDER_IMAGES_FULLSIZE,
                                 output:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                 debug:bool=False):
    if granularities is None:
        granularities = [constants.GRANULARITY_TOKEN, constants.GRANULARITY_CHARACTER]
    for section in models:
        for model_name in section["models"]:
            logger.debug('为模型 [%s] 生成完整覆盖率缩略图...', model_name)
            for granularity in granularities:
                coverage = find_coverage_file(model_name,
                                            granularity=granularity,
                                            folder=input)
                if coverage is not None:
                    generate_thumbnail(coverage, folder=output)
                else:
                    if debug:
                        logger.debug('> 未发现 [%s] 覆盖率图。', granularity)

def generate_embedding_thumbnails(models:List[dict],
                                  granularities:List[str]=None,
                                  positions:List[str]=None,
                                  input:str=constants.FOLDER_IMAGES_FULLSIZE,
                                  output:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                  debug:bool=False):
    if granularities is None:
        granularities = [constants.GRANULARITY_TOKEN, constants.GRANULARITY_CHARACTER, constants.GRANULARITY_WORD]
    if positions is None:
        positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    for section in models:
        for model_name in section["models"]:
            logger.debug('为模型 [%s] 生成向量分布图缩略图...', model_name)
            for granularity in granularities:
                for position in positions:
                    embedding = find_embedding_file(model_name,
                                                    granularity=granularity,
                                                    position=position,
                                                    folder=input,
                                                    debug=debug)
                    if embedding is not None:
                        generate_thumbnail(embedding, folder=output)
                    else:
                        if debug:
                            logger.debug('> 未发现 [%s] [%s] 向量分布图。', granularity, position)

def get_oss_url(image) -> str:
    ## OSS加速器地址，浏览器访问该地址会提示下载，而不是直接看图
    # base_url = "https://lab99-syd-pub.oss-accelerate.aliyuncs.com/vocab-coverage/"
    ## OSS直接链接地址，访问正常，但是国内访问会慢
    # base_url = "https://lab99-syd-pub.oss-ap-southeast-2.aliyuncs.com/vocab-coverage/"
    ## 备案后的域名指向OSS加速器地址，这种情况会直接看图，而不是下载，国内访问速度也还可以
    base_url = "http://syd.jiefu.org/vocab-coverage/"
    ## 本地测试使用
    # base_url = "images/"
    if image.startswith("images/assets/"):
        image = image.replace("images/assets/", "")
    elif image.startswith("images/"):
        image = image.replace("images/", "")
    return base_url + image

def generate_markdown_for_model_graph(image_file, thumbnail:str=constants.FOLDER_IMAGES_THUMBNAIL) -> str:
    content = " "
    if image_file is None or len(image_file) == 0:
        content = " "
    else:
        thumbnail_file = find_thumbnail_file(image_file, thumbnail)
        if thumbnail_file is None:
            logger.debug("Cannot find thumbnail file for %s, use the full size instead.", image_file)
            thumbnail_file = image_file
        # use OSS for image
        image_file = get_oss_url(image_file)
        thumbnail_file = get_oss_url(thumbnail_file)
        content = f"[![]({thumbnail_file})]({image_file})"
    return content

def generate_markdown_for_model(model_name:str,
                                folder:str=constants.FOLDER_IMAGES,
                                fullsize:str=constants.FOLDER_IMAGES_FULLSIZE,
                                thumbnail:str=constants.FOLDER_THUMBNAIL) -> str:
    granularities = [constants.GRANULARITY_TOKEN, constants.GRANULARITY_CHARACTER, constants.GRANULARITY_WORD]
    coverages = {granularity: find_coverage_file(model_name, granularity=granularity, folder=fullsize) for granularity in granularities}
    positions = [constants.EMBEDDING_POSITION_INPUT, constants.EMBEDDING_POSITION_OUTPUT]
    embeddings = {granularity:
                  {position:
                    find_embedding_file(model_name, granularity=granularity, position=position, folder=fullsize)
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
    # title = f"| {model_name} | | |\n"
    # title = f"#### {model_name}\n"

    # header =  "| 颗粒度 | 完整覆盖率分析 | 输入向量分布 | 输出向量分布 |\n"
    # header += "| :---: | :---: | :---: | :---: |\n"
    # token_content = "| **{granularity}** | {coverage} | {input_embedding} | {output_embedding} |\n".format(
    #     granularity='Token',
    #     coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_TOKEN], thumbnail=thumbnail),
    #     input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
    #     output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    # )
    # character_content = "| **{granularity}** | {coverage} | {input_embedding} | {output_embedding} |\n".format(
    #     granularity='汉字',
    #     coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_CHARACTER], thumbnail=thumbnail),
    #     input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
    #     output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    # )
    # word_content = "| **{granularity}** | {coverage} | {input_embedding} | {output_embedding} |\n".format(
    #     granularity='汉字词汇',
    #     coverage='', #generate_markdown_for_model_graph(coverages[constants.GRANULARITY_WORD], thumbnail=thumbnail),
    #     input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
    #     output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    # )
    # header = "| Token 完整覆盖率 | Token 输入向量分布 | Token 输出向量分布 | 汉字完整覆盖率 | 汉字输入向量分布 | 汉字输出向量分布 | 词语输入向量分布 | 词语输出向量分布 |\n"
    # header += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    header = f"| {model_name} "
    token_content = "| {coverage} | {input_embedding} | {output_embedding} |".format(
        coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_TOKEN], thumbnail=thumbnail),
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_TOKEN][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )
    character_content = " {coverage} | {input_embedding} | {output_embedding} |".format(
        coverage=generate_markdown_for_model_graph(coverages[constants.GRANULARITY_CHARACTER], thumbnail=thumbnail),
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_CHARACTER][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )
    word_content = " {input_embedding} | {output_embedding} |\n".format(
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )

    return header + token_content + character_content + word_content

def generate_markdown_for_models(models:List[str],
                                 fullsize:str=constants.FOLDER_IMAGES_FULLSIZE,
                                 thumbnail:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                 level:int=3) -> str:
    content = ""
    with io.StringIO() as f:
        # f.write("| 颗粒度 | 完整覆盖率分析 | 输入向量分布图 | 输出向量分布图 |\n")
        # f.write("| :---: | :---: | :---: | :---: |\n")
        # header = "| 模型名称 | Token<br/>完整性图 | Token<br/>输入分布 | Token<br/>输出分布 | 汉字<br/>完整性图 | 汉字<br/>输入分布 | 汉字<br/>输出分布 | 词语<br/>输入分布 | 词语<br/>输出分布 |\n"
        # header = "| 模型名称 | ![](images/static/empty.2000.png) 完整覆盖率分析<br/>(Token) | ![](images/static/empty.2000.png) 输入向量分布图<br/>(Token) | ![](images/static/empty.2000.png) 输出向量分布图<br/>(Token) | ![](images/static/empty.2000.png) 完整覆盖率分析<br/>(汉字) | ![](images/static/empty.2000.png) 输入向量分布图<br/>(汉字) | ![](images/static/empty.2000.png) 输出向量分布图<br/>(汉字) | ![](images/static/empty.2000.png) 输入向量分布图<br/>(词语) | ![](images/static/empty.2000.png) 输出向量分布图<br/>(词语) |\n"
        # header =  "|     |     |     |     |     |     |     |     |     |\n"
        header =  "| 模型 | 完整性分析 (子字) | 入向量分布 (子字) | 出向量分布 (子字) | 完整性分析 (汉字) | 入向量分布 (汉字) | 出向量分布 (汉字) | 入向量分布 (词语) | 出向量分布 (词语) |\n"
        header += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        # header += "| **模型** |**(Token)**|**(Token)**|**(Token)**|**(汉字)**|**(汉字)**|**(汉字)**|**(词语)**|**(词语)**|\n"
        f.write(header)
        # Table body
        for model_name in models:
            # model_mark = "#" * level
            # f.write(f"{model_mark} {model_name}\n\n")
            f.write(generate_markdown_for_model(model_name, fullsize=fullsize, thumbnail=thumbnail))
        content = f.getvalue()
    return content

def markdown_text2anchor(text):
    anchor = ""
    for c in text:
        if c == ' ':
            anchor += '-'
        elif c in '()':
            continue
        else:
            anchor += c.lower()
    return anchor

def generate_markdown_for_all(models:List[dict],
                              fullsize:str=constants.FOLDER_IMAGES_FULLSIZE,
                              thumbnail:str=constants.FOLDER_IMAGES_THUMBNAIL,
                              output:str="graphs.md",
                              section_level:int=2):
    with open(output, "w", encoding='utf-8') as f:
        # 输出目录
        f.write("# 所有模型的分析图\n\n")
        f.write("## 目录\n\n")
        for section in models:
            section_name = section['name']
            f.write(f"- [{section_name}](#{markdown_text2anchor(section_name)})\n")
        f.write("\n\n")
        # 输出模型
        section_mark = "#" * section_level
        for section in models:
            f.write(f"{section_mark} {section['name']}\n\n")
            f.write(generate_markdown_for_models(section["models"], fullsize=fullsize, thumbnail=thumbnail, level=section_level+1))
            f.write("\n\n")

def generate_markdown_from_template(template_file:str, model_list_file:str,
                                    fullsize:str=constants.FOLDER_IMAGES_FULLSIZE,
                                    thumbnail:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                    output:str="README.md"):
    with open(template_file, "r", encoding='utf-8') as f:
        template = f.read()
    with open(model_list_file, "r", encoding='utf-8') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    for tag, section in models.items():
        content = generate_markdown_for_models(section["models"], fullsize=fullsize, thumbnail=thumbnail)
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
    cmd_coverage.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_FULLSIZE}）")

    cmd_embedding = subcommands.add_parser('embedding', help='Generate embedding graphs')
    cmd_embedding.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmd_embedding.add_argument("--granularity", type=str, default="token", help="统计颗粒度，可选值为 (token、char、word）或组合（如：token,char），（默认为 token）")
    cmd_embedding.add_argument("--position", type=str, default="input,output", help="向量位置，可选值为 input, output 或 input,output（默认为 input,output）")
    cmd_embedding.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_FULLSIZE}）")
    cmd_embedding.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmd_embedding.add_argument("--no_cleanup", action="store_true", help="是否保留模型缓存文件")
    cmd_embedding.add_argument("--reducer", type=str, default="tsne", help="降维算法，可选值为 tsne 或 umap（默认为 tsne）")

    cmd_thumbnail = subcommands.add_parser('thumbnail', help='Generate thumbnails for embedding graphs')
    cmd_thumbnail.add_argument("--input", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"输入文件夹，用以查找全尺寸的图片文件，默认为 {constants.FOLDER_IMAGES_FULLSIZE}")
    cmd_thumbnail.add_argument("--output", type=str, default=constants.FOLDER_IMAGES_THUMBNAIL, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_THUMBNAIL}）")
    cmd_thumbnail.add_argument("--debug", action="store_true", help="是否输出调试信息")

    cmd_markdown = subcommands.add_parser('markdown', help='Generate markdown file for graphs')
    cmd_markdown.add_argument("--fullsize", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"全尺寸图片目录，默认为 {constants.FOLDER_IMAGES_FULLSIZE}")
    cmd_markdown.add_argument("--thumbnail", type=str, default=constants.FOLDER_IMAGES_THUMBNAIL, help=f"缩略图所在目录，默认为 {constants.FOLDER_IMAGES_THUMBNAIL}")
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
                           granularities=args.granularity.split(','),
                           positions=args.position.split(','),
                           reducer=args.reducer,
                           folder=args.folder,
                           cleanup=not args.no_cleanup,
                           debug=args.debug)
    elif args.command == "thumbnail":
        generate_coverage_thumbnails(models, input=args.input, output=args.output, debug=args.debug)
        generate_embedding_thumbnails(models, input=args.input, output=args.output, debug=args.debug)
    elif args.command == "markdown":
        # markdown for all models
        logger.info("Generating markdown for all models to %s", args.markdown)
        generate_markdown_for_all(models, fullsize=args.fullsize, thumbnail=args.thumbnail, output=args.markdown)
        # markdown for readme
        template_file = os.path.join('docs', 'README.template.md')
        models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_readme.yaml')
        logger.info("Generating markdown for readme by %s and %s", template_file, models_file)
        generate_markdown_from_template(template_file=template_file, model_list_file=models_file, fullsize=args.fullsize, thumbnail=args.thumbnail, output='README.md')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
