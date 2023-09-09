# -*- coding: utf-8 -*-

import argparse
import io
import logging
import os
import sys
import traceback
from typing import List
import yaml
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from vocab_coverage.classifier import Classifier
from vocab_coverage.crawler import get_chinese_charsets, get_token_charsets, get_chinese_word_dicts, get_chinese_sentence_datasets
from vocab_coverage.coverage import coverage_analysis
from vocab_coverage.embedding import embedding_analysis, release_resource
from vocab_coverage.lexicon import load_lexicon
from vocab_coverage import constants
from vocab_coverage.utils import (
    logger,
    generate_coverage_filename,
    generate_embedding_filename,
    generate_thumbnail_filename,
    generate_model_path
)

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
                      groups:List[str]=None,
                      granularities:List[str]=None,
                      folder=constants.FOLDER_IMAGES,
                      debug:bool=False):
    if groups is None:
        groups = []
    else:
        groups = [group for group in groups if len(group) > 0]    
    if granularities is None:
        granularities = [constants.GRANULARITY_TOKEN, constants.GRANULARITY_CHARACTER]
    for section in models:
        if len(groups) > 0 and section['group'] not in groups:
            # 指定了组的列表，但是当前组不在列表中，跳过
            if debug:
                logger.debug("[%s] Skip. Not in %s", section['group'], groups)
            continue
        for model_name in section["models"]:
            for granularity in granularities:
                # check coverage files
                coverage = find_coverage_file(model_name, granularity=granularity, folder=folder)
                if coverage is not None:
                    # fix coverage filename
                    standard_coverage = generate_coverage_filename(model_name, granularity=granularity, folder=folder)
                    if coverage and standard_coverage != coverage:
                        logger.info("Renaming %s => %s", coverage, standard_coverage)
                        os.rename(coverage, standard_coverage)
                        coverage = standard_coverage
                    if debug:
                        logger.debug("[%s] Skip. Exists [%s] coverage.", model_name, granularity)
                    continue
                try:
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
                       groups:List[str]=None,
                       granularities:List[str]=None,
                       positions:List[str]=None,
                       reducer:str=constants.REDUCER_TSNE,
                       folder=constants.FOLDER_IMAGES,
                       cleanup:bool=True,
                       no_cache:bool=False,
                       override:bool=False,
                       batch_size:int=100,
                       debug:bool=False):
    if groups is None:
        groups = []
    else:
        groups = [group for group in groups if len(group) > 0]    
    if granularities is None:
        granularities = constants.GRANULARITY_SETS
    if positions is None:
        positions = constants.EMBEDDING_POSITION_ALL

    for section in models:
        if len(groups) > 0 and section['group'] not in groups:
            # 指定了组的列表，但是当前组不在列表中，跳过
            if debug:
                logger.debug("[%s] Skip. Not in %s", section['group'], groups)
            continue
        for model_name in section["models"]:
            # check embedding files
            position_candidates = {}
            for granularity in granularities:
                position_candidates[granularity] = []
                for position in positions:
                    embedding_file = find_embedding_file(model_name, granularity=granularity, position=position, folder=folder, debug=debug)
                    if embedding_file is None or override:
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
            try:
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
                                    no_cache=no_cache,
                                    override=override,
                                    batch_size=batch_size,
                                    debug=debug)
                    logger.info("[%s] Generated [%s] embedding at %s", model_name, granularity, position_candidates[granularity])
            # pylint: disable=broad-except
            except Exception as ex:
                logger.error("[%s] embedding_analysis() failed. [%s]", model_name, ex)
                traceback.print_exc()
            finally:
                # release cache (Both for GPU and Storage)
                release_resource(model_name, clear_cache=cleanup)


def generate_thumbnail(model_name:str, filename:str, folder=constants.FOLDER_IMAGES, debug:bool=False):
    if not(filename and os.path.exists(filename)):
        raise ValueError(f"Cannot find file {filename}")
    thumbnail_filename = generate_thumbnail_filename(filename, folder=folder)
    if os.path.exists(thumbnail_filename):
        # 对比全尺寸文件的时间和缩略图的时间，如果缩略图的时间比全尺寸文件的时间晚，则不需要重新生成缩略图
        if os.path.getmtime(thumbnail_filename) > os.path.getmtime(filename):
            if debug:
                logger.debug("[%s] 无需生成缩略图 (%s)...", model_name, filename)
            return
    logger.info("[%s] 生成缩略图 (%s)...", model_name, filename)
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
            for granularity in granularities:
                coverage = find_coverage_file(model_name,
                                            granularity=granularity,
                                            folder=input)
                if coverage is not None:
                    generate_thumbnail(model_name, coverage, folder=output)
                else:
                    if debug:
                        logger.debug('[%s] 未发现 [%s] 覆盖率图。', model_name, granularity)

def generate_embedding_thumbnails(models:List[dict],
                                  granularities:List[str]=None,
                                  positions:List[str]=None,
                                  input:str=constants.FOLDER_IMAGES_FULLSIZE,
                                  output:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                  debug:bool=False):
    if granularities is None:
        granularities = constants.GRANULARITY_SETS
    if positions is None:
        positions = constants.EMBEDDING_POSITION_ALL
    for section in models:
        for model_name in section["models"]:
            for granularity in granularities:
                for position in positions:
                    embedding = find_embedding_file(model_name,
                                                    granularity=granularity,
                                                    position=position,
                                                    folder=input,
                                                    debug=debug)
                    if embedding is not None:
                        generate_thumbnail(model_name, embedding, folder=output)
                    else:
                        if debug:
                            logger.debug('[%s] 未发现 [%s] [%s] 向量分布图。', model_name, granularity, position)

def get_oss_url(image) -> str:
    # base_url = "https://lab99-syd-pub.oss-accelerate.aliyuncs.com/vocab-coverage/"
    # base_url = "https://lab99-syd-pub.oss-ap-southeast-2.aliyuncs.com/vocab-coverage/"
    base_url = "http://syd.jiefu.org/vocab-coverage/"
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
    granularities = constants.GRANULARITY_SETS
    coverages = {granularity: find_coverage_file(model_name, granularity=granularity, folder=fullsize) for granularity in granularities}
    positions = constants.EMBEDDING_POSITION_ALL
    embeddings = {granularity:
                  {position:
                    find_embedding_file(model_name, granularity=granularity, position=position, folder=fullsize)
                    for position in positions}
                    for granularity in granularities}
    if all(coverage is None for coverage in coverages.values()) and all(embedding is None for embedding in embeddings.values()):
        logger.error("[%s] cannot find any coverage or embedding file.", model_name)
        return ""

    # construct the markdown
    model_name = model_name.replace("/", "<br/>/<br/>")
    model_name = f'<b>{model_name}</b>'
    # title = f"| {model_name} | | |\n"
    # title = f"#### {model_name}\n"

    # header =  "| 颗粒度 | 完整覆盖率分析 | 输入向量分布图 | 输出向量分布图 |\n"
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
    word_content = " {input_embedding} | {output_embedding} |".format(
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_WORD][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )
    sentence_content = " {input_embedding} | {output_embedding} |".format(
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_SENTENCE][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_SENTENCE][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )
    paragraph_content = " {input_embedding} | {output_embedding} |".format(
        input_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_PARAGRAPH][constants.EMBEDDING_POSITION_INPUT], thumbnail=thumbnail),
        output_embedding=generate_markdown_for_model_graph(embeddings[constants.GRANULARITY_PARAGRAPH][constants.EMBEDDING_POSITION_OUTPUT], thumbnail=thumbnail)
    )

    return header + token_content + character_content + word_content + sentence_content + paragraph_content + "\n"

def generate_markdown_for_models(models:List[str],
                                 fullsize:str=constants.FOLDER_IMAGES_FULLSIZE,
                                 thumbnail:str=constants.FOLDER_IMAGES_THUMBNAIL,
                                 level:int=3) -> str:
    content = ""
    with io.StringIO() as f:
        # f.write("| 颗粒度 | 完整覆盖率分析 | 输入向量分布 | 输出向量分布 |\n")
        # f.write("| :---: | :---: | :---: | :---: |\n")
        header = "| 模型 | 完整性分析 (子词) | 入向量分布 (子词) | 出向量分布 (子词) | 完整性分析 (汉字) | 入向量分布 (汉字) | 出向量分布 (汉字) | 入向量分布 (词语) | 出向量分布 (词语) | 入向量分布 (句子) | 出向量分布 (句子) | 入向量分布 (段落) | 出向量分布 (段落) |\n"
        header += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        f.write(header)
        # Table body
        for model_name in models:
            # model_mark = "#" * level
            # f.write(f"{model_mark} {model_name}\n\n")
            f.write(generate_markdown_for_model(model_name, fullsize=fullsize, thumbnail=thumbnail))
        content = f.getvalue()
    return content

def markdown_text2anchor(text:str):
    anchor = ''
    for c in text:
        if c == ' ':
            anchor += '-'
        elif c in '()（）':
            # 括号会被忽略
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
        # 标题和目录
        f.write("# 所有模型的分析图\n\n")
        f.write("## 目录\n\n")
        for section in models:
            if section['group'] == 'new':
                # group: new 是为了辅助生成新模型的分析图，不需要在文档中显示
                continue
            name = section['name']
            f.write(f"- [{name}](#{markdown_text2anchor(name)})\n")
        f.write("\n\n")
        # 模型内容
        section_mark = "#" * section_level
        for section in models:
            if section['group'] == 'new':
                # group: new 是为了辅助生成新模型的分析图，不需要在文档中显示
                continue
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

    cmd_crawler = subcommands.add_parser('crawler', help='爬取用以统计识字率的字表文件')
    cmd_crawler.add_argument("--granularity", type=str, default="char", help="爬取的字表类型，可选值为 token, char（默认为 char）")
    cmd_crawler.add_argument("--local_source", type=str, default="", help="使用本地文件来避免重复下载")
    cmd_crawler.add_argument("--size", type=int, default=None, help="每类语料数量")
    cmd_crawler.add_argument("-f", "--file", type=str, default="", help="用以统计识字率的字表文件（默认为内置字符集文件）")
    cmd_crawler.add_argument("--indent", type=int, default=None, help="输出 JSON 文件时的缩进量（默认为 None）")

    cmd_coverage = subcommands.add_parser('coverage', help='Generate coverage graphs')
    cmd_coverage.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key")
    cmd_coverage.add_argument("--granularity", type=str, default="token,char", help="统计颗粒度，可选值为 token 或 char（默认为 token,char）")
    cmd_coverage.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmd_coverage.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_FULLSIZE}）")

    cmd_embedding = subcommands.add_parser('embedding', help='Generate embedding graphs')
    cmd_embedding.add_argument("--group", type=str, default="", help="要生成的模型组（默认为全部），组名称见 models.json 中的 key，可以为多项，用逗号分隔")
    cmd_embedding.add_argument("--granularity", type=str, default="token,char,word,sentence,paragraph", help="统计颗粒度，可选值为 (token、char、word）或组合（如：token,char），（默认为全部组合）")
    cmd_embedding.add_argument("--position", type=str, default="input,output", help="向量位置，可选值为 input, output 或 input,output（默认为 input,output）")
    cmd_embedding.add_argument("--folder", type=str, default=constants.FOLDER_IMAGES_FULLSIZE, help=f"输出文件夹（默认为 {constants.FOLDER_IMAGES_FULLSIZE}）")
    cmd_embedding.add_argument("--debug", action="store_true", help="是否输出调试信息")
    cmd_embedding.add_argument("--no_cleanup", action="store_true", help="是否保留模型缓存文件")
    cmd_embedding.add_argument("--reducer", type=str, default="tsne", help="降维算法，可选值为 tsne 或 umap（默认为 tsne）")
    cmd_embedding.add_argument("--no_cache", action="store_true", help="是否忽略 embedding 缓存")
    cmd_embedding.add_argument("--override", action="store_true", help="是否覆盖已有的 embedding 图片")
    cmd_embedding.add_argument("--batch_size", type=int, default=None, help="每批次处理的数据量")

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

    models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), constants.FILE_MODELS)
    with open(models_file, "r", encoding='utf-8') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    if args.command == 'crawler':
        # 爬取用以统计识字率的字表文件
        if args.granularity == constants.GRANULARITY_CHARACTER:
            datasets = get_chinese_charsets(debug=True)
        elif args.granularity == constants.GRANULARITY_TOKEN:
            datasets = get_token_charsets(debug=True)
        elif args.granularity == constants.GRANULARITY_WORD:
            if args.size is None:
                args.size = 1000
            datasets = get_chinese_word_dicts(category_size=args.size, debug=True)
        elif args.granularity == constants.GRANULARITY_SENTENCE:
            if args.size is None:
                args.size = 1000
            colormap = 'cmaps:percent_11lev'
            length_range = (25, 50)
            datasets = get_chinese_sentence_datasets(length_range, colormap=colormap, file=args.local_source, debug=True)
        elif args.granularity == constants.GRANULARITY_PARAGRAPH:
            if args.size is None:
                args.size = 1000
            colormap = 'tab10'
            length_range = (150,200)
            datasets = get_chinese_sentence_datasets(length_range, colormap=colormap, file=args.local_source, debug=True)
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
        Classifier(datasets, granularity=args.granularity).save(args.file, args.indent)
    elif args.command == "coverage":
        generate_coverage(models,
                          groups=args.group.split(','),
                          granularities=args.granularity.split(','),
                          folder=args.folder,
                          debug=args.debug)
    elif args.command == "embedding":
        generate_embedding(models,
                           groups=args.group.split(','),
                           granularities=args.granularity.split(','),
                           positions=args.position.split(','),
                           reducer=args.reducer,
                           folder=args.folder,
                           cleanup=not args.no_cleanup,
                           no_cache=args.no_cache,
                           override=args.override,
                           batch_size=args.batch_size,
                           debug=args.debug)
    elif args.command == "thumbnail":
        generate_coverage_thumbnails(models, input=args.input, output=args.output, debug=args.debug)
        generate_embedding_thumbnails(models, input=args.input, output=args.output, debug=args.debug)
    elif args.command == "markdown":
        # markdown for all models
        logger.info("Generating markdown for all models to %s", args.markdown)
        generate_markdown_for_all(models, fullsize=args.fullsize,
                                  thumbnail=args.thumbnail, output=args.markdown)
        # markdown for readme
        template_file = os.path.join('docs', 'README.template.md')
        models_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   constants.FILE_MODELS_README)
        logger.info("Generating markdown for readme by %s and %s", template_file, models_file)
        generate_markdown_from_template(template_file=template_file,
                                        model_list_file=models_file,
                                        fullsize=args.fullsize,
                                        thumbnail=args.thumbnail,
                                        output=constants.FILE_README_MD)
    else:
        parser.print_help()

if __name__ == "__main__":
    load_dotenv()
    main()
