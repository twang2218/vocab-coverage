# 语言模型中文识字率分析

- [语言模型中文识字率分析](#语言模型中文识字率分析)
  - [项目介绍](#项目介绍)
  - [分析结果](#分析结果)
    - [中文支持较差的模型](#中文支持较差的模型)
    - [基于 bert-base-chinese 字表的模型](#基于-bert-base-chinese-字表的模型)
    - [ERNIE](#ernie)
    - [基于原生 LLaMA 的模型](#基于原生-llama-的模型)
    - [基于汉字扩表后的 LLaMA 的模型](#基于汉字扩表后的-llama-的模型)
    - [中文大语言模型](#中文大语言模型)
    - [其它大语言模型](#其它大语言模型)

## 项目介绍

本项目的目的是为了调查各个语言模型的中文识字率的情况，以此可以作为后续模型评估分析的参考。

为了分析模型的中文识字率，我们使用中华人民共和国教育部于2013年颁布的[《通用规范汉字表》](https://zh.wikipedia.org/zh-cn/%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8)（以下简称《通用规范汉字表》）作为标准，对各个语言模型在《通用规范汉字表》中可以识别的一、二、三级字表的汉字的数量进行统计。

在该字表中，共收录了8105个汉字，其中一级字表（常用字集）3500个，二级字表3000个，三级字表1605个。字表内容从[中文百科](https://www.zwbk2009.com/)中获取。

对于语言模型是否认知某个汉字的判断，我们通过对应语言模型所使用的 Tokenizer 是否可以对该汉字进行 `encode` 来判断。

- 模型不认识某汉字的判定为：
  - 模型对该汉字的编码结果为空；
  - 模型对该汉字的编码结果为 `unk_token_id`；
- 模型认识某汉字的判定为：
  - 模型对该汉字的编码结果长度为1；
- 如果编码结果长度大于1，这有可能是因为使用了 BBPE 的原因，一个不常出现的汉字被拆分成了多个 token。由于汉字被以UTF-8的形式编码，拆散该编码并不能体现汉字语义，因此，一个汉字被打散的编码越多，我们认为该模型对该汉字的认知程度可能越低。所以，对于编码结果长度大于1的情况，我们认为该模型对该汉字的认知程度为 `1 / len(encode_result)`，用以控制半透明程度。在识字率的计数中，将计数为 `0`。

> 在进行判断前，会先行去除前缀后缀的特殊token。

## 分析结果

### 中文支持较差的模型

|                   |                     |
|-------------------|---------------------|
|![](images/bert-base-cased.png)|![](images/roberta-large.png)|

### 基于 bert-base-chinese 字表的模型

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/bert-base-chinese.png)       |![](images/hfl_chinese-bert-wwm-ext.png)       |
|![](images/hfl_chinese-macbert-base.png)|![](images/hfl_chinese-legal-electra-base-generator.png)|
|![](images/shibing624_text2vec-base-chinese.png)| |

### ERNIE

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/nghuyong_ernie-1.0-base-zh.png)   |![](images/nghuyong_ernie-2.0-base-en.png)       |
|![](images/nghuyong_ernie-3.0-nano-zh.png)|![](images/nghuyong_ernie-3.0-xbase-zh.png)|
|![](images/nghuyong_ernie-health-zh.png)|![](images/nghuyong_ernie-gram-zh.png)|

### 基于原生 LLaMA 的模型

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/decapoda-research_llama-7b-hf.png)       | ![](images/TheBloke_koala-7B-HF.png)    |
|![](images/lmsys_vicuna-7b-delta-v1.1.png)       | ![](images/TheBloke_guanaco-7B-HF.png)    |
|![](images/TheBloke_wizardLM-7B-HF.png)       | ![](images/togethercomputer_RedPajama-INCITE-7B-Chat.png)    |
|![](images/openlm-research_open_llama_7b.png)       |    |

### 基于汉字扩表后的 LLaMA 的模型

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/shibing624_chinese-alpaca-plus-7b-hf.png)       | ![](images/shibing624_chinese-alpaca-plus-13b-hf.png)     |

### 中文大语言模型

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/THUDM_chatglm-6b.png) |![](images/fnlp_moss-moon-003-sft.png)             |
|![](images/shibing624_mengzi-t5-base-chinese-correction.png) | ![](images/shibing624_prompt-t5-base-chinese.png) |


### 其它大语言模型

|                                        |                                             |
|----------------------------------------|---------------------------------------------|
|![](images/bigscience_bloom-7b1.png)    | ![](images/tiiuae_falcon-7b-instruct.png)   |
|![](images/nomic-ai_gpt4all-j.png)      | ![](images/mosaicml_mpt-7b-instruct.png)    |
|![](images/OpenAssistant_oasst-sft-4-pythia-12b-epoch-3.5.png) |                      |

