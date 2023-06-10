
MODELS_BERT = \
	bert-base-cased \
	bert-base-chinese \
	roberta-large \
	hfl/chinese-bert-wwm-ext \
	hfl/chinese-macbert-base

MODELS_ERNIE = \
	nghuyong/ernie-1.0-base-zh \
	nghuyong/ernie-2.0-base-en \
	nghuyong/ernie-3.0-nano-zh \
	nghuyong/ernie-3.0-xbase-zh \
	nghuyong/ernie-health-zh \
	nghuyong/ernie-gram-zh

MODELS_LLAMA = \
	decapoda-research/llama-7b-hf \
	togethercomputer/RedPajama-INCITE-7B-Chat \
	TheBloke/guanaco-7B-HF \
	TheBloke/koala-7B-HF \
	TheBloke/wizardLM-7B-HF \
	TheBloke/vicuna-7B-1.1-HF \
	openlm-research/open_llama_7b

MODELS_LLM = \
	THUDM/chatglm-6b \
	fnlp/moss-moon-003-sft \
	bigscience/bloom-7b1 \
	mosaicml/mpt-7b-instruct \
	tiiuae/falcon-7b-instruct \
	nomic-ai/gpt4all-j \
	OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5

MODELS_SHIBING624 = \
	shibing624/text2vec-base-chinese \
	shibing624/chinese-alpaca-plus-7b-hf \
	shibing624/chinese-alpaca-plus-13b-hf \
	shibing624/prompt-t5-base-chinese \
	shibing624/mengzi-t5-base-chinese-correction


install-deps:
	pip install -r requirements.txt

update-deps:
	pip freeze > requirements.txt

generate-charsets:
	python generate_charsets.py

run: run-bert run-ernie run-llama run-llm run-shibing624

run-bert:
	@for model in $(MODELS_BERT); do \
		python vocab_check.py --model_name $$model; \
	done

run-ernie:
	@for model in $(MODELS_ERNIE); do \
		python vocab_check.py --model_name $$model; \
	done

run-llama:
	@for model in $(MODELS_LLAMA); do \
		python vocab_check.py --model_name $$model; \
	done

run-llm:
	@for model in $(MODELS_LLM); do \
		python vocab_check.py --model_name $$model; \
	done

run-shibing624:
	@for model in $(MODELS_SHIBING624); do \
		python vocab_check.py --model_name $$model; \
	done

