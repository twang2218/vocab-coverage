
MODELS_BERT = \
	bert-base-cased \
	roberta-large \
	xlnet-base-cased \
	albert-base-v2 \
	google/flan-t5-base \
	google/electra-base-discriminator \
	bert-base-chinese \
	hfl/chinese-bert-wwm-ext \
	hfl/chinese-macbert-base \
	hfl/chinese-legal-electra-base-generator

MODELS_SBERT = \
	sentence-transformers/all-MiniLM-L6-v2 \
	sentence-transformers/all-mpnet-base-v2 \
	sentence-transformers/multi-qa-mpnet-base-dot-v1 \
	sentence-transformers/paraphrase-MiniLM-L6-v2 \
	sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
	sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
	sentence-transformers/distiluse-base-multilingual-cased-v2 \
	sentence-transformers/all-roberta-large-v1

MODELS_ERNIE = \
	nghuyong/ernie-1.0-base-zh \
	nghuyong/ernie-2.0-base-en \
	nghuyong/ernie-3.0-nano-zh \
	nghuyong/ernie-3.0-xbase-zh \
	nghuyong/ernie-health-zh \
	nghuyong/ernie-gram-zh \
	swtx/ernie-2.0-base-chinese

MODELS_LLAMA = \
	decapoda-research/llama-7b-hf \
	togethercomputer/RedPajama-INCITE-7B-Chat \
	TheBloke/guanaco-7B-HF \
	TheBloke/koala-7B-HF \
	TheBloke/wizardLM-7B-HF \
	lmsys/vicuna-7b-delta-v1.1 \
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

MODELS_OPENAI = \
	OpenAI/gpt-4 \
	OpenAI/gpt-3.5-turbo \
	OpenAI/text-embedding-ada-002 \
	OpenAI/text-davinci-003

install-deps:
	pip install -r requirements.txt

update-deps:
	pip freeze > requirements.txt

charsets:
	python generate_charsets.py

run: run-bert run-sbert run-ernie run-llama run-llm run-shibing624 run-openai

run-bert:
	@for model in $(MODELS_BERT); do \
		python vocab_check.py --model_name $$model; \
	done

run-sbert:
	@for model in $(MODELS_SBERT); do \
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

run-openai:
	@for model in $(MODELS_OPENAI); do \
		python vocab_check.py --model_name $$model; \
	done
