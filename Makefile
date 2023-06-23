
MODELS_BERT = \
	bert-base-cased \
	roberta-large \
	xlnet-base-cased \
	albert-base-v2 \
	google/flan-t5-base \
	google/electra-base-discriminator \
	bert-base-chinese \
	moka-ai/m3e-base \
	junnyu/wobert_chinese_plus_base \
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
	baichuan-inc/baichuan-7B \
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
	OpenAI/text-davinci-003 \
	OpenAI/gpt2 \
	OpenAI/text-ada-001

REMOTE_HOST = vast

define vocab_coverage_model
	@for model in $(1); do \
		python vocab_coverage/main.py model --model_name $$model; \
	done
endef

define vocab_embeddings_model
	@for model in $(1); do \
		python vocab_coverage/main.py embedding --model_name $$model --debug; \
	done
endef


install-deps:
	pip install -r requirements.txt

update-deps:
	pip freeze > requirements.txt

package:
	@echo "[ Clean up... ]"
	rm -rf dist build vocab_coverage.egg-info
	@echo "[ Build package... ]"
	python setup.py sdist bdist_wheel
	twine check dist/*
	@echo "[ Upload package... ]"
	twine upload dist/* --verbose

ENV_TEST_NAME = vocab_package_test
test-env:
	if conda env list | grep -q "^$(ENV_TEST_NAME) "; then \
		conda env remove -n $(ENV_TEST_NAME) -y; \
	fi
	conda create -n $(ENV_TEST_NAME) python=3.10 -y
	@echo "conda activate $(ENV_TEST_NAME)"

test-package:
	pip install -e .; \
	vocab-coverage --help

charsets:
	python vocab_coverage/main.py charset --charset_file charset.json

# model vocab coverage
model: model-bert model-sbert model-ernie model-llama model-llm model-shibing624 model-openai

model-bert:
	$(call vocab_coverage_model, $(MODELS_BERT))

model-sbert:
	$(call vocab_coverage_model, $(MODELS_SBERT))

model-ernie:
	$(call vocab_coverage_model, $(MODELS_ERNIE))

model-llama:
	$(call vocab_coverage_model, $(MODELS_LLAMA))

model-llm:
	$(call vocab_coverage_model, $(MODELS_LLM))

model-shibing624:
	$(call vocab_coverage_model, $(MODELS_SHIBING624))

model-openai:
	$(call vocab_coverage_model, $(MODELS_OPENAI))

# model embedding analysis

embedding: embedding-bert embedding-sbert embedding-ernie embedding-llama embedding-llm embedding-shibing624

embedding-bert:
	$(call vocab_embeddings_model, $(MODELS_BERT))

embedding-sbert:
	$(call vocab_embeddings_model, $(MODELS_SBERT))

embedding-ernie:
	$(call vocab_embeddings_model, $(MODELS_ERNIE))

embedding-llama:
	$(call vocab_embeddings_model, $(MODELS_LLAMA))

embedding-llm:
	$(call vocab_embeddings_model, $(MODELS_LLM))

embedding-shibing624:
	$(call vocab_embeddings_model, $(MODELS_SHIBING624))

embedding-resize:
	@input_dir="images/embeddings"; \
	mkdir -p "$$input_dir/70"; \
	mkdir -p "$$input_dir/small"; \
	for file in "$$input_dir"/*.jpg; do \
		filename=$$(basename "$$file"); \
		echo "Processing $$filename"; \
		output_file="$$output_dir/$$filename"; \
		convert "$$file" -quality 70 "$$input_dir/70/$$filename" || exit 1 ;\
		convert "$$file" -quality 20 -resize 30% "$$input_dir/small/$$filename" || exit 2 ; \
	done

# remote

remote-sync:
	rsync -avzP --exclude-from=.gitignore --exclude='*.png' --exclude='.git' --exclude='images' . $(REMOTE_HOST):./vocab-coverage

remote-download-images:
	rsync -avzP $(REMOTE_HOST):./vocab-coverage/images/ ./images/

remote-provision: remote-sync
	ssh $(REMOTE_HOST) 'cd vocab-coverage && bash provision.sh install'

gpu:
	watch -n 1 nvidia-smi

cpu:
	gotop

jupyter:
	jupyter lab --port=7860 --allow-root
