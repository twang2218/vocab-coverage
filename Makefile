
MODELS_BERT = \
	bert-base-cased \
	bert-base-multilingual-cased \
	roberta-large \
	xlnet-base-cased \
	albert-base-v2 \
	xlm-roberta-base \
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
	swtx/ernie-2.0-base-chinese \
	nghuyong/ernie-3.0-nano-zh \
	nghuyong/ernie-3.0-base-zh \
	nghuyong/ernie-3.0-xbase-zh \
	nghuyong/ernie-health-zh \
	nghuyong/ernie-gram-zh \

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
	THUDM/chatglm2-6b \
	fnlp/moss-moon-003-sft-int4 \
	baichuan-inc/baichuan-7B \
	bigscience/bloom-7b1 \
	mosaicml/mpt-7b-instruct \
	tiiuae/falcon-7b-instruct \
	nomic-ai/gpt4all-j \
	OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5

MODELS_SHIBING624 = \
	shibing624/text2vec-base-chinese \
	shibing624/text2vec-base-chinese-sentence \
	shibing624/text2vec-base-chinese-paraphrase \
	shibing624/text2vec-base-multilingual \
	shibing624/prompt-t5-base-chinese \
	shibing624/mengzi-t5-base-chinese-correction \
	shibing624/chinese-alpaca-plus-7b-hf \
	shibing624/chinese-alpaca-plus-13b-hf

MODELS_OPENAI = \
	OpenAI/gpt-4 \
	OpenAI/gpt-3.5-turbo \
	OpenAI/text-embedding-ada-002 \
	OpenAI/text-davinci-003 \
	OpenAI/gpt2 \
	OpenAI/text-ada-001

MODELS_OPENAI_EMBEDDING = \
	OpenAI/text-embedding-ada-002

REMOTE_HOST = vast

define vocab_coverage_model
	@for model in $(1); do \
		filename=$$(echo $$model | sed 's/\//_/g'); \
		image_coverage=images/coverage/coverage.$$filename.jpg; \
		if [ -f $$image_coverage ]; then \
			echo "[$$model]: 词表中文识别率图文件已存在，跳过生成。"; \
		else \
			python vocab_coverage/main.py model --model_name $$model; \
		fi; \

		python vocab_coverage/main.py model --model_name $$model; \
	done
endef

define vocab_embeddings_model
	@for model in $(1); do \
		filename=$$(echo $$model | sed 's/\//_/g'); \
		image_input_embeddings=images/embeddings/embeddings.$$filename.input.jpg; \
		image_output_embeddings=images/embeddings/embeddings.$$filename.output.jpg; \
		ARGS=""; \
		if [ -f $$image_input_embeddings ]; then \
			echo "[$$model]: Input 词向量分布图文件已存在，跳过生成。"; \
			ARGS="--skip_input_embeddings" ; \
		fi; \
		if [ -f $$image_output_embeddings ]; then \
			echo "[$$model]: Output 词向量分布图文件已存在，跳过生成。"; \
		else \
			ARGS="$${ARGS} --output_embeddings" ; \
		fi; \
		if [ "$$ARGS" = "--skip_input_embeddings" ]; then \
			echo "[$$model]: 全部词向量分布图文件已存在，跳过生成。"; \
		else \
			python vocab_coverage/main.py embedding --debug $$ARGS --model_name $$model ; \
		fi; \
	done
endef


create-env:
	conda create -n vocab python=3.10 -y ; \
	conda activate vocab ; \
	pip install -r requirements.txt

update-env:
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

embedding: embedding-bert embedding-sbert embedding-ernie embedding-llama embedding-llm embedding-shibing624 embedding-openai

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

# 如果不存在 .env 文件，也没有OPENAI_API_KEY环境变量，会报错
embedding-openai:
	if [ ! -f .env ] && [ -z "$$OPENAI_API_KEY" ]; then \
		echo "Please set OPENAI_API_KEY in .env file or environment variable"; \
		exit 1; \
	fi
	$(call vocab_embeddings_model, $(MODELS_OPENAI_EMBEDDING))

embedding-thumbnails:
	@input_dir="images/embeddings"; \
	mkdir -p "$$input_dir/thumbnails"; \
	for file in "$$input_dir"/*.jpg; do \
		filename=$$(basename "$$file"); \
		thumbnail_file="$$input_dir/thumbnails/$$filename"; \
		if [ ! -f $$thumbnail_file ]; then \
			echo "Creating thumbnail for $$filename"; \
			convert "$$file" -quality 20 -resize 30% "$$input_dir/thumbnails/$$filename" || exit 2 ; \
		fi; \
	done

# remote

remote-sync:
	rsync -avzP --exclude-from=.gitignore --exclude='*.png' --exclude='.git' --exclude='images' . $(REMOTE_HOST):./vocab

remote-download-images:
	rsync -avzP $(REMOTE_HOST):./vocab/images/ ./images/

remote-provision: remote-sync
	ssh $(REMOTE_HOST) 'cd vocab && bash provision.sh install'

gpu:
	watch -n 1 nvidia-smi

cpu:
	gotop

jupyter:
	jupyter lab --port=7860 --allow-root
