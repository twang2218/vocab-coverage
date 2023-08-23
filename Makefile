
SHELL := /bin/bash
REMOTE_HOST = vast

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

package-clean:
	python setup.py clean --all
	rm -rf dist build vocab_coverage.egg-info

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
	python vocab_coverage/main.py charset --charset_file vocab_coverage/charset.json

generate-coverage: generate-coverage-char generate-coverage-token

generate-coverage-char:
	python vocab_coverage/generate.py coverage --debug --granularity=char

generate-coverage-token:
	python vocab_coverage/generate.py coverage --debug --granularity=token

generate-embedding:
	python vocab_coverage/generate.py embedding --debug --granularity=token,char,word --position=input,output

generate-embedding-token:
	python vocab_coverage/generate.py embedding --debug --granularity=token --position=input,output

generate-embedding-char:
	python vocab_coverage/generate.py embedding --debug --granularity=char --position=input,output

generate-embedding-word:
	python vocab_coverage/generate.py embedding --debug --granularity=word --position=input,output

generate-thumbnail:
	python vocab_coverage/generate.py thumbnail

generate-markdown: generate-thumbnail
	python vocab_coverage/generate.py markdown

generate: generate-coverage generate-embedding generate-thumbnail generate-markdown

sync-from-vast:
	rsync -avP 'vast:./vocab/images/fullsize/*' images/fullsize/

sync-graph: sync-from-vast generate-markdown sync-to-oss

clean-cache:
	rm -rf ~/.cache/huggingface/hub/*

sync-to-oss: sync-to-oss-fullsize sync-to-oss-thumbnail

sync-to-oss-fullsize:
	aliyun oss sync \
		--region=ap-southeast-2 \
		--update \
		--include='*.jpg' \
		--jobs=10 \
		images/fullsize oss://lab99-syd-pub/vocab-coverage/fullsize/

sync-to-oss-thumbnail:
	aliyun oss sync \
		--region=ap-southeast-2 \
		--update \
		--include='*.jpg' \
		--jobs=10 \
		images/thumbnail oss://lab99-syd-pub/vocab-coverage/thumbnail/

# sync-to-oss:
# 	aliyun oss cp \
# 		--region=ap-southeast-2 \
# 		--recursive \
# 		--include='*.jpg' \
# 		--jobs=10 \
# 		images/assets oss://lab99-syd-pub/vocab-coverage/

# sync-to-oss-thumbnails:
# 	aliyun oss cp \
# 		--region=ap-southeast-2 \
# 		--recursive \
# 		--include='*.jpg' \
# 		--jobs=10 \
# 		images/thumbnails oss://lab99-syd-pub/vocab-coverage/thumbnails/

sync-from-oss:
	aliyun oss sync \
		--region=ap-southeast-2 \
		--update \
		--include='*.jpg' \
		--jobs=10 \
		oss://lab99-syd-pub/vocab-coverage/ images/

# remote

remote-sync:
	rsync -avzP --exclude-from=.gitignore --exclude='*.png' --exclude='.git' --exclude='images' . $(REMOTE_HOST):./vocab

sync-to-remote-cache:
	rsync -avzP vocab_coverage/.cache/ $(REMOTE_HOST):./vocab/.cache/

sync-from-remote-cache:
	rsync -avzP $(REMOTE_HOST):./vocab/.cache/ vocab_coverage/.cache/

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
