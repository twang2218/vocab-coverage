
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

generate-coverage:
	python vocab_coverage/generate.py coverage --debug

generate-embedding:
	python vocab_coverage/generate.py embedding --debug

generate-thumbnails:
	python vocab_coverage/generate.py thumbnails --debug

generate-markdown:
	python vocab_coverage/generate.py markdown

generate: generate-coverage generate-embedding generate-thumbnails generate-markdown

clean-cache:
	rm -rf ~/.cache/huggingface/hub/*

sync-to-oss:
	aliyun oss cp \
		--region=ap-southeast-2 \
		--recursive \
		--include='*.jpg' \
		--jobs=10 \
		images/assets oss://lab99-syd-pub/vocab-coverage/

sync-to-oss-thumbnails:
	aliyun oss cp \
		--region=ap-southeast-2 \
		--recursive \
		--include='*.jpg' \
		--jobs=10 \
		images/thumbnails oss://lab99-syd-pub/vocab-coverage/thumbnails/

sync-from-oss:
	aliyun oss cp \
		--region=ap-southeast-2 \
		--recursive \
		--include='*.jpg' \
		--jobs=10 \
		oss://lab99-syd-pub/vocab-coverage/ images/assets

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
