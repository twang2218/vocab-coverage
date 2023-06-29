#!/bin/bash

# SSH config
# [.ssh/config]
# Host vast
#   HostName ssh5.vast.ai
#   User root
#   Port 18040

# SSH with port mapping
# ssh vast -L 8080:localhost:8080

# Project name
PROJECT_NAME=vocab

install() {
    MINICONDA=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

    # Install system tools
    GOTOP=https://github.com/xxxserxxx/gotop/releases/download/v4.2.0/gotop_v4.2.0_linux_amd64.deb
    wget $GOTOP -O gotop.deb
    dpkg -i gotop.deb
    rm gotop.deb
    apt install -y htop tree make
    apt install -y fonts-noto-cjk fonts-anonymous-pro fonts-noto-color-emoji

    # Install miniconda
    wget $MINICONDA -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init

    # create environment
    set -xe
    conda create -n ${PROJECT_NAME} -y python=3.10
    conda activate ${PROJECT_NAME}

    pip install -r ~/${PROJECT_NAME}/requirements.txt
    pip install jupyterlab

    # for GPU int4 quantization via bitsandbytes
    # Fix issue: https://github.com/Facico/Chinese-Vicuna/issues/64#issuecomment-1595677969
    pip install nvidia-cusparse-cu11
    ln -s $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0 $CONDA_PREFIX/lib/libcudart.so
    ln -s $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.11 $CONDA_PREFIX/lib/libcusparse.so.11
}

run_make() {
    # train ${PROJECT_NAME}
    cd ~/${PROJECT_NAME}
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda activate ${PROJECT_NAME}
    make "$@"
}

setup_tmux() {
    echo "TODO: setup tmux"
}

# split window
# ctrl-b + " 横切
# ctrl-b + % 纵切
# ctrl-b + 方向键切换焦点
# 其中一个窗口观察GPU使用情况： watch -n 1 nvidia-smi
# 另一个窗口观察CPU使用情况： gotop

main() {
    cmd=$1
    shift
    case $cmd in
        install)
            install
            ;;
        make)
            run_make "$@"
            ;;
        setup_tmux)
            setup_tmux
            ;;
        *)
            echo "Usage: $0 {install|make|setup_tmux}"
            exit 1
    esac
}

main "$@"
