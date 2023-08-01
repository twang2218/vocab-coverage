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

install_conda() {
    # Install miniconda
    MINICONDA=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
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

    # cuML
    sudo apt-get install -y --allow-change-held-packages cuda-toolkit-12-0 libcublas-12-0
    # it is a held package

    # for GPU int4 quantization via bitsandbytes
    # Fix issue: https://github.com/Facico/Chinese-Vicuna/issues/64#issuecomment-1595677969
    pip install nvidia-cusparse-cu11
    ln -s $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0 $CONDA_PREFIX/lib/libcudart.so
    ln -s $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.11 $CONDA_PREFIX/lib/libcusparse.so.11
}

install_cuml() {
    pip install umap-learn
    pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com
}

install_tools() {
    # Install system tools
    GOTOP=https://github.com/xxxserxxx/gotop/releases/download/v4.2.0/gotop_v4.2.0_linux_amd64.deb
    wget $GOTOP -O gotop.deb
    sudo dpkg -i gotop.deb
    rm gotop.deb
    sudo apt-get update
    sudo apt-get install -y htop tree make
    sudo apt-get install -y fonts-noto-cjk fonts-anonymous-pro fonts-noto-color-emoji
}

run_make() {
    # train ${PROJECT_NAME}
    cd ~/${PROJECT_NAME}
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda activate ${PROJECT_NAME}
    make "$@"
}

setup_tmux() {
    echo "Set up tmux"

    # 创建水平窗格
    tmux split-window -v

    # # 切换到下方窗格
    # tmux select-pane -D

    # # 执行命令a
    # tmux send-keys "make cpu" Enter

    # 获取当前窗格的宽度
    pane_width=$(tmux display-message -p "#{pane_width}")

    # 计算每个等分的宽度
    split_width=$((pane_width / 3))

    # 创建第一个子窗格
    tmux split-window -h

    # 调整第一个子窗格的宽度
    tmux resize-pane -x $((split_width * 2))

    # 创建第二个子窗格
    tmux split-window -h

    # 调整第二个子窗格的宽度
    tmux resize-pane -x $split_width

    # 运行 cpu 监控
    tmux send-keys "make cpu" Enter

    # 切换到第二个子窗格
    tmux select-pane -L

    # 运行 gpu 监控
    tmux send-keys "make gpu" Enter

    # 切换到第三个子窗格
    tmux select-pane -L

    # 运行 jupyter lab
    tmux send-keys "conda activate vocab ; make jupyter" Enter
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
            install_tools
            install_conda
            ;;
        install-cuml)
            install_cuml
            ;;
        make)
            run_make "$@"
            ;;
        tmux)
            setup_tmux
            ;;
        *)
            echo "Usage: $0 {install|install-cuml|make|tmux}"
            exit 1
    esac
}

main "$@"
