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
PROJECT_NAME=vocab-coverage

install() {
    MINICONDA=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

    # Install miniconda
    wget $MINICONDA -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init

    # Install system tools
    GOTOP=https://github.com/xxxserxxx/gotop/releases/download/v4.2.0/gotop_v4.2.0_linux_amd64.deb
    wget $GOTOP -O gotop.deb
    dpkg -i gotop.deb
    rm gotop.deb
    apt install -y htop tree make

    # create environment
    conda create -n ${PROJECT_NAME} -y python=3.10
    conda activate ${PROJECT_NAME}

    pip install -r ~/${PROJECT_NAME}/requirements.txt
}

run_make() {
    # train ${PROJECT_NAME}
    cd ~/${PROJECT_NAME}
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda activate ${PROJECT_NAME}
    make "$@"
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
        *)
            echo "Usage: $0 {install|make}"
            exit 1
    esac
}

main "$@"
