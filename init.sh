#!/bin/bash

apt install -y ccache python3.10-venv build-essential cmake python3-dev kmod
# wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
# dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
# cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
# apt-get update
# sudo apt-get -y install cudnn-cuda-12
# rm cudnn-local-*
python3.10 -m venv venv
source venv/bin/activate
pip install -U "huggingface_hub[cli]"
pip install requests
pip install -r requirements-build.txt
huggingface-cli login
export MAX_JOBS=2
pip install -e .

# python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-2-7b-hf --dtype=auto --max-num-seqs=4 --max-num-batched-tokens=10000 2>&1 | tee -a aman.log
