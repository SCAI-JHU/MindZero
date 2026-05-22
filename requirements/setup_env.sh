# hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 + mindzero.def 的 %post.

sudo python3 -m pip uninstall -y \
  torch \
  torchvision \
  torchaudio \
  flash-attn \
  vllm \
  transformer-engine \
  apex \
  megatron-core

sudo python3 -m pip install --no-cache-dir \
  "vllm==0.11.0" \
  "torch==2.8.0" \
  "torchvision==0.23.0" \
  "torchaudio==2.8.0" \
  "tensordict==0.10.0" \
  "torchdata==0.11.0" \
  "transformers[hf_xet]==4.57.0" \
  "accelerate==1.10.1" \
  "datasets==4.1.1" \
  "peft==0.17.1" \
  "hf-transfer==0.1.9" \
  "numpy==1.26.4" \
  "pyarrow==21.0.0" \
  "grpcio==1.75.1" \
  "optree==0.15.0" \
  "pandas==2.2.3" \
  "ray[default]==2.49.2" \
  "codetiming==1.4.0" \
  "hydra-core==1.3.2" \
  "pylatexenc==2.10" \
  "qwen-vl-utils==0.0.14" \
  "wandb==0.24.2" \
  "liger-kernel==0.6.2" \
  "mathruler==0.1.0" \
  "pytest==8.1.1" \
  "yapf==0.43.0" \
  "py-spy==0.4.1" \
  "pre-commit==4.3.0" \
  "ruff==0.13.3"

ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
WHL_BASE="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
WHL="requirements/${WHL_BASE}"
if [ ! -f "${WHL}" ]; then
  mkdir -p "$(dirname "${WHL}")"
  curl -L -o "${WHL}" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHL_BASE}"
fi
sudo python3 -m pip install --no-cache-dir "${WHL}"

sudo python3 -m pip install --no-cache-dir \
  "nvitop==1.6.2" \
  "json-repair==0.57.1" \
  "litellm==1.81.11" \
  "prettytable==3.17.0"
