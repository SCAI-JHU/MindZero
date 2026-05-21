<div align="center">
  <h1 align="center">
    MindZero
    <sup>
      <img src="assets/logo.png" alt="SCAI logo" width="23" height="40" align="absmiddle" />
    <sup>
  </h1>

  <p><b>Learning Online Mental Reasoning With Zero Annotations</b></p>

  [![Project Page](https://img.shields.io/badge/Homepage-Visit-blue?labelColor=gray&logo=homeassistantcommunitystore&logoColor=367BAF&style=flat-square)](https://scai.cs.jhu.edu/MindZero/)
  [![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset%20&%20Models-yellow?labelColor=gray&logo=huggingface&style=flat-square)](https://huggingface.co/collections/SCAI-JHU/mindzero/)
  [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red?&labelColor=gray&logo=arxiv&logoColor=brown&style=flat-square)](https://arxiv.org)
</div>


## 💡 TL;DR

> **MindZero** is a self-supervised reinforcement learning framework that trains multimodal large language models (MLLMs) for efficient and robust online mental reasoning.

During training, the model is rewarded for generating mental state hypotheses that maximize the likelihood of observed actions estimated by a planner, similar to model-based ToM reasoning. This method thus eliminates the need for explicit mental state annotations. After training, MindZero internalizes model-based reasoning into fast single-pass inference.

Across mental reasoning and AI assistance tasks, MindZero enhances MLLMs' intrinsic Theory of Mind (ToM) ability and significantly outperforms model-based methods in both accuracy and efficiency.

<p align="center">
  <img src="assets/framework.png"  alt="Overview"  width="600">
</p>

> **Highlights:** Small models & SOTA performance · Zero mental state annotations · Online inference with robust uncertainty updates over multiple hypotheses · Efficient reasoning suitable for real-time assistance


## 📝 Quick Start

### Code

```sh
git clone https://github.com/SCAI-JHU/MindZero /path/to/MindZero
cd /path/to/MindZero
git submodule update --init --recursive
```

### Environment

We use Apptainer (a safer Docker without root access) to manage the environment.

Alternatively, you can use the Docker image [`hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`](https://hub.docker.com/layers/hiyouga/verl/ngc-th2.8.0-cu12.9-vllm0.11.0) if you have Docker access.

```sh
# Set your MindZero path
mindzero_path="/path/to/MindZero"
cd ${mindzero_path}

# Build the container
apptainer build --fakeroot requirements/mindzero.sif requirements/mindzero.def

# Launch the container on a GPU-enabled node/environment
apptainer shell \
--nv \
--cleanenv \
--bind ${mindzero_path}:${mindzero_path} \
--bind /home/$(whoami):/home/$(whoami) \
--pwd ${mindzero_path} \
--shell /usr/bin/bash \
${mindzero_path}/requirements/mindzero.sif
```

### Dataset & Models

- Dataset
  ```sh
  hf download --repo-type dataset SCAI-JHU/MindZero
  ```

- Pretrained models
  ```sh
  # Reward model
  hf download Qwen/Qwen3-VL-235B-A22B-Instruct-2507-FP8  # Gridworld
  hf download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8     # Household

  # Base model
  hf download Qwen/Qwen3-VL-4B-Instruct                  # Gridworld
  hf download Qwen/Qwen3-VL-8B-Instruct                  # Gridworld
  hf download Qwen/Qwen3-4B-Instruct-2507                # Household
  hf download meta-llama/Llama-3.2-3B-Instruct           # Household
  hf download meta-llama/Llama-3.1-8B-Instruct           # Household
  ```

- MindZero checkpoints
  ```sh
  # Gridworld ToM
  hf download SCAI-JHU/MindZero-gw-tom-Qwen3-VL-4B-Instruct
  hf download SCAI-JHU/MindZero-gw-tom-Qwen3-VL-8B-Instruct
  # Gridworld Assistance
  hf download SCAI-JHU/MindZero-gw-asst-Qwen3-VL-4B-Instruct
  hf download SCAI-JHU/MindZero-gw-asst-Qwen3-VL-8B-Instruct
  # Household ToM
  hf download SCAI-JHU/MindZero-hh-tom-Qwen3-4B-Instruct-2507
  hf download SCAI-JHU/MindZero-hh-tom-Llama-3.2-3B-Instruct
  hf download SCAI-JHU/MindZero-hh-tom-Llama-3.1-8B-Instruct
  # Household Assistance
  hf download SCAI-JHU/MindZero-hh-asst-Qwen3-4B-Instruct-2507
  hf download SCAI-JHU/MindZero-hh-asst-Llama-3.2-3B-Instruct
  hf download SCAI-JHU/MindZero-hh-asst-Llama-3.1-8B-Instruct
  ```
### Training

1. Serve the reward model with vLLM (Minimum requirement: 4xA100 80GB).
   ```sh
   # Gridworld
   bash scripts/vllm_serve.sh hf:Qwen/Qwen3-VL-235B-A22B-Instruct-2507-FP8 qwen3-235b-fp8-vl 0,1,2,3 9991
   # Household
   bash scripts/vllm_serve.sh hf:Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 qwen3-235b-fp8 0,1,2,3 9991
   ```
   Then adjust [`mods/client_configs.py`](mods/client_configs.py) if your vLLM server is not on `http://localhost:9991`.

2. Launch RL training with EasyR1 on the remaining GPUs.
   ```sh
   export WANDB_API_KEY="wandb_v1_xxxxxxxx"

   # Gridworld-QA
   python scripts/train_config.py --domain gw --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-VL-4B-Instruct
   python scripts/train_config.py --domain gw --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-VL-8B-Instruct

   # Gridworld-Assistance
   python scripts/train_config.py --domain gw --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-VL-4B-Instruct
   python scripts/train_config.py --domain gw --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-VL-8B-Instruct
   # Household-QA
   python scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-4B-Instruct-2507
   python scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model meta-llama/Llama-3.2-3B-Instruct
   python scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model meta-llama/Llama-3.1-8B-Instruct

   # Household-Assistance
   python scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-4B-Instruct-2507
   python scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model meta-llama/Llama-3.2-3B-Instruct
   python scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model meta-llama/Llama-3.1-8B-Instruct
   ```

### Evaluation

- QA: [`mods/test_and_save.py`](mods/test_and_save.py)
- Assistance
  - Gridworld: [`scripts/eval_gw_speedup.sh`](scripts/eval_gw_speedup.sh)
  - Household: https://github.com/ShunchiZhang/online_watch_and_help/tree/MindZero

## 📖 Citation

```
@inproceedings{zhang2026mindzero,
  title     = {MindZero: Learning Online Mental Reasoning With Zero Annotations},
  author    = {Shunchi Zhang and Jin Lu and Chuanyang Jin and Yichao Zhou and Zhining Zhang and Tianmin Shu},
  booktitle = {Proceedings of the 43st International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
