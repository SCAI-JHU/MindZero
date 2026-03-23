<div align="center">
  <h1 align="center">
    MindZero<!--
--><sup>
    <img src="assets/logo.png" alt="SCAI logo" width="23" height="40" align="absmiddle" />
    <sup>
  </h1>

  <p><b>Learning Online Mental Reasoning With Zero Annotations</b></p>
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

### Data
https://huggingface.co/datasets/SCAI-JHU/MindZero


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

### Training

1. Serve reward model using vLLM
   ```sh
   # Gridworld
   bash scripts/vllm_serve.sh hf:Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 qwen3-235b-fp8-vl
   # Household
   bash scripts/vllm_serve.sh hf:Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 qwen3-235b-fp8
   ```

2. Launch RL training using EasyR1 (a clean fork of veRL)
   ```sh
   export WANDB_API_KEY="wandb_v1_xxxxxxxx"

   # Gridworld-QA
   python scripts/train_config.py --domain gw --task tom --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-VL-4B-Instruct-2507
   python scripts/train_config.py --domain gw --task tom --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-VL-8B-Instruct-2507

   # Gridworld-Assistance
   python scripts/train_config.py --domain gw --task asst --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-VL-4B-Instruct-2507
   python scripts/train_config.py --domain gw --task asst --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-VL-8B-Instruct-2507

   # Household-QA
   python scripts/train_config.py --domain hh --task tom --gpu_ids 0,1,2,3 --model_path meta-llama/Llama-3.2-3B-Instruct
   python scripts/train_config.py --domain hh --task tom --gpu_ids 0,1,2,3 --model_path meta-llama/Llama-3.1-8B-Instruct
   python scripts/train_config.py --domain hh --task tom --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-4B-Instruct-2507

   # Household-Assistance
   python scripts/train_config.py --domain hh --task asst --gpu_ids 0,1,2,3 --model_path meta-llama/Llama-3.2-3B-Instruct
   python scripts/train_config.py --domain hh --task asst --gpu_ids 0,1,2,3 --model_path meta-llama/Llama-3.1-8B-Instruct
   python scripts/train_config.py --domain hh --task asst --gpu_ids 0,1,2,3 --model_path Qwen/Qwen3-4B-Instruct-2507
   ```

### Evaluation

- QA: [`mods/test_and_save.py`](mods/test_and_save.py)
- Assistance
  - Gridworld: [`scripts/eval_gw_speedup.sh`](scripts/eval_gw_speedup.sh)
  - Household: https://github.com/ShunchiZhang/online_watch_and_help/tree/MindZero

## 📖 Citation

```
Coming soon :)
```
