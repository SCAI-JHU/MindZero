<div align="center">
  <h1><i>MindZero</i></h1>

  **Learning Online Mental Reasoning With Zero Annotations**

  *ICML 2026*

  🌐 [Website](https://scai.cs.jhu.edu/MindZero/)
  ·
  📄 [Paper](https://arxiv.org/pdf/2606.00240)
  ·
  🤗 [Dataset & Models](https://huggingface.co/collections/SCAI-JHU/mindzero/)
  
  🎙️ [Talk](https://recorder-v3.slideslive.com/?share=112361&s=9f8eaaf6-e910-44f2-98fb-2d1852389bfb)
  ·
  🖥️ [Slides](https://scai.cs.jhu.edu/MindZero/slides)
</div>

## 💡 TL;DR

***MindZero*** is a self-supervised reinforcement learning framework that trains multimodal large language models (MLLMs) for efficient and robust online mental reasoning.

During training, the model is rewarded for generating mental state hypotheses that maximize the likelihood of observed actions estimated by a planner, similar to model-based ToM reasoning. This method thus eliminates the need for explicit mental state annotations. After training, *MindZero* internalizes model-based reasoning into fast single-pass inference.

Across mental reasoning and AI assistance tasks in gridworld and household domains, *MindZero* enhances MLLMs' intrinsic ToM ability and significantly outperforms model-based methods in both accuracy and efficiency.

<p align="center">
  <img src="assets/framework.png"  alt="Overview"  width="600">
</p>

## 📝 Quick Start

### Code & Dataset & Models

```sh
# Clone & initialize repository
export mindzero_path="/path/to/MindZero"
git clone https://github.com/SCAI-JHU/MindZero ${mindzero_path}
cd ${mindzero_path}
git submodule update --init --recursive

# Download HuggingFace dataset and models
bash requirements/hf_download.sh
```

### Environment

We provide 3 equivalent ways to set up the environment. Choose the one that suits you best:

1. Manually install Python packages by running [`requirements/setup_env.sh`](requirements/setup_env.sh)
2. Use Docker image [`hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`](https://hub.docker.com/layers/hiyouga/verl/ngc-th2.8.0-cu12.9-vllm0.11.0)
3. Use Apptainer (a safer Docker without root access):
   ```sh
   apptainer build --fakeroot requirements/mindzero.sif requirements/mindzero.def
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

1. Serve the reward model with vLLM (Minimum requirement: 4xA100 80GB).
   ```sh
   # Only for Household
   bash scripts/vllm_serve.sh hf:Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 qwen3-235b-fp8 0,1,2,3 9991 256
   ```
   Then adjust [`mods/client_configs.py`](mods/client_configs.py) if your vLLM server is not on `http://localhost:9991`.

2. Launch RL training with EasyR1 on the remaining GPUs.
   ```sh
   export WANDB_API_KEY="wandb_v1_xxxxxxxx"

   # Gridworld-QA
   python3 scripts/train_config.py --domain gw --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-VL-4B-Instruct
   python3 scripts/train_config.py --domain gw --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-VL-8B-Instruct

   # Gridworld-Assistance
   python3 scripts/train_config.py --domain gw --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-VL-4B-Instruct
   python3 scripts/train_config.py --domain gw --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-VL-8B-Instruct

   # Household-QA
   python3 scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model Qwen/Qwen3-4B-Instruct-2507
   python3 scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model meta-llama/Llama-3.2-3B-Instruct
   python3 scripts/train_config.py --domain hh --task tom --gpu 4,5,6,7 --model meta-llama/Llama-3.1-8B-Instruct

   # Household-Assistance
   python3 scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model Qwen/Qwen3-4B-Instruct-2507
   python3 scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model meta-llama/Llama-3.2-3B-Instruct
   python3 scripts/train_config.py --domain hh --task asst --gpu 4,5,6,7 --model meta-llama/Llama-3.1-8B-Instruct
   ```

### Evaluation

- QA: [`mods/test_and_save.py`](mods/test_and_save.py)
- Gridworld Proactive Assistance: [`scripts/eval_gw_speedup.sh`](scripts/eval_gw_speedup.sh)
- Household Proactive Assistance: https://github.com/ShunchiZhang/online_watch_and_help/tree/MindZero

## 📖 Citation

If you find this work useful, please consider starring the repository and citing our paper:

```bibtex
@inproceedings{zhang2026mindzero,
  title     = {MindZero: Learning Online Mental Reasoning With Zero Annotations},
  author    = {Shunchi Zhang and Jin Lu and Chuanyang Jin and Yichao Zhou and Zhining Zhang and Tianmin Shu},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
