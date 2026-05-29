
# ===== Dataset =====
hf download --repo-type dataset SCAI-JHU/MindZero

# ===== Pretrained models =====
# Reward model
hf download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8     # Household

# Base model
hf download Qwen/Qwen3-VL-4B-Instruct                  # Gridworld
hf download Qwen/Qwen3-VL-8B-Instruct                  # Gridworld
hf download Qwen/Qwen3-4B-Instruct-2507                # Household
hf download meta-llama/Llama-3.2-3B-Instruct           # Household
hf download meta-llama/Llama-3.1-8B-Instruct           # Household

# ===== MindZero checkpoints =====
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
