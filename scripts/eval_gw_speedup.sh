#!/bin/bash

# --- 参数处理逻辑 ---
MODEL_NAME=${1:-None}
MODEL=${2:-vlm}
PROMPT_MODE=${3:-normal}

# 定义要运行的 seed 列表
SEEDS=(10 20 30)

echo "---------------------------------------"
echo "Starting evaluation with:"
echo "Model: $MODEL"
echo "Model Name: $MODEL_NAME"
echo "Prompt Mode: $PROMPT_MODE"
echo "---------------------------------------"

# 循环运行三个 seed
for SEED in "${SEEDS[@]}"
do
    echo "[Run] Executing with --seed $SEED ..."

    python -m mod.construction.eval_speedup_full \
        --model "$MODEL" \
        --model_name "$MODEL_NAME" \
        --seed "$SEED" \
        --prompt_mode "$PROMPT_MODE"

    echo "[Done] Finished seed $SEED."
    echo "---------------------------------------"
done

echo "All tasks completed!"

python -m mod.construction.eval_speedup_full \
  --model vlm \
  --model_name 8b \
  --seed 10 \
  --api_base=http://n13:9991/v1 \
  --parquet_path /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/eval.parquet \
  --output_dir /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/asst_eval_8b_1755

python -m mod.construction.eval_speedup_full \
  --model no \
  --model_name 8b \
  --seed 10 \
  --api_base=http://n13:9991/v1 \
  --parquet_path /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/eval.parquet \
  --output_dir /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/asst_eval_8b_1755

python -m mod.construction.eval_speedup_full \
  --compute_speedup \
  --seed 10 \
  --parquet_path /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/eval.parquet \
  --output_dir /weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM/data/gw_0125_hiyouga/asst/asst_eval_8b_1755
