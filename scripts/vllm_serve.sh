model_path=${1:-"hf:Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"}
served_name=${2:-"local-model"}
gpu_ids=${3:-"0,1,2,3"}
port=${4:-9991}
n_gpus=$((1 + $(echo ${gpu_ids} | grep -o "," | wc -l)))

if [[ ${model_path} == hf:* ]]; then
  # * download & use pretrained model
  model_path=${model_path#hf:}
  echo "[1/2] Download or retrive huggingface models (${model_path}) ..."
  hf download $model_path
else
  # * convert finetuned model to huggingface format
  if [[ -z $(compgen -G ${model_path}/huggingface/*.safetensors) ]]; then
    echo "[1/2] Safetensors not found. Converting..."
    python3 scripts/model_merger.py --local_dir $model_path
    echo "[1/2] Conversion complete."
  else
    echo "[1/2] Safetensors found."
  fi
  model_path=${model_path}/huggingface
fi

echo "[2/2] Starting vLLM server..."
# Docs: https://docs.vllm.ai/en/latest/cli/serve.html
# save_vram_args="--enable-chunked-prefill --enforce-eager"
if [[ ${model_path} == *vl* ]] || [[ ${model_path} == *VL* ]]; then
  mm_args="--limit-mm-per-prompt '{\"image\": 20}'"
  echo "* Multi-modality Enabled! ${mm_args}"
else
  mm_args=""
  echo "* Multi-modality Disabled!"
fi

cmd="CUDA_VISIBLE_DEVICES=${gpu_ids} vllm serve \
  ${model_path} \
  --allowed-local-media-path /weka/scratch/tshu2 \
  --host 0.0.0.0 \
  --port ${port} \
  --trust-remote-code \
  --tensor-parallel-size ${n_gpus} \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --served-model-name ${served_name} \
  ${mm_args}"
echo $cmd | tr -s ' '
eval $cmd

# curl -X GET http://localhost:$port/v1/models
