import argparse
import os
from pprint import pprint

MODEL2ABBR = {
    "Qwen/Qwen3-VL-4B-Instruct": "q3vl4",
    "Qwen/Qwen3-VL-8B-Instruct": "q3vl8",
    "Qwen/Qwen3-4B-Instruct-2507": "q3l4",
    "meta-llama/Llama-3.2-3B-Instruct": "lm3l3",
    "meta-llama/Llama-3.1-8B-Instruct": "lm3l8",
    "../LLaMA-Factory/saves/qwen3-4b/lora_export/vh_proposer_0118_train_80_step_90/huggingface": "q3l4ft",
    "../LLaMA-Factory/saves/llama3-3b/lora_export/vh_0126-fmt-train_100/huggingface": "lm3l3ft",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True, choices=("gw", "hh"))
    parser.add_argument("--task", type=str, required=True, choices=("tom", "asst"))
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL2ABBR.keys()),
    )
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument(
        "--num_particles",
        type=int,
        default=8,
        help="Expected number of goal particles in gw_0125_asst reward (default: 8).",
    )
    return parser.parse_args()


def get_verl_args(args):
    version = f"{args.domain}_{args.task}"

    verl_args = [
        f"config=configs/{version}.yaml",
        f"trainer.experiment_name={MODEL2ABBR[args.model]}-{args.tag}",
        f"trainer.n_gpus_per_node={len(args.gpu.split(','))}",
        f"worker.actor.model.model_path={args.model}",
    ]

    return " ".join(sorted(verl_args))


def set_env_vars(args):
    env_vars = dict(
        CUDA_VISIBLE_DEVICES=args.gpu,
        PYTHONUNBUFFERED="1",
        PYTHONPATH=".",
        WANDB_ENTITY="scai-mindzero",
        GW_NUM_GOAL_PARTICLES=str(args.num_particles),
    )
    pprint(env_vars)
    for k, v in env_vars.items():
        os.environ[k] = v


if __name__ == "__main__":
    args = parse_args()
    set_env_vars(args)
    verl_args = get_verl_args(args)
    cmd = f"python3 -m libs.EasyR1.verl.trainer.main {verl_args}"
    pprint(cmd.split())

    os.system(cmd)
