import argparse
import os
from pathlib import Path
from pprint import pprint


MODEL2ABBR = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "q25vl3",
    "Qwen/Qwen3-VL-4B-Instruct": "q3vl4",
    "Qwen/Qwen3-VL-8B-Instruct": "q3vl8",
    "Qwen/Qwen3-4B-Instruct-2507": "q3l4",
    "Qwen/Qwen3-0.6B": "q3l06",
    "meta-llama/Llama-3.2-3B-Instruct": "lm3l3",
    "meta-llama/Llama-3.1-8B-Instruct": "lm3l8",
    "../LLaMA-Factory/saves/qwen3-4b/lora_export/vh_proposer_0118_train_80_step_90/huggingface": "q3l4ft",
    "../LLaMA-Factory/saves/llama3-3b/lora_export/vh_0126-fmt-train_100/huggingface": "lm3l3ft",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-0.6B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "../LLaMA-Factory/saves/qwen3-4b/lora_export/vh_proposer_0118_train_80_step_90/huggingface",
            "../LLaMA-Factory/saves/llama3-3b/lora_export/vh_0126-fmt-train_100/huggingface",
        ),
    )
    parser.add_argument("--train", type=str, default="")
    parser.add_argument("--test", type=str, default="")
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument(
        "--easyr1_root",
        type=str,
        default="/weka/scratch/tshu2/szhan256/github/hiyouga/EasyR1",
        choices=[
            "/home/ubuntu/icml/yichao/EasyR1",
            "/weka/scratch/tshu2/szhan256/github/hiyouga/EasyR1",
            "/weka/scratch/tshu2/szhan256/yichao/github/hiyouga/EasyR1",
        ],
    )
    parser.add_argument(
        "--tom_root",
        type=str,
        default="/weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM",
        choices=[
            "/home/ubuntu/icml/yichao/EasyR1/lambda_pack",
            "/weka/scratch/tshu2/szhan256/github/ShunchiZhang/StructuredToM",
            "/weka/scratch/tshu2/szhan256/yichao/github/ShunchiZhang/StructuredToM",
            "/weka/scratch/tshu2/szhan256/yichao/github/ShunchiZhang/StructuredToM4",
        ],
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        choices=["zyclbt", "shunchi-zhang"],
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        choices=["mindzero", "easy-r1"],
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=8,
        help="Expected number of goal particles in gw_0125_asst reward (default: 8).",
    )
    return parser.parse_args()


def get_verl_args(args):
    version = f"{args.data}_{args.task}"
    data_home = args.data_root / f"{args.data}_hiyouga/{args.task}"
    exp_name = "^^".join(
        [
            MODEL2ABBR[args.model],
            f"{args.reward}",
            f"{args.prompt}",
            f"{args.train}",
            f"{args.test}",
            args.tag,
        ]
    )

    if version == "gw_0125_tom":
        # max_input_len = 512
        max_input_len = 512
        if args.prompt == "nonthink":
            max_output_len = 5
        else:
            max_output_len = 1024

    elif version == "gw_0125_asst":
        max_input_len = 768
        if args.prompt in ("min_json", "none", "min_json_distance"):
            max_output_len = 512
        else:
            max_output_len = 4096

    elif version.startswith("vh_0126_asst"):
        max_input_len = 1280
        if args.prompt in ("min_json", "none"):
            max_output_len = 1280
        else:
            max_output_len = 4096

    elif version == "mmtom_0112_tom":
        max_input_len = 768
        max_output_len = 8192

    elif version == "web_0127_tom":
        max_input_len = 6127
        max_output_len = 5850

    else:
        max_input_len = 10000
        max_output_len = 20000

    if "asst" in args.task:
        if args.reward == "_fmt_only":
            save_freq = 3
            val_freq = 3
            save_limit = 5
            total_epochs = 1

        elif args.data == "gw_0125":
            if "VL-4B" in args.model:
                save_freq = 150
                val_freq = 10
                save_limit = 5
                total_epochs = 8
            elif "VL-8B" in args.model:
                save_freq = 200
                val_freq = 10
                save_limit = 4
                total_epochs = 10
        else:
            save_freq = 10
            val_freq = 10
            save_limit = 10
            total_epochs = 3

    elif "mmtom_" in args.data:
        # Qwen3-4B vLLM on 4xH100, test 600 cases: inference ~10 min + eval???
        save_freq = 20
        val_freq = 20
        save_limit = 1
        total_epochs = 20

    else:
        save_freq = 5
        val_freq = 5
        save_limit = 1
        total_epochs = 20

    verl_args = [
        f"config={args.config_root / version}.yaml",
        f"data.train_files={data_home / f'train{args.train}.parquet'}",
        f"data.val_files={data_home / f'test{args.test}.parquet'}",
        f"data.format_prompt={args.prompt_root / args.prompt}.jinja",
        f"worker.reward.reward_function={args.reward_root / version}.py:compute_score{args.reward}",
        f"trainer.project_name={version}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.val_freq={val_freq}",
        f"trainer.save_freq={save_freq}",
        f"trainer.save_limit={save_limit}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.n_gpus_per_node={len(args.gpu.split(','))}",
        f"worker.actor.model.model_path={args.model}",
        f"data.max_prompt_length={max_input_len}",
        f"data.max_response_length={max_output_len}",
        f"worker.rollout.max_num_batched_tokens={(max_input_len + max_output_len) * 2}",
    ]

    return " ".join(sorted(verl_args))


def set_env_vars(args):
    env_vars = dict(
        CUDA_VISIBLE_DEVICES=args.gpu,
        PYTHONUNBUFFERED="1",
        PYTHONPATH=".",
        WANDB_ENTITY=args.wandb_entity,
        WANDB_PROJECT=args.wandb_project,
        GW_NUM_GOAL_PARTICLES=str(args.num_particles),
    )
    pprint(env_vars)
    for k, v in env_vars.items():
        os.environ[k] = v


if __name__ == "__main__":
    args = parse_args()
    args.tom_root = Path(args.tom_root)
    args.easyr1_root = Path(args.easyr1_root)
    args.data_root = args.tom_root / "data"
    args.config_root = args.easyr1_root / "examples/configs"
    args.reward_root = args.easyr1_root / "examples/reward_function"
    args.prompt_root = args.easyr1_root / "examples/format_prompt"

    set_env_vars(args)

    verl_args = get_verl_args(args)
    cmd = f"python3 -m verl.trainer.main {verl_args}"
    pprint(cmd.split())

    os.system(cmd)
