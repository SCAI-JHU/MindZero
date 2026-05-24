import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", "/tmp/mindzero_hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/mindzero_hf_cache/datasets")

import pandas as pd
from datasets import Dataset, load_dataset
from jinja2 import Template
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from mods.client_configs import PROPOSER_CONFIGS
from mods.construction.estimate_likelihood import compute_score


DEFAULT_LOCAL_DATA = (
    REPO_ROOT
    / "../../ShunchiZhang/StructuredToM/data"
    / "gw_0125_hiyouga/tom"
    / "test.parquet"
).resolve()


def load_rows(data_path: str):
    if data_path.endswith(".parquet"):
        return Dataset.from_parquet(data_path)

    return load_dataset(data_path, "gw_tom", split="test")


def image_to_url(image) -> str:
    if isinstance(image, dict):
        image = image["bytes"]
    if isinstance(image, bytes):
        image_b64 = base64.b64encode(image).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def build_messages(example: dict[str, Any], prompt_template: str):
    prompt = Template(prompt_template.strip()).render(content=example["problem"])
    content = []
    images = iter(example.get("images") or [])
    for i, text in enumerate(prompt.split("<image>")):
        if i != 0:
            content.append({"type": "image_url", "image_url": {"url": image_to_url(next(images))}})
        if text:
            content.append({"type": "text", "text": text})

    return [{"role": "user", "content": content or prompt}]


async def infer_one(client, sem, example, prompt_template, gen_kwargs):
    async with sem:
        response = await client.chat.completions.create(
            messages=build_messages(example, prompt_template),
            **gen_kwargs,
        )
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "response": response.choices[0].message.content,
        }


async def infer_all(dataset, client, prompt_template, gen_kwargs, concurrency):
    sem = asyncio.Semaphore(concurrency)
    tasks = [infer_one(client, sem, dataset[i], prompt_template, gen_kwargs) for i in range(len(dataset))]
    return await tqdm.gather(*tasks, desc="gw_tom validation-equivalent test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen3-8b-vl-final", choices=sorted(PROPOSER_CONFIGS))
    parser.add_argument("--data-path", default=os.getenv("GW_TOM_TEST_DATA", DEFAULT_LOCAL_DATA.as_posix()))
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = PROPOSER_CONFIGS[args.model_key]
    gen_kwargs = dict(config["gen_kwargs"])
    gen_kwargs.update(
        temperature=0.8,
        top_p=0.95,
        n=1,
        max_tokens=5,
    )

    with open(REPO_ROOT / "prompts/input/gw_tom.jinja", encoding="utf-8") as f:
        prompt_template = f.read()

    dataset = load_rows(args.data_path)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.dry_run:
        print(json.dumps({"data_path": args.data_path, "num_examples": len(dataset), "gen_kwargs": gen_kwargs}, indent=2))
        print(json.dumps(build_messages(dataset[0], prompt_template), ensure_ascii=False, indent=2)[:2000])
        return

    client = AsyncOpenAI(**config["aclient_kwargs"])
    responses = asyncio.run(infer_all(dataset, client, prompt_template, gen_kwargs, args.concurrency))

    dataset = dataset.add_column("input_tokens", [r["input_tokens"] for r in responses])
    dataset = dataset.add_column("output_tokens", [r["output_tokens"] for r in responses])
    dataset = dataset.add_column("response", [r["response"] for r in responses])

    scores = pd.DataFrame(compute_score(dataset, reward_type="sum")).to_dict()
    for key, values in scores.items():
        dataset = dataset.add_column(key, list(values.values()))

    if args.save_path is None:
        save_path = Path("inference/gw_0125_hiyouga/tom") / f"{gen_kwargs['model']}.val_equiv.json"
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_keys = ["question_type", "answer", "response", "input_tokens", "output_tokens", *scores.keys()]
    subset = dataset.select_columns(save_keys)
    with save_path.with_suffix(".details.json").open("w", encoding="utf-8") as f:
        json.dump(subset.to_list(), f, ensure_ascii=False, indent=2)

    summary = {}
    for key, values in subset.to_dict().items():
        try:
            nums = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        avg = sum(nums) / len(nums)
        summary[key] = f"{avg:.2%}" if 0 <= avg <= 1 else f"{avg:.2f}"

    with save_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
