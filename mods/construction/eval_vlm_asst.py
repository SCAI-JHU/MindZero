import argparse
import base64
import json
import os

import requests
from datasets import Dataset
from tqdm import tqdm

from estimate_distribution import compute_score


def _encode_image_bytes(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_messages(images, problem_text):
    content = []
    for item in images:
        image_bytes = item["bytes"] if isinstance(item, dict) else item
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_bytes(image_bytes)},
            }
        )
    content.append({"type": "text", "text": problem_text})
    return [{"role": "user", "content": content}]


def _build_prompt(problem_text):
    return (
        f"{problem_text}\n\n"
        "Return ONLY a valid JSON object that matches the schema. "
        "Do not include any extra text, code fences, or explanations."
    )


def _load_parquet(path):
    return Dataset.from_parquet(path)


def _save_json(path, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=True, indent=2)


def evaluate(
    parquet_path,
    model_name,
    api_base,
    output_path,
    batch_size=8,
    max_samples=None,
    max_tokens=512,
    timeout=120,
    temperature=0.01,
    gamma=0.95,
    action_cost=0.1,
    min_prob=1e-12,
):
    dataset = _load_parquet(parquet_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}

    pending = []
    all_scores = []
    total = 0
    token_sums = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for row in tqdm(dataset, desc="Evaluating"):
        prompt_text = _build_prompt(row["problem"])
        messages = _build_messages(row["images"], prompt_text)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        item = dict(row)
        item["response"] = content
        item["prompt_tokens"] = prompt_tokens
        item["completion_tokens"] = completion_tokens
        item["total_tokens"] = total_tokens
        pending.append(item)

        if len(pending) >= batch_size:
            batch_scores = compute_score(
                pending,
                temperature=temperature,
                gamma=gamma,
                action_cost=action_cost,
                min_prob=min_prob,
            )
            for score, source in zip(batch_scores, pending):
                total += 1
                token_sums["prompt_tokens"] += source["prompt_tokens"]
                token_sums["completion_tokens"] += source["completion_tokens"]
                token_sums["total_tokens"] += source["total_tokens"]
                score_fields = {k: v for k, v in score.items() if k != "details"}
                all_scores.append(
                    {
                        "response": source["response"],
                        "prompt_tokens": source["prompt_tokens"],
                        "completion_tokens": source["completion_tokens"],
                        "total_tokens": source["total_tokens"],
                        **score_fields,
                    }
                )
            pending = []

    if pending:
        batch_scores = compute_score(
            pending,
            temperature=temperature,
            gamma=gamma,
            action_cost=action_cost,
            min_prob=min_prob,
        )
        for score, source in zip(batch_scores, pending):
            total += 1
            token_sums["prompt_tokens"] += source["prompt_tokens"]
            token_sums["completion_tokens"] += source["completion_tokens"]
            token_sums["total_tokens"] += source["total_tokens"]
            score_fields = {k: v for k, v in score.items() if k != "details"}
            all_scores.append(
                {
                    "response": source["response"],
                    "prompt_tokens": source["prompt_tokens"],
                    "completion_tokens": source["completion_tokens"],
                    "total_tokens": source["total_tokens"],
                    **score_fields,
                }
            )

    if total > 0:
        avg_prompt = token_sums["prompt_tokens"] / total
        avg_completion = token_sums["completion_tokens"] / total
        avg_total = token_sums["total_tokens"] / total
        print(
            "Average tokens: "
            f"prompt={avg_prompt:.1f}, completion={avg_completion:.1f}, total={avg_total:.1f}"
        )
    _save_json(output_path, all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--action_cost", type=float, default=0.1)
    parser.add_argument("--min_prob", type=float, default=1e-12)
    args = parser.parse_args()

    evaluate(
        parquet_path=args.parquet_path,
        model_name=args.model_name,
        api_base=args.api_base,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        temperature=args.temperature,
        gamma=args.gamma,
        action_cost=args.action_cost,
        min_prob=args.min_prob,
    )


if __name__ == "__main__":
    main()
