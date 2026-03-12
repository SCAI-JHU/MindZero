import argparse
import base64
import json
import os
import re

import requests
from datasets import Dataset
from tqdm import tqdm

from estimate_likelihood import compute_score


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
    hint = (
        "The Human tends to pick up the closer of the two target objects first, "
        "then place it next to the farther one."
    )
    return (
        f"{problem_text}\n\n"
        f"{hint}\n\n"
        # "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        # "The reasoning process MUST BE enclosed within <thinking> </thinking> tags. "
        # "The final answer MUST BE put in \\boxed{}. "
        # "Keep the <thinking> part under 3 sentences and avoid repetition."
        "Answer with ONLY one of the following: \\boxed{a} or \\boxed{b}. "
        "Do not include any other text, explanations, or code fences."
    )


def _parse_answer(text):
    if not text:
        return None
    lowered = text.lower()
    boxed = re.findall(r"\\boxed\{\s*\(?\s*([ab])\s*\)?[^}]*\}", lowered)
    if boxed:
        return boxed[-1]
    paren = re.findall(r"\(\s*([ab])\s*\)", lowered)
    if paren:
        return paren[-1]
    match = re.findall(r"\b([ab])\b", lowered)
    return match[-1] if match else None


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
    correct = 0
    by_type = {}
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
        if "answer" in item and "ground_truth" not in item:
            item["ground_truth"] = item["answer"]
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
                accuracy = int(score["accuracy"])
                correct += accuracy
                question_type = source.get("question_type")
                if question_type is not None:
                    stats = by_type.setdefault(
                        question_type, {"correct": 0, "total": 0}
                    )
                    stats["correct"] += accuracy
                    stats["total"] += 1
                token_sums["prompt_tokens"] += source["prompt_tokens"]
                token_sums["completion_tokens"] += source["completion_tokens"]
                token_sums["total_tokens"] += source["total_tokens"]
                all_scores.append(
                    {
                        "response": source["response"],
                        "ground_truth": source.get("ground_truth"),
                        "question_type": question_type,
                        "accuracy": score["accuracy"],
                        "prompt_tokens": source["prompt_tokens"],
                        "completion_tokens": source["completion_tokens"],
                        "total_tokens": source["total_tokens"],
                        "prediction": score["prediction"],
                        "overall": score["overall"],
                        "prediction_estimate": score["prediction_estimate"],
                        "gt_estimate": score["gt_estimate"],
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
            accuracy = int(score["accuracy"])
            correct += accuracy
            question_type = source.get("question_type")
            if question_type is not None:
                stats = by_type.setdefault(question_type, {"correct": 0, "total": 0})
                stats["correct"] += accuracy
                stats["total"] += 1
            token_sums["prompt_tokens"] += source["prompt_tokens"]
            token_sums["completion_tokens"] += source["completion_tokens"]
            token_sums["total_tokens"] += source["total_tokens"]
            all_scores.append(
                {
                    "response": source["response"],
                    "ground_truth": source.get("ground_truth"),
                    "question_type": question_type,
                    "accuracy": score["accuracy"],
                    "prompt_tokens": source["prompt_tokens"],
                    "completion_tokens": source["completion_tokens"],
                    "total_tokens": source["total_tokens"],
                    "prediction": score["prediction"],
                    "overall": score["overall"],
                    "prediction_estimate": score["prediction_estimate"],
                    "gt_estimate": score["gt_estimate"],
                }
            )

    accuracy = correct / max(1, total)
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.3f}")
    if by_type:
        type_acc = {
            key: (val["correct"] / val["total"] if val["total"] else 0.0)
            for key, val in by_type.items()
        }
        print(f"Accuracy by type: {json.dumps(type_acc, ensure_ascii=True)}")
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
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
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

