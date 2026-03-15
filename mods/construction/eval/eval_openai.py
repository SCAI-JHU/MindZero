import argparse
import base64
import json
import os
import re

from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm


def _encode_image_bytes(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_messages(images, problem_text, text_only=False):
    content = []
    if not text_only:
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
        # "Answer with ONLY one of the following: \\boxed{a} or \\boxed{b}. "
        # "Do not include any other text, explanations, or code fences."
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <thinking> </thinking> tags. "
        "The final answer MUST BE put in \\boxed{}. "
        "Keep the <thinking> part under 3 sentences and avoid repetition."
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


def _update_counts(stats, question_type, correct, prompt_tokens, completion_tokens):
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["prompt_tokens"] += prompt_tokens
    stats["completion_tokens"] += completion_tokens

    if question_type not in stats["by_type"]:
        stats["by_type"][question_type] = {
            "total": 0,
            "correct": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    bucket = stats["by_type"][question_type]
    bucket["total"] += 1
    bucket["correct"] += int(correct)
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens


def _safe_usage(usage):
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)


def _load_parquet(path):
    return Dataset.from_parquet(path)


def evaluate(
    parquet_path,
    model_name,
    output_path=None,
    max_samples=None,
    max_completion_tokens=64,
    timeout=120,
    text_only=False,
):
    dataset = _load_parquet(parquet_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    results = []
    stats = {
        "total": 0,
        "correct": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "by_type": {},
    }

    api_key = "[[HIDDEN]]"
    assert api_key, "Missing OPENAI_API_KEY in environment."
    client = OpenAI(api_key=api_key)

    for idx, row in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt_text = _build_prompt(row["problem"])
        messages = _build_messages(
            row.get("images", []), prompt_text, text_only=text_only
        )
        data = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )
        content = data.choices[0].message.content
        pred = _parse_answer(content)
        gold = row["answer"].strip().lower()
        correct = pred == gold
        prompt_tokens, completion_tokens = _safe_usage(getattr(data, "usage", None))
        _update_counts(
            stats,
            row.get("question_type", "unknown"),
            correct,
            prompt_tokens,
            completion_tokens,
        )
        if output_path is not None:
            results.append(
                {
                    "index": idx,
                    "question_type": row.get("question_type", "unknown"),
                    "answer": gold,
                    "prediction": pred,
                    "correct": bool(correct),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "output_text": content,
                }
            )

    total = max(1, stats["total"])
    overall_acc = stats["correct"] / total
    avg_prompt = stats["prompt_tokens"] / total
    avg_completion = stats["completion_tokens"] / total

    print(f"Total: {stats['total']}")
    print(f"Accuracy: {overall_acc:.3f}")
    print(f"Avg prompt tokens: {avg_prompt:.1f}")
    print(f"Avg completion tokens: {avg_completion:.1f}")
    print("By question_type:")
    for qtype, bucket in sorted(stats["by_type"].items(), key=lambda x: str(x[0])):
        t = max(1, bucket["total"])
        acc = bucket["correct"] / t
        avg_p = bucket["prompt_tokens"] / t
        avg_c = bucket["completion_tokens"] / t
        print(f"  {qtype}: n={bucket['total']} acc={acc:.3f} in={avg_p:.1f} out={avg_c:.1f}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--model_name", default="gpt-5.2")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()

    evaluate(
        parquet_path=args.parquet_path,
        model_name=args.model_name,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        text_only=args.text_only,
    )


if __name__ == "__main__":
    main()
