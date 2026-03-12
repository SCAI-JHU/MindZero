import argparse
import json

from datasets import Dataset

from estimate_likelihood import (
    _build_env,
    _object_labels_from_config,
    _resolve_object_index,
)
from inference import ExactInference


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _ground_truth_from_row(row):
    if "ground_truth" in row and row["ground_truth"] is not None:
        return row["ground_truth"]
    return row.get("answer")


def _infer_choice_probs(
    env_config,
    initial_state,
    actions,
    choices,
    human_goal_given,
    temperature,
    gamma,
    action_cost,
    min_prob,
):
    object_labels = _object_labels_from_config(env_config)
    condition_object = _resolve_object_index(
        human_goal_given, object_labels, env_config["object_colors"]
    )

    env = _build_env(env_config)
    env.set_state(initial_state)
    inference = ExactInference(
        env,
        temperature=temperature,
        gamma=gamma,
        action_cost=action_cost,
        min_prob=min_prob,
        track_pick_put=False,
        print_each_step=False,
    )
    inference.reset()
    for action_pair in actions:
        inference.step(tuple(action_pair))

    choice_probs = {}
    for label, choice_text in choices.items():
        choice_object = _resolve_object_index(
            choice_text, object_labels, env_config["object_colors"]
        )
        if choice_object == condition_object:
            choice_probs[label] = 0.0
            continue
        goal = tuple(sorted((int(choice_object), int(condition_object))))
        choice_probs[label] = inference.goal_distribution.probs.get(goal, 0.0)
    return choice_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--action_cost", type=float, default=0.1)
    parser.add_argument("--min_prob", type=float, default=1e-12)
    args = parser.parse_args()

    dataset = Dataset.from_parquet(args.parquet_path)
    total = len(dataset) if args.max_rows is None else min(len(dataset), args.max_rows)

    results = []
    correct = 0
    by_type = {}

    for i in range(total):
        row = dataset[i]
        env_config = _parse_json_field(row["env_config"])
        initial_state = _parse_json_field(row["initial_state"])
        actions = _parse_json_field(row["actions"])
        choices = _parse_json_field(row["choices"])
        human_goal_given = row["human_goal_given"]
        question_type = row.get("question_type")

        choice_probs = _infer_choice_probs(
            env_config,
            initial_state,
            actions,
            choices,
            human_goal_given,
            args.temperature,
            args.gamma,
            args.action_cost,
            args.min_prob,
        )
        prediction = max(choice_probs, key=lambda k: choice_probs[k])
        ground_truth = _ground_truth_from_row(row)
        if isinstance(ground_truth, str):
            ground_truth = ground_truth.strip().lower()
        accuracy = 1.0 if ground_truth == prediction else 0.0
        gt_prob = choice_probs.get(ground_truth, 0.0)
        max_false_prob = 0.0
        for label, prob in choice_probs.items():
            if label == ground_truth:
                continue
            if prob > max_false_prob:
                max_false_prob = prob
        if max_false_prob == 0.0:
            gt_ratio_gt_1000 = gt_prob > 0.0
        else:
            gt_ratio_gt_1000 = int((gt_prob / max_false_prob) >= 1000.0)

        results.append(
            {
                "row_idx": i,
                "question_type": question_type,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "accuracy": accuracy,
                "choice_probs": choice_probs,
                "gt_ratio_gt_1000": gt_ratio_gt_1000,
            }
        )

        correct += int(accuracy)
        if question_type is not None:
            stats = by_type.setdefault(question_type, {"correct": 0, "total": 0})
            stats["correct"] += int(accuracy)
            stats["total"] += 1

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

    overall_acc = correct / total if total > 0 else 0.0
    print(json.dumps({"accuracy": overall_acc, "count": total}))
    if by_type:
        type_acc = {
            key: (val["correct"] / val["total"] if val["total"] else 0.0)
            for key, val in by_type.items()
        }
        print(json.dumps({"accuracy_by_type": type_acc}))


if __name__ == "__main__":
    main()

