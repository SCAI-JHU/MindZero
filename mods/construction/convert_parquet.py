import argparse
import json
import os

from datasets import Dataset

from qa_export import (
    _find_first_pick_put,
    _format_color_list,
    _holding_object_at_time,
    _object_label,
    _object_positions_at_time,
    _question_text_pick,
    _question_text_put_after_pick,
    _question_text_put_before_pick,
    _render_image_at_time,
    _run_inference_at_time,
    _state_text_at_time,
)


def _maybe_load(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _select_time_step(env_config, initial_state, actions, question_type):
    pick_index, put_index = _find_first_pick_put(env_config, initial_state, actions)
    if pick_index is None or put_index is None or put_index <= pick_index:
        return max(1, len(actions))
    pre_pick_time = max(0, pick_index // 2 + 1)
    mid_put_time = pick_index + max(1, (put_index - pick_index) // 2) + 1
    mid_put_time = min(mid_put_time, put_index)
    if question_type in (1, 2):
        return max(1, pre_pick_time)
    return max(1, mid_put_time)


def _infer_condition_label(
    env_config,
    initial_state,
    actions,
    time_step,
    object_labels,
    question_type,
):
    if question_type == 3:
        condition_obj = _holding_object_at_time(
            env_config, initial_state, actions, time_step
        )
    else:
        inference = _run_inference_at_time(
            env_config,
            initial_state,
            actions,
            time_step,
            temperature=0.1,
            gamma=0.95,
            action_cost=0.1,
            min_prob=0.0,
        )
        if question_type == 1:
            dist = inference.put_distribution.probs
        else:
            dist = inference.pick_distribution.probs
        condition_obj = max(dist, key=lambda k: dist[k]) if dist else None
    if condition_obj is None:
        return None
    return object_labels[condition_obj]


def convert_parquet(input_parquet, output_parquet, save_images=False, output_dir=None):
    dataset = Dataset.from_parquet(input_parquet)
    results = []
    if output_dir is None:
        output_dir = os.path.dirname(output_parquet)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in enumerate(dataset):
        env_config = _maybe_load(row["env_config"])
        initial_state = _maybe_load(row["initial_state"])
        actions = _maybe_load(row["actions"])
        question_type = int(row.get("question_type", 1))

        object_colors = env_config["object_colors"]
        object_shapes = env_config["object_shapes"]
        num_objects = env_config["num_objects"]
        object_labels = [
            _object_label(object_colors[i], object_shapes[i]) for i in range(num_objects)
        ]
        objects_list = _format_color_list(object_labels)

        time_step = _select_time_step(env_config, initial_state, actions, question_type)
        time_step = min(time_step, len(actions)) if actions else 1
        action_slice = actions[:time_step]

        state_text = _state_text_at_time(
            env_config, initial_state, actions, time_step, object_labels
        )
        object_positions = _object_positions_at_time(
            env_config, initial_state, actions, time_step, object_labels
        )
        image_bytes = _render_image_at_time(
            env_config,
            initial_state,
            actions,
            time_step,
            output_dir,
            idx,
            save_image=save_images,
        )

        condition_label = row.get("human_goal_given")
        if not condition_label:
            condition_label = _infer_condition_label(
                env_config,
                initial_state,
                actions,
                time_step,
                object_labels,
                question_type,
            )
        assert condition_label, "Missing human_goal_given and inference failed."
        condition_pos = object_positions.get(condition_label)

        if question_type == 1:
            question_text = _question_text_pick(
                objects_list, condition_label, condition_pos, state_text
            )
        elif question_type == 2:
            question_text = _question_text_put_before_pick(
                objects_list, condition_label, condition_pos, state_text
            )
        else:
            question_text = _question_text_put_after_pick(
                objects_list, condition_label, condition_pos, state_text
            )

        choices = _maybe_load(row.get("choices", {}))
        options = [(label, choices[label]) for label in ["a", "b"] if label in choices]
        option_desc = " ".join(
            [
                f"({label}) {text} at {object_positions.get(text)}."
                for label, text in options
            ]
        )
        problem = question_text + " " + option_desc

        results.append(
            {
                "images": [{"bytes": image_bytes}],
                "problem": problem,
                "answer": row.get("answer", ""),
                "choices": json.dumps(choices, ensure_ascii=True),
                "question_type": question_type,
                "human_goal_given": condition_label,
                "env_config": json.dumps(env_config, ensure_ascii=True),
                "initial_state": json.dumps(initial_state, ensure_ascii=True),
                "actions": json.dumps(action_slice, ensure_ascii=True),
            }
        )

    Dataset.from_list(results).to_parquet(output_parquet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    convert_parquet(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        save_images=args.save_images,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

