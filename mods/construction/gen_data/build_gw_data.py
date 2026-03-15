import io
import json
import os
import random
from functools import partial

from datasets import Dataset
from PIL import Image

from ..env import ConstructionEnv
from ..estimate_distribution import GoalParticles
from ..inference import ExactInference


def export_dataset_from_records(
    records_dir,
    output_dir,
    train_count=2000,
    test_count=200,
    seed=0,
    infer_end_time=False,
    threshold=0.8,
    infer_temperature=0.1,
    infer_gamma=0.95,
    infer_action_cost=0.1,
    infer_min_prob=0.0,
    min_pick_steps=8,
    min_put_steps=8,
    min_high=0.4,
    max_low=0.1,
):
    rng = random.Random(seed)
    record_paths = _collect_record_paths(records_dir)
    rng.shuffle(record_paths)

    total = min(len(record_paths), train_count + test_count)
    train_paths = record_paths[: min(train_count, total)]
    test_paths = record_paths[min(train_count, total) : total]

    train_examples, next_idx = _build_examples_for_paths(
        train_paths,
        output_dir,
        start_idx=0,
        seed=rng.randint(0, 1_000_000),
        infer_end_time=infer_end_time,
        threshold=threshold,
        infer_temperature=infer_temperature,
        infer_gamma=infer_gamma,
        infer_action_cost=infer_action_cost,
        infer_min_prob=infer_min_prob,
        min_pick_steps=min_pick_steps,
        min_put_steps=min_put_steps,
        min_high=min_high,
        max_low=max_low,
    )
    test_examples, _ = _build_examples_for_paths(
        test_paths,
        output_dir,
        start_idx=next_idx,
        seed=rng.randint(0, 1_000_000),
        infer_end_time=infer_end_time,
        threshold=threshold,
        infer_temperature=infer_temperature,
        infer_gamma=infer_gamma,
        infer_action_cost=infer_action_cost,
        infer_min_prob=infer_min_prob,
        min_pick_steps=min_pick_steps,
        min_put_steps=min_put_steps,
        min_high=min_high,
        max_low=max_low,
    )

    os.makedirs(output_dir, exist_ok=True)
    Dataset.from_list(train_examples).to_parquet(
        os.path.join(output_dir, "train.parquet")
    )
    Dataset.from_list(test_examples).to_parquet(
        os.path.join(output_dir, "test.parquet")
    )


def _build_examples_for_paths(
    record_paths,
    output_dir,
    start_idx=0,
    seed=0,
    infer_end_time=False,
    threshold=0.8,
    infer_temperature=0.1,
    infer_gamma=0.95,
    infer_action_cost=0.1,
    infer_min_prob=0.0,
    min_pick_steps=8,
    min_put_steps=8,
    min_high=0.4,
    max_low=0.1,
):
    examples = []
    idx = start_idx
    rng = random.Random(seed)
    for path in record_paths:
        items, idx = build_examples(
            record_path=path,
            output_dir=output_dir,
            idx=idx,
            seed=rng.randint(0, 1_000_000),
            infer_end_time=infer_end_time,
            threshold=threshold,
            infer_temperature=infer_temperature,
            infer_gamma=infer_gamma,
            infer_action_cost=infer_action_cost,
            infer_min_prob=infer_min_prob,
            min_pick_steps=min_pick_steps,
            min_put_steps=min_put_steps,
            min_high=min_high,
            max_low=max_low,
        )
        examples.extend(items)
    return examples, idx


def build_examples(
    record_path,
    output_dir,
    idx=0,
    seed=0,
    start_time=0,
    end_time=None,
    infer_end_time=False,
    threshold=0.8,
    infer_temperature=0.1,
    infer_gamma=0.95,
    infer_action_cost=0.1,
    infer_min_prob=0.0,
    min_pick_steps=8,
    min_put_steps=8,
    min_high=0.4,
    max_low=0.1,
):
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    env_config = record["env_config"]
    initial_state = record["initial_state"]
    actions = record["actions"]

    pick_index, put_index = _find_first_pick_put(env_config, initial_state, actions)
    if pick_index is None or put_index is None or put_index <= pick_index:
        return [], idx
    if pick_index < min_pick_steps or (put_index - pick_index) < min_put_steps:
        return [], idx
    object_colors = env_config["object_colors"]
    object_shapes = env_config["object_shapes"]
    num_objects = env_config["num_objects"]
    object_labels = [
        _object_label(object_colors[i], object_shapes[i]) for i in range(num_objects)
    ]
    objects_list = _format_color_list(object_labels)

    pre_pick_time = max(0, pick_index // 2 + 1)
    mid_put_time = pick_index + max(1, (put_index - pick_index) // 2) + 1
    mid_put_time = min(mid_put_time, put_index)
    max_time_step = max(pre_pick_time, mid_put_time)
    # print(f"pre_pick_time: {pre_pick_time}, mid_put_time: {mid_put_time}")

    if infer_end_time:
        end_time = _infer_end_time(
            record,
            threshold,
            infer_temperature,
            infer_gamma,
            infer_action_cost,
            infer_min_prob,
        )
    if end_time is None or end_time > len(actions) or end_time <= max_time_step:
        end_time = len(actions)
    action_slice = actions[start_time:end_time]

    question_specs = [
        (1, pre_pick_time, "pick"),
        (2, pre_pick_time, "put"),
        (3, mid_put_time, "put"),
    ]

    rng = random.Random(seed)
    examples = []
    for question_type, time_step, target in question_specs:
        image_bytes = _render_image_at_time(
            env_config, initial_state, action_slice, time_step, output_dir, idx
        )
        inference = _run_inference_at_time(
            env_config,
            initial_state,
            action_slice,
            time_step,
            infer_temperature,
            infer_gamma,
            infer_action_cost,
            infer_min_prob,
        )
        if target == "pick":
            dist = inference.pick_distribution.probs
            condition_obj = _top_object(inference.put_distribution.probs)
            if condition_obj is None:
                return [], idx
            condition_label = object_labels[condition_obj]
            question_text = _question_text_pick(objects_list, condition_label)
        else:
            dist = inference.put_distribution.probs
            if question_type == 3:
                condition_obj = _holding_object_at_time(
                    env_config, initial_state, action_slice, time_step
                )
                if condition_obj is None:
                    return [], idx
                condition_label = object_labels[condition_obj]
                question_text = _question_text_put_after_pick(
                    objects_list, condition_label
                )
            else:
                condition_obj = _top_object(inference.pick_distribution.probs)
                if condition_obj is None:
                    return [], idx
                condition_label = object_labels[condition_obj]
                question_text = _question_text_put_before_pick(
                    objects_list, condition_label
                )

        option_pair = _select_option_pair(
            dist,
            rng,
            min_high=min_high,
            max_low=max_low,
            exclude={condition_obj},
        )
        if option_pair is None:
            return [], idx
        top_obj, low_obj = option_pair

        goal_labels = [object_labels[top_obj], object_labels[low_obj]]
        rng.shuffle(goal_labels)
        options = [("a", goal_labels[0]), ("b", goal_labels[1])]
        option_desc = " ".join([f"({label}) {text}." for label, text in options])
        answer = "a" if goal_labels[0] == object_labels[top_obj] else "b"

        problem = question_text + " " + option_desc
        choices = json.dumps({label: text for label, text in options})
        actions_before = json.dumps(action_slice[:time_step], ensure_ascii=True)

        examples.append(
            {
                "images": [{"bytes": image_bytes}],
                "problem": problem,
                "answer": answer,
                "choices": choices,
                "question_type": question_type,
                "human_goal_given": condition_label,
                "env_config": json.dumps(env_config, ensure_ascii=True),
                "initial_state": json.dumps(initial_state, ensure_ascii=True),
                "actions": actions_before,
            }
        )
        idx += 1
    return examples, idx


def build_examples2(
    record_path,
    output_dir,
    idx=0,
    seed=0,
    start_time=0,
    end_time=None,
    infer_end_time=False,
    threshold=0.8,
    infer_temperature=0.1,
    infer_gamma=0.95,
    infer_action_cost=0.1,
    infer_min_prob=0.0,
    save_images=True,
):
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    env_config = record["env_config"]
    initial_state = record["initial_state"]
    actions = record["actions"]

    object_colors = env_config["object_colors"]
    object_shapes = env_config["object_shapes"]
    num_objects = env_config["num_objects"]
    object_labels = [
        _object_label(object_colors[i], object_shapes[i]) for i in range(num_objects)
    ]
    objects_list = _format_color_list(object_labels)

    if infer_end_time:
        end_time = _infer_end_time(
            record,
            threshold,
            infer_temperature,
            infer_gamma,
            infer_action_cost,
            infer_min_prob,
        )
    if end_time is None or end_time > len(actions):
        end_time = len(actions)
    action_slice = actions[start_time:end_time]

    examples = []
    total_steps = len(action_slice)
    max_step = max(0, total_steps - 1)
    schema_text = json.dumps(GoalParticles.model_json_schema(), ensure_ascii=True)
    for time_step in range(1, max_step + 1):
        image_bytes = _render_image_at_time(
            env_config,
            initial_state,
            action_slice,
            time_step,
            output_dir,
            idx,
            save_image=save_images,
        )
        propose = partial(_question_text_goal_probs, schema=schema_text)
        problem = propose(objects_list=objects_list)
        actions_before = json.dumps(action_slice[:time_step], ensure_ascii=True)

        examples.append(
            {
                "images": [{"bytes": image_bytes}],
                "problem": problem,
                "answer": "NONE",
                "env_config": json.dumps(env_config, ensure_ascii=True),
                "initial_state": json.dumps(initial_state, ensure_ascii=True),
                "actions": actions_before,
                "cur_step": time_step,
                "total_steps": max_step,
            }
        )
        idx += 1
    return examples, idx


def is_episode_suitable(
    record,
    temperature=0.1,
    gamma=0.95,
    action_cost=0.1,
    min_prob=0.0,
    min_pick_steps=8,
    min_put_steps=8,
    min_high=0.4,
    max_low=0.1,
):
    env_config = record["env_config"]
    initial_state = record["initial_state"]
    actions = record["actions"]

    pick_index, put_index = _find_first_pick_put(env_config, initial_state, actions)
    if pick_index is None or put_index is None or put_index <= pick_index:
        return False
    if pick_index < min_pick_steps or (put_index - pick_index) < min_put_steps:
        return False

    pre_pick_time = max(0, pick_index // 2 + 1)
    mid_put_time = pick_index + max(1, (put_index - pick_index) // 2) + 1
    mid_put_time = min(mid_put_time, put_index)
    question_specs = [
        (1, pre_pick_time, "pick"),
        (2, pre_pick_time, "put"),
        (3, mid_put_time, "put"),
    ]

    rng = random.Random(0)
    for question_type, time_step, target in question_specs:
        inference = _run_inference_at_time(
            env_config,
            initial_state,
            actions,
            time_step,
            temperature,
            gamma,
            action_cost,
            min_prob,
        )
        dist = (
            inference.pick_distribution.probs
            if target == "pick"
            else inference.put_distribution.probs
        )
        if target == "pick":
            condition_obj = _top_object(inference.put_distribution.probs)
            if condition_obj is None:
                return False
        else:
            if question_type == 3:
                condition_obj = _holding_object_at_time(
                    env_config, initial_state, actions, time_step
                )
                if condition_obj is None:
                    return False
            else:
                condition_obj = _top_object(inference.pick_distribution.probs)
                if condition_obj is None:
                    return False
        if (
            _select_option_pair(
                dist, rng, min_high=min_high, max_low=max_low, exclude={condition_obj}
            )
            is None
        ):
            return False
    return True


def _collect_record_paths(records_dir):
    paths = [
        os.path.join(records_dir, name)
        for name in os.listdir(records_dir)
        if name.endswith(".json")
    ]
    return sorted(paths)


def _render_image_at_time(
    env_config,
    initial_state,
    action_slice,
    time_step,
    output_dir,
    idx,
    save_image=True,
):
    env = ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )
    env.set_state(initial_state)
    for action in action_slice[:time_step]:
        env.step(tuple(action))
    render_array = env.render(
        mode="rgb_array", show_goal=False, show_traj=True, traj_len=-1
    )
    image = Image.fromarray(render_array)
    if save_image:
        image_dir = os.path.join(output_dir, "image")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{idx}.png")
        image.save(image_path)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _find_first_pick_put(env_config, initial_state, actions):
    env = ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )
    env.set_state(initial_state)
    pick_index = None
    put_index = None
    for i, action_pair in enumerate(actions):
        prev_holding = env.human_holding
        env.step(tuple(action_pair))
        curr_holding = env.human_holding
        if pick_index is None and prev_holding is None and curr_holding is not None:
            pick_index = i
        if (
            pick_index is not None
            and put_index is None
            and prev_holding is not None
            and curr_holding is None
        ):
            put_index = i
            break
    # print(f"pick_index: {pick_index}, put_index: {put_index}")
    return pick_index, put_index


def _run_inference_at_time(
    env_config,
    initial_state,
    actions,
    time_step,
    temperature,
    gamma,
    action_cost,
    min_prob,
):
    env = ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )
    env.set_state(initial_state)
    inference = ExactInference(
        env,
        temperature=temperature,
        gamma=gamma,
        action_cost=action_cost,
        min_prob=min_prob,
        print_each_step=False,
        track_pick_put=True,
    )
    inference.reset()
    for i in range(time_step):
        inference.step(tuple(actions[i]))
    return inference


def _top_object(dist):
    if not dist:
        return None
    return max(dist, key=lambda k: dist[k])


def _select_option_pair(dist, rng, min_high=0.4, max_low=0.1, exclude=None):
    if not dist:
        return None
    if exclude is None:
        exclude_set = set()
    elif isinstance(exclude, (set, list, tuple)):
        exclude_set = set(exclude)
    else:
        exclude_set = {exclude}
    filtered_dist = {obj: prob for obj, prob in dist.items() if obj not in exclude_set}
    if not filtered_dist:
        return None
    top_obj = max(filtered_dist, key=lambda k: filtered_dist[k])
    if filtered_dist[top_obj] < min_high:
        return None
    low_candidates = [
        obj for obj, prob in filtered_dist.items() if prob < max_low and obj != top_obj
    ]
    if not low_candidates:
        return None
    low_obj = rng.choice(low_candidates)
    return top_obj, low_obj


def _holding_object_at_time(env_config, initial_state, actions, time_step):
    env = ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )
    env.set_state(initial_state)
    for action in actions[:time_step]:
        env.step(tuple(action))
    return env.human_holding


def _question_text_pick(objects_list, condition_label):
    return (
        "<image> You are a helper agent in a GridWorld environment. "
        "You are the red robot, and the Human is the green robot. "
        f"There are multiple objects: {objects_list}. "
        "The Human's goal is to place two of the objects next to each other. "
        "The Human can move up, down, left, right, or stay, and can pick up an object when "
        "standing on it and not holding one, and can put down an object when holding one "
        "and the cell is empty. "
        "The Human's action trajectory so far is shown in the image. "
        f"Given that the Human intends to place an object next to the {condition_label}, "
        "which object is the Human more likely to pick up next?"
    )


def _question_text_put_before_pick(objects_list, condition_label):
    return (
        "<image> You are a helper agent in a GridWorld environment. "
        "You are the red robot, and the Human is the green robot. "
        f"There are multiple objects: {objects_list}. "
        "The Human's goal is to place two of the objects next to each other. "
        "The Human can move up, down, left, right, or stay, and can pick up an object when "
        "standing on it and not holding one, and can put down an object when holding one "
        "and the cell is empty. "
        "The Human's action trajectory so far is shown in the image. "
        f"Given that the Human intends to pick up the {condition_label}, "
        "which object is the Human more likely to place an object next to?"
    )


def _question_text_put_after_pick(objects_list, condition_label):
    return (
        "<image> You are a helper agent in a GridWorld environment. "
        "You are the red robot, and the Human is the green robot. "
        f"There are multiple objects: {objects_list}. "
        "The Human's goal is to place two of the objects next to each other. "
        "The Human can move up, down, left, right, or stay, and can pick up an object when "
        "standing on it and not holding one, and can put down an object when holding one "
        "and the cell is empty. "
        "The Human's action trajectory so far is shown in the image. "
        f"Given that the Human is holding the {condition_label}, "
        "which object is the Human more likely to place the carried object next to?"
    )

def _question_text_goal_probs(objects_list, schema):
    return (
        "<image> You are a helper agent in a GridWorld environment. "
        "You are the red robot, and the Human is the green robot. "
        f"There are multiple objects: {objects_list}. "
        "The Human's goal is to place two of the objects next to each other. "
        "The Human can move up, down, left, right, or stay, and can pick up an object when "
        "standing on it and not holding one, and can put down an object when holding one "
        "and the cell is empty. "
        "The Human's action trajectory so far is shown in the image. "
        "Please propose a probability distribution that includes 8 candidate paired goals "
        "and their probabilities. "
        "Your response should include the probability distribution formatted according to "
        f"this JSON schema: {schema}."
    )

def _sample_other_option(correct_goal, object_colors, object_shapes, seed):
    rng = random.Random(seed)
    goals = [
        (
            _object_label(object_colors[i], object_shapes[i]),
            _object_label(object_colors[j], object_shapes[j]),
        )
        for i in range(len(object_colors))
        for j in range(i + 1, len(object_colors))
        if (
            _object_label(object_colors[i], object_shapes[i]),
            _object_label(object_colors[j], object_shapes[j]),
        )
        != correct_goal
    ]
    return goals[rng.randint(0, len(goals) - 1)]


def _infer_end_time(
    record,
    threshold,
    temperature,
    gamma,
    action_cost,
    min_prob,
):
    env_config = record["env_config"]
    initial_state = record["initial_state"]
    actions = record["actions"]

    env = ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )
    env.set_state(initial_state)
    inference = ExactInference(
        env,
        temperature=temperature,
        gamma=gamma,
        action_cost=action_cost,
        min_prob=min_prob,
        print_each_step=False,
    )
    inference.reset()

    correct_goal = tuple(initial_state["human_goal"])
    if inference.goal_distribution.probs.get(correct_goal, 0.0) >= threshold:
        return 0

    for i, action_pair in enumerate(actions):
        inference.step(tuple(action_pair))
        if inference.goal_distribution.probs.get(correct_goal, 0.0) >= threshold:
            return i + 1
    return len(actions)


def _goal_label(goal):
    return f"{goal[0]}/{goal[1]}"


def _object_label(color, shape):
    return f"{color} {shape}"


def _format_color_list(colors):
    if len(colors) == 1:
        return colors[0]
    if len(colors) == 2:
        return f"{colors[0]} and {colors[1]}"
    return ", ".join(colors[:-1]) + f", and {colors[-1]}"
