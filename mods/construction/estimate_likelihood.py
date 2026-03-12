import argparse
import json
import math

from datasets import Dataset

from .agent import HumanAgent
from .env import PICK, PUT, ConstructionEnv
from .utils import boltzmann_policy


def _action_prob(agent, obs, action_human):
    holding = obs["agents"][0]["holding"]
    if holding is not None and holding not in agent.goal:
        return 0.0
    if agent._should_pick(obs):
        return 1.0 if action_human == PICK else 0.0
    if agent._should_put(obs):
        return 1.0 if action_human == PUT else 0.0
    q_values = agent._compute_q(obs)
    if not q_values:
        return 0.0
    action_dist = boltzmann_policy(q_values, agent.temperature)
    return action_dist.get(action_human, 0.0)


def estimate(
    env,
    choice_object,
    actions,
    human_goal_given,
    temperature=0.01,
    gamma=0.95,
    action_cost=0.1,
    min_prob=1e-12,
):
    goal = (int(choice_object), int(human_goal_given))
    agent = HumanAgent(
        env,
        goal=goal,
        seed=None,
        priority=True,
        temperature=temperature,
        gamma=gamma,
        action_cost=action_cost,
    )
    log_probs = []
    for action_human, action_helper in actions:
        obs = env._get_obs()
        prob = _action_prob(agent, obs, action_human)
        prob = max(prob, min_prob)
        log_probs.append(math.log(prob))
        env.step((action_human, action_helper))
    return log_probs


def _load_row(parquet_path, row_idx):
    dataset = Dataset.from_parquet(parquet_path)
    row = dataset[int(row_idx)]
    env_config = json.loads(row["env_config"])
    initial_state = json.loads(row["initial_state"])
    actions = json.loads(row["actions"])
    choices = json.loads(row["choices"])
    human_goal_given = row["human_goal_given"]
    return env_config, initial_state, actions, choices, human_goal_given


def _object_labels_from_config(env_config):
    object_colors = env_config["object_colors"]
    object_shapes = env_config["object_shapes"]
    return [
        f"{object_colors[i]} {object_shapes[i]}"
        for i in range(env_config["num_objects"])
    ]


def _resolve_object_index(label, object_labels, object_colors):
    if label in object_labels:
        return object_labels.index(label)
    color_to_index = {}
    for i, color in enumerate(object_colors):
        color_to_index.setdefault(color, i)
    if label in color_to_index:
        return color_to_index[label]
    parts = label.split(" ")
    if len(parts) >= 2:
        candidate = " ".join(parts[:2])
        if candidate in object_labels:
            return object_labels.index(candidate)
    if parts and parts[0] in color_to_index:
        return color_to_index[parts[0]]
    assert False, f"Unknown object label: {label}"
    return None


def _build_env(env_config):
    return ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )


def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"


def compute_score(
    reward_inputs,
    temperature=0.01,
    gamma=0.95,
    action_cost=0.1,
    min_prob=1e-12,
    reward_type=None,
):
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        if len(response) <= 10:
            prediction = response[0].lower()
        else:
            prediction = extract_boxed_content(response).strip().lower()
        # if prediction not in ("a", "b"):
        #     lowered = response.lower()
        #     match = re.findall(r"\\boxed\{\s*([ab])\s*\}", lowered)
        #     if not match:
        #         match = re.findall(r"\(\s*([ab])\s*\)", lowered)
        #     if not match:
        #         match = re.findall(r"\b([ab])\b", lowered)
        #     prediction = match[-1] if match else None

        ground_truth = reward_input["answer"] if "answer" in reward_input else reward_input["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = ground_truth.strip().lower()
        accuracy = 1.0 if prediction == ground_truth else 0.0

        env_config = reward_input["env_config"]
        if isinstance(env_config, str):
            env_config = json.loads(env_config)
        initial_state = reward_input["initial_state"]
        if isinstance(initial_state, str):
            initial_state = json.loads(initial_state)
        actions = reward_input["actions"]
        if isinstance(actions, str):
            actions = json.loads(actions)
        choices = reward_input["choices"]
        if isinstance(choices, str):
            choices = json.loads(choices)
        human_goal_given = reward_input["human_goal_given"]

        object_labels = _object_labels_from_config(env_config)
        condition_object = _resolve_object_index(
            human_goal_given, object_labels, env_config["object_colors"]
        )

        def _estimate_choice(choice_label):
            if choice_label is None or choice_label not in choices:
                return [math.log(min_prob)]
            choice_object_label = choices[choice_label]
            choice_object = _resolve_object_index(
                choice_object_label, object_labels, env_config["object_colors"]
            )
            env = _build_env(env_config)
            env.set_state(initial_state)
            log_probs = estimate(
                env,
                choice_object,
                actions,
                condition_object,
                temperature=temperature,
                gamma=gamma,
                action_cost=action_cost,
                min_prob=min_prob,
            )

            return log_probs

        prediction_estimate = _estimate_choice(prediction)
        sum_reward = float(sum(prediction_estimate))
        last_reward = float(prediction_estimate[-1])

        gt_estimate = _estimate_choice(ground_truth)
        sum_gt_reward = float(sum(gt_estimate))
        last_gt_reward = float(gt_estimate[-1])

        # question_type = reward_input["question_type"]

        scores.append(
            {
                "overall": dict(sum=sum_reward, last=last_reward)[reward_type],
                "p_sum": sum_reward,
                "p_last": last_reward,
                "p_sum_gt": sum_gt_reward,
                "p_last_gt": last_gt_reward,
                "accuracy": accuracy,
                # "prediction": prediction,
                # "prediction_estimate": prediction_estimate,
                # "gt_estimate": gt_estimate,
                # "question_type": question_type,
            }
        )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--row_idx", type=int, required=True)
    parser.add_argument("--choice", required=True, help="Format: a or b")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--action_cost", type=float, default=0.1)
    parser.add_argument("--min_prob", type=float, default=1e-12)
    args = parser.parse_args()

    choice_label = args.choice.strip().lower()
    env_config, initial_state, actions, choices, human_goal_given = _load_row(args.parquet_path, args.row_idx)
    assert choice_label in choices, f"Choice must be one of {list(choices.keys())}"
    choice_object_label = choices[choice_label]
    env = _build_env(env_config)
    env.set_state(initial_state)
    object_labels = _object_labels_from_config(env_config)
    choice_object = _resolve_object_index(choice_object_label, object_labels, env_config["object_colors"])
    condition_object = _resolve_object_index(human_goal_given, object_labels, env_config["object_colors"])
    assert choice_object != condition_object, "Choice object matches human_goal_given"
    log_probs = estimate(
        env,
        choice_object,
        actions,
        condition_object,
        temperature=args.temperature,
        gamma=args.gamma,
        action_cost=args.action_cost,
        min_prob=args.min_prob,
    )
    print(json.dumps(log_probs))


if __name__ == "__main__":
    main()
