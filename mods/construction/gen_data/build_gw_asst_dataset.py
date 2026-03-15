import argparse
import json
import os
import random

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from ..agent import HumanAgent
from ..env import STAY, ConstructionEnv
from build_gw_data import build_examples2, is_episode_suitable
from episode_recorder import EnvSaver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="../data/asst_data3")
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--action-cost", type=float, default=0.1)
    parser.add_argument("--infer-end-time", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--infer-temperature", type=float, default=0.01)
    parser.add_argument("--infer-gamma", type=float, default=0.95)
    parser.add_argument("--infer-action-cost", type=float, default=0.1)
    parser.add_argument("--infer-min-prob", type=float, default=0.0001)
    parser.add_argument("--min-pick-steps", type=int, default=6)
    parser.add_argument("--min-put-steps", type=int, default=6)
    parser.add_argument("--min-high", type=float, default=0.3)
    parser.add_argument("--max-low", type=float, default=0.01)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    record_dir = os.path.join(args.output_dir, "episode_record")
    os.makedirs(record_dir, exist_ok=True)
    if args.save_images:
        image_dir = os.path.join(args.output_dir, "image")
        os.makedirs(image_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    for episode_id in tqdm(range(1, args.train_episodes + 1), desc="Generating episodes"):
        while True:
            layout_id, object_colors, object_shapes = _episode_spec(episode_id)
            env_seed = int(rng.randint(0, 1_000_000))
            env = ConstructionEnv(
                grid_size=(10, 10),
                max_steps=args.max_steps,
                seed=env_seed,
                layout_id=layout_id,
                object_colors=object_colors,
                object_shapes=object_shapes,
            )
            obs = env.reset(human_goal=None)
            goal_candidates = _goal_candidates(env)
            if not goal_candidates:
                continue
            env.human_goal = goal_candidates[int(rng.randint(0, len(goal_candidates)))]
            obs = env._get_obs()

            human_agent = HumanAgent(
                env,
                goal=env.human_goal,
                seed=int(rng.randint(0, 1_000_000)),
                temperature=args.temperature,
                gamma=args.gamma,
                action_cost=args.action_cost,
            )

            saver = EnvSaver(env)
            saver.start()

            done = False
            timed_out = False
            for _ in range(args.max_steps):
                action_human = human_agent.sample_action(obs)
                action_helper = STAY
                obs, done, info = saver.step((action_human, action_helper))
                timed_out = bool(info.get("timeout", False))
                if done:
                    break

            if not done or timed_out:
                continue

            record_path = os.path.join(record_dir, f"{episode_id}.json")
            saver.save(record_path)
            with open(record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            if not is_episode_suitable(
                record,
                temperature=args.infer_temperature,
                gamma=args.infer_gamma,
                action_cost=args.infer_action_cost,
                min_prob=args.infer_min_prob,
                min_pick_steps=args.min_pick_steps,
                min_put_steps=args.min_put_steps,
                min_high=args.min_high,
                max_low=args.max_low,
            ):
                os.remove(record_path)
                continue
            break

    _export_parquet(
        record_dir=record_dir,
        output_dir=args.output_dir,
        train_episodes=args.train_episodes,
        seed=args.seed,
        infer_end_time=args.infer_end_time,
        threshold=args.threshold,
        infer_temperature=args.infer_temperature,
        infer_gamma=args.infer_gamma,
        infer_action_cost=args.infer_action_cost,
        infer_min_prob=args.infer_min_prob,
        save_images=args.save_images,
    )


def _episode_spec(idx):
    # * placeholder for custom episode spec based on idx
    object_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "yellow",
        "brown",
    ]
    random.shuffle(object_colors)
    if idx % 40 <= 14:
        layout_id = 0
    else:
        layout_id = idx % 40 - 14
    all_shapes = ["circle", "square", "star"]
    object_shape = random.choice(all_shapes)
    object_shapes = [object_shape] * len(object_colors)
    return layout_id, object_colors, object_shapes


def _goal_candidates(env):
    candidates = []
    for i in range(env.num_blocks):
        for j in range(i + 1, env.num_blocks):
            if (
                env.get_dist(env.block_positions[i], env.block_positions[j]) > 6
                and env.get_dist(env.human_pos, env.block_positions[i]) > 6
                and env.get_dist(env.human_pos, env.block_positions[j]) > 6
            ):
                candidates.append((i, j))
    return candidates


def _export_parquet(
    record_dir,
    output_dir,
    train_episodes,
    seed,
    infer_end_time,
    threshold,
    infer_temperature,
    infer_gamma,
    infer_action_cost,
    infer_min_prob,
    save_images,
):
    rng = np.random.RandomState(seed)
    train_examples = []
    idx = 0
    for episode_id in range(1, train_episodes + 1):
        record_path = os.path.join(record_dir, f"{episode_id}.json")
        items, idx = build_examples2(
            record_path=record_path,
            output_dir=output_dir,
            idx=idx,
            seed=int(rng.randint(0, 1_000_000)),
            infer_end_time=infer_end_time,
            threshold=threshold,
            infer_temperature=infer_temperature,
            infer_gamma=infer_gamma,
            infer_action_cost=infer_action_cost,
            infer_min_prob=infer_min_prob,
            save_images=save_images,
        )
        train_examples.extend(items)

    Dataset.from_list(train_examples).to_parquet(
        os.path.join(output_dir, "train.parquet")
    )


if __name__ == "__main__":
    main()

