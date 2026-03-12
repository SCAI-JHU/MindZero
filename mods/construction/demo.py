import json
import os

import numpy as np
from tqdm import tqdm

from agent import HumanAgent
from env import STAY, ConstructionEnv
from inference import ExactInference
from qa_export import export_dataset_from_records, is_episode_suitable
from recording import EnvSaver
from visualize import (
    save_ascii_from_actions,
    save_gif_episode,
    save_gif_from_actions,
)


def run_storage_demo(env_seed, agent_seed, agent_goal, layout_id):
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
    object_shapes = ["square"] * len(object_colors)
    env = ConstructionEnv(
        grid_size=(10, 10),
        max_steps=500,
        seed=env_seed,
        layout_id=layout_id,
        object_colors=object_colors,
        object_shapes=object_shapes,
    )
    env.reset(human_goal=agent_goal)

    human_agent = HumanAgent(
        env,
        goal=agent_goal,
        seed=agent_seed,
        temperature=0.01,
        gamma=0.95,
        action_cost=0.1,
    )
    inference = ExactInference(
        env,
        temperature=0.01,
        gamma=0.95,
        action_cost=0.1,
        print_each_step=True,
        min_prob=0.0001,
        track_pick_put=True,
    )
    inference.reset()

    saver = EnvSaver(env)
    saver.start()

    def step_fn(actions):
        return saver.step(actions, step_fn=inference.step)

    save_gif_episode(
        env,
        human_agent,
        max_steps=100,
        action_helper=STAY,
        output_path="construction_demo.gif",
        step_fn=step_fn,
    )
    saver.save("episode_record.json")

    replay = EnvSaver.load("episode_record.json")
    save_gif_from_actions(
        replay.env,
        replay.actions,
        output_path="construction_demo_replay.gif",
    )
    replay.env.set_state(replay.initial_state)
    save_ascii_from_actions(
        replay.env,
        replay.actions,
        output_path="construction_demo_replay.txt",
    )
    _generate_small_dataset(output_dir="data/sample", num_samples=10)


def _generate_small_dataset(output_dir, num_samples):
    record_dir = os.path.join(output_dir, "episode_record")
    os.makedirs(record_dir, exist_ok=True)

    rng = np.random.RandomState(0)
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
    object_shapes = ["square"] * len(object_colors)
    saved = 0
    attempts = 0
    with tqdm(total=num_samples, desc="Generating episodes") as pbar:
        while saved < num_samples:
            attempts += 1
            env_seed = int(rng.randint(0, 1_000_000))
            layout_id = int(rng.randint(0, 30))
            # layout_id = 0
            env = ConstructionEnv(
                grid_size=(10, 10),
                max_steps=100,
                seed=env_seed,
                layout_id=layout_id,
                object_colors=object_colors,
                object_shapes=object_shapes,
            )
            obs = env.reset(human_goal=None)
            goal_candidates = []
            for i in range(env.num_blocks):
                for j in range(i + 1, env.num_blocks):
                    if (
                        env.get_dist(env.block_positions[i], env.block_positions[j]) > 6
                        and env.get_dist(env.human_pos, env.block_positions[i]) > 6
                        and env.get_dist(env.human_pos, env.block_positions[j]) > 6
                    ):
                        goal_candidates.append((i, j))
            if not goal_candidates:
                continue
            env.human_goal = goal_candidates[int(rng.randint(0, len(goal_candidates)))]
            obs = env._get_obs()

            human_agent = HumanAgent(
                env,
                goal=env.human_goal,
                seed=int(rng.randint(0, 1_000_000)),
                temperature=0.01,
                gamma=0.95,
                action_cost=0.1,
            )

            saver = EnvSaver(env)
            saver.start()

            pick_index = None
            put_index = None
            done = False
            for step_idx in range(100):
                prev_holding = env.human_holding
                action_human = human_agent.sample_action(obs)
                action_helper = STAY
                obs, done, info = saver.step((action_human, action_helper))
                curr_holding = env.human_holding
                if (
                    pick_index is None
                    and prev_holding is None
                    and curr_holding is not None
                ):
                    pick_index = step_idx
                if (
                    pick_index is not None
                    and put_index is None
                    and prev_holding is not None
                    and curr_holding is None
                ):
                    put_index = step_idx
                if done:
                    break

            if not done or pick_index is None or put_index is None:
                continue

            record_path = os.path.join(record_dir, f"{saved}.json")
            saver.save(record_path)
            with open(record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            if not is_episode_suitable(
                record,
                temperature=0.1,
                gamma=0.95,
                action_cost=0.1,
                min_prob=0.0,
                min_pick_steps=6,
                min_put_steps=6,
                min_high=0.3,
                max_low=0.01,
            ):
                os.remove(record_path)
                continue
            saved += 1
            pbar.update(1)

    export_dataset_from_records(
        records_dir=record_dir,
        output_dir=output_dir,
        train_count=8,
        test_count=2,
        seed=0,
        min_pick_steps=6,
        min_put_steps=6,
        min_high=0.3,
        max_low=0.01,
    )


if __name__ == "__main__":
    run_storage_demo(env_seed=42, agent_seed=42, agent_goal=(0, 7), layout_id=0)
