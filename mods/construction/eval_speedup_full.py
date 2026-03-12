import argparse
import base64
import csv
import io
import json
import os
import time

import requests
from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .agent import HelperAgent, HumanAgent
from .env import STAY, ConstructionEnv
from .estimate_distribution import (
    _object_labels_from_config,
    _parse_distribution2,
    _resolve_object_index,
)


def _parse_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _build_env(env_config, seed):
    return ConstructionEnv(
        grid_size=tuple(env_config["grid_size"]),
        num_blocks=env_config["num_objects"],
        max_steps=env_config["max_steps"],
        seed=seed,
        blocked_positions=env_config["block_pos"],
        object_colors=env_config.get("object_colors"),
        object_shapes=env_config.get("object_shapes"),
    )


def _simulate_episode(
    env_config,
    initial_state,
    mode,
    seed,
    goal_model=None,
    row=None,
    model_name=None,
    api_base=None,
    api_key=None,
    max_tokens=None,
    timeout=None,
    prompt_mode=None,
    openai_api_base=None,
    openai_api_key=None,
    gemini_api_base=None,
    gemini_api_key=None,
    gemini_model_name=None,
    gemini_reasoning_enabled=False,
):
    env = _build_env(env_config, seed)
    env.set_state(initial_state)
    goal = env.human_goal

    human_agent = HumanAgent(
        env,
        goal=goal,
        seed=seed,
        temperature=0.01,
        gamma=0.95,
        action_cost=0.1,
        random_prob=0.15,
    )

    helper_agent = None
    if mode in ("bip", "gt", "vlm", "openai", "gemini"):
        helper_agent = HelperAgent(
            env,
            seed=seed + 1 if seed is not None else None,
            temperature=0.01,
            gamma=0.95,
            action_cost=0.1,
            goal_mode=mode,
            can_teleport=True,
        )

    action_history = []
    responses = []
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    max_goal_matches = []
    initial_snapshot = env.get_state()
    gt_goal = tuple(initial_state.get("human_goal") or [])
    done = False
    for _ in range(env.max_steps):
        obs = env._get_obs()
        obs["initial_state"] = initial_snapshot
        obs["action_history"] = action_history
        last_human = action_history[-1][0] if action_history else None
        if mode == "bip":
            if last_human == STAY and helper_agent.goal_distribution is not None:
                distribution = helper_agent.goal_distribution
            else:
                distribution = helper_agent._get_goal_distribution(obs)
                helper_agent.goal_distribution = distribution
            responses.append(_format_bip_distribution(distribution, env_config))
            prompt_tokens.append(0)
            completion_tokens.append(0)
            total_tokens.append(0)
        if mode in ("vlm", "openai"):
            if row is None:
                raise ValueError("row is required when mode is vlm or openai")
            if last_human == STAY and helper_agent.goal_distribution is not None:
                distribution = helper_agent.goal_distribution
                responses.append(None)
                prompt_tokens.append(0)
                completion_tokens.append(0)
                total_tokens.append(0)
            else:
                image_bytes = _render_image_bytes(env)
                row_step = dict(row)
                row_step["images"] = [{"bytes": image_bytes}]
                row_step["actions"] = list(action_history)
                if mode == "openai":
                    api_base = openai_api_base
                    api_key = openai_api_key
                distribution, response, usage = _vlm_goal_distribution(
                    row_step,
                    env_config,
                    model_name,
                    api_base,
                    api_key,
                    max_tokens,
                    timeout,
                    prompt_mode,
                    use_responses=(mode == "openai"),
                )
                helper_agent.goal_distribution = distribution
                responses.append(response)
                prompt_tokens.append(int(usage.get("prompt_tokens") or 0))
                completion_tokens.append(int(usage.get("completion_tokens") or 0))
                total_tokens.append(
                    int(
                        usage.get("total_tokens")
                        or (prompt_tokens[-1] + completion_tokens[-1])
                    )
                )
            if distribution is None:
                print(response)
        if mode == "gemini":
            if row is None:
                raise ValueError("row is required when mode is gemini")
            if last_human == STAY and helper_agent.goal_distribution is not None:
                distribution = helper_agent.goal_distribution
                responses.append(None)
                prompt_tokens.append(0)
                completion_tokens.append(0)
                total_tokens.append(0)
            else:
                image_bytes = _render_image_bytes(env)
                row_step = dict(row)
                row_step["images"] = [{"bytes": image_bytes}]
                row_step["actions"] = list(action_history)
                distribution, response, usage = _gemini_goal_distribution(
                    row_step,
                    env_config,
                    gemini_model_name,
                    gemini_api_base,
                    gemini_api_key,
                    max_tokens,
                    timeout,
                    prompt_mode,
                    gemini_reasoning_enabled,
                )
                helper_agent.goal_distribution = distribution
                responses.append(response)
                prompt_tokens.append(int(usage.get("promptTokenCount") or 0))
                completion_tokens.append(int(usage.get("candidatesTokenCount") or 0))
                total_tokens.append(int(usage.get("totalTokenCount") or 0))
            if distribution is None:
                print(response)

        action_human = human_agent.sample_action(obs)
        if helper_agent is None:
            action_helper = STAY
        else:
            if mode in ("vlm", "openai", "gemini"):
                action_helper = helper_agent.sample_action(obs, distribution)
            else:
                action_helper = helper_agent.sample_action(obs)
        if helper_agent is not None:
            max_goal = helper_agent._every_max_goal
            if max_goal is not None and len(gt_goal) == 2:
                gt_goal_resolved = gt_goal
                if any(isinstance(item, str) for item in gt_goal):
                    object_labels = _object_labels_from_config(env_config)
                    object_colors = env_config["object_colors"]
                    gt_goal_resolved = (
                        _resolve_object_index(gt_goal[0], object_labels, object_colors),
                        _resolve_object_index(gt_goal[1], object_labels, object_colors),
                    )
                match = (
                    1
                    if tuple(sorted(max_goal)) == tuple(sorted(gt_goal_resolved))
                    else 0
                )
                max_goal_matches.append(match)
        _, done, info = env.step((action_human, action_helper))
        action_history.append([action_human, action_helper])
        if done:
            break
    return (
        action_history,
        int(env.time_step),
        bool(done),
        responses,
        max_goal_matches,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def _render_frames(env, actions, pause_end=5):
    frames = [Image.fromarray(env.render(mode="rgb_array"))]
    for action_pair in actions:
        _, done, info = env.step(tuple(action_pair))
        frames.append(Image.fromarray(env.render(mode="rgb_array")))
        if done:
            for _ in range(pause_end):
                frames.append(Image.fromarray(env.render(mode="rgb_array")))
            break
    return frames


def _load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _annotate_frames(frames, label_text, header_text, top_gap=24, bottom_gap=28, font_size=18):
    font = _load_font(font_size)
    font_header = _load_font(font_size)
    annotated = []
    for frame in frames:
        width, height = frame.size
        canvas = Image.new("RGB", (width, height + top_gap + bottom_gap), (255, 255, 255))
        canvas.paste(frame, (0, top_gap))
        draw = ImageDraw.Draw(canvas)

        header_bbox = draw.textbbox((0, 0), header_text, font=font_header)
        header_w = header_bbox[2] - header_bbox[0]
        draw.text(
            ((width - header_w) / 2, (top_gap - font_size) / 2),
            header_text,
            fill=(0, 0, 0),
            font=font_header,
        )

        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        text_y = height + top_gap + (bottom_gap - font_size) / 2
        draw.text(((width - label_w) / 2, text_y), label_text, fill=(0, 0, 0), font=font)
        annotated.append(canvas)
    return annotated


def _render_image_bytes(env):
    image = Image.fromarray(env.render(mode="rgb_array"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _encode_image_bytes(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _encode_image_bytes_raw(image_bytes):
    return base64.b64encode(image_bytes).decode("ascii")


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


def _build_openai_input(images, problem_text):
    content = []
    for item in images:
        image_bytes = item["bytes"] if isinstance(item, dict) else item
        content.append(
            {
                "type": "input_image",
                "image_url": _encode_image_bytes(image_bytes),
            }
        )
    content.append({"type": "input_text", "text": problem_text})
    return [{"role": "user", "content": content}]


def _build_prompt(problem_text, prompt_mode):
    if prompt_mode == "normal":
        return (
            f"{problem_text}\n\n"
            "Please output the minified JSON."
        )
    elif prompt_mode == "distance":
        return (
            f"{problem_text}\n\n"
            "Note that the Human (green robot) consistently prioritizes picking up the object closest to its initial starting position first, subsequently placing it next to the object that was initially further away. In your JSON response, ensure that for every GoalParticle, object1 is strictly the object closer to the Human (green robot)'s starting position, and object2 is the object further from it.\n"
            "Please output the minified JSON."
        )
    else:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}")


def _build_gemini_contents(images, prompt_text):
    parts = []
    for item in images:
        image_bytes = item["bytes"] if isinstance(item, dict) else item
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": _encode_image_bytes_raw(image_bytes),
                }
            }
        )
    parts.append({"text": prompt_text})
    return [{"role": "user", "parts": parts}]


def _uniform_goal_distribution(obs):
    block_ids = [block["id"] for block in obs["blocks"]]
    goals = []
    for i in range(len(block_ids)):
        for j in range(i + 1, len(block_ids)):
            goals.append((block_ids[i], block_ids[j]))
    if not goals:
        return []
    prob = 1.0 / len(goals)
    return [(goal, prob) for goal in goals]


def _format_bip_distribution(distribution, env_config, min_prob=0.01):
    object_labels = _object_labels_from_config(env_config)
    particles = []
    for goal, prob in distribution:
        if prob < min_prob:
            continue
        obj_a, obj_b = goal
        if obj_a >= len(object_labels) or obj_b >= len(object_labels):
            continue
        particles.append(
            {
                "object1": object_labels[obj_a],
                "object2": object_labels[obj_b],
                "p": float(prob),
            }
        )
    particles = sorted(particles, key=lambda item: item["p"], reverse=True)
    return json.dumps({"particles": particles}, ensure_ascii=True, indent=2)


def _vlm_goal_distribution(
    row,
    env_config,
    model_name,
    api_base,
    api_key,
    max_tokens,
    timeout,
    prompt_mode,
    use_responses=False,
):
    if use_responses:
        url = f"{api_base.rstrip('/')}/responses"
    else:
        url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    prompt_text = _build_prompt(row["problem"], prompt_mode)
    if use_responses:
        input_payload = _build_openai_input(row["images"], prompt_text)
        payload = {
            "model": model_name,
            "input": input_payload,
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
        }
    else:
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
    usage = data.get("usage") or {}
    if use_responses:
        response = data.get("output_text")
        if not response:
            response = ""
            for output in data.get("output", []):
                for item in output.get("content", []):
                    if item.get("type") == "output_text":
                        response = item.get("text", "")
                        break
                if response:
                    break
    else:
        response = data["choices"][0]["message"]["content"]
    dist = _parse_distribution2(response, env_config)
    if not dist:
        return None, response, usage

    object_labels = _object_labels_from_config(env_config)
    object_colors = env_config["object_colors"]
    distribution = []
    for particle in dist.particles:
        label_a = particle.object1.label()
        label_b = particle.object2.label()
        obj_a = _resolve_object_index(label_a, object_labels, object_colors)
        obj_b = _resolve_object_index(label_b, object_labels, object_colors)
        distribution.append(((obj_a, obj_b), float(particle.p)))
    return distribution, response, usage


def _gemini_goal_distribution(
    row,
    env_config,
    model_name,
    api_base,
    api_key,
    max_tokens,
    timeout,
    prompt_mode,
    reasoning_enabled,
):
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    prompt_text = _build_prompt(row["problem"], prompt_mode)
    messages = _build_messages(row["images"], prompt_text)
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "extra_body": {"reasoning": {"enabled": bool(reasoning_enabled)}},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    response = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    dist = _parse_distribution2(response, env_config)
    if not dist:
        return None, response, usage

    object_labels = _object_labels_from_config(env_config)
    object_colors = env_config["object_colors"]
    distribution = []
    for particle in dist.particles:
        label_a = particle.object1.label()
        label_b = particle.object2.label()
        obj_a = _resolve_object_index(label_a, object_labels, object_colors)
        obj_b = _resolve_object_index(label_b, object_labels, object_colors)
        distribution.append(((obj_a, obj_b), float(particle.p)))
    return distribution, response, usage


def _load_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv_rows(csv_path, rows, fieldnames):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_speedup_from_csv(csv_path):
    rows = _load_csv_rows(csv_path)
    if not rows:
        print(f"No csv found at {csv_path}")
        return
    fieldnames = rows[0].keys()
    if "no" not in fieldnames:
        print("Missing baseline column 'no' in csv")
        return
    model_cols = [col for col in fieldnames if col not in ("episode_id", "no")]
    for model in model_cols:
        speedups = []
        for row in rows:
            try:
                solo_steps = int(float(row["no"]))
                model_steps = int(float(row[model]))
            except (TypeError, ValueError):
                continue
            if solo_steps > 0 and model_steps > 0:
                speedups.append(solo_steps / model_steps - 1)
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        print(f"Average speedup ({model}): {avg_speedup:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path", default="/home/ubuntu/icml/yichao/EasyR1/lambda_pack/data/gw_0125_hiyouga/asst/eval.parquet"
    )
    parser.add_argument(
        "--model", choices=["no", "bip", "gt", "vlm", "openai", "gemini"], default="bip"
    )
    parser.add_argument("--model_name")
    parser.add_argument("--prompt_mode", choices=["normal", "distance"], default="normal")
    parser.add_argument("--api_base", default="http://localhost:9991/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--openai_api_base", default="https://api.openai.com/v1")
    parser.add_argument(
        "--openai_api_key", default=os.environ.get("OPENAI_API_KEY", "")
    )
    parser.add_argument(
        "--gemini_api_base", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument("--gemini_api_key", default="[[HIDDEN]]")
    parser.add_argument(
        "--gemini_reasoning",
        action="store_true",
        help="Enable OpenRouter extra_body reasoning for gemini",
    )
    # model_name is shared across vlm/openai/gemini
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--output_dir", default="/home/ubuntu/icml/yichao/EasyR1/lambda_pack/data/gw_0125_hiyouga/asst/asst_eval"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute_speedup", action="store_true")
    args = parser.parse_args()

    args.output_dir = f'{args.output_dir}_{args.seed}'
    csv_path = os.path.join(args.output_dir, "eval_summary.csv")
    if args.compute_speedup:
        _compute_speedup_from_csv(csv_path)
        return
    if args.model in ("vlm", "openai", "gemini") and not args.model_name:
        if args.model == "openai":
            default_name = "gpt-5.2"
        elif args.model == "gemini":
            default_name = "google/gemini-3-flash-preview"
        else:
            default_name = None
        args.model_name = default_name
    if args.model == "vlm" and not args.model_name:
        raise ValueError("model_name is required when --model=vlm")
    if args.model == "openai" and not args.openai_api_key:
        raise ValueError("openai_api_key is required when --model=openai")
    if args.model == "gemini" and not args.gemini_api_key:
        raise ValueError("gemini_api_key is required when --model=gemini")

    dataset = Dataset.from_parquet(args.parquet_path)
    episodes = [row for row in dataset]

    if args.model in ("vlm", "openai", "gemini"):
        model_dir = os.path.join(args.output_dir, args.model_name)
    else:
        model_dir = os.path.join(args.output_dir, args.model)
    gifs_dir = os.path.join(model_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    results_path = os.path.join(model_dir, "results.json")
    results = []
    progress = tqdm(enumerate(episodes), total=len(episodes), desc="Evaluating")
    for row_idx, row in progress:
        env_config = _parse_json(row["env_config"])
        initial_state = _parse_json(row["initial_state"])

        start_time = time.time()
        (
            model_actions,
            model_steps,
            model_done,
            responses,
            max_goal_matches,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ) = _simulate_episode(
            env_config,
            initial_state,
            args.model,
            args.seed,
            row=row,
            model_name=args.model_name,
            api_base=args.api_base,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            prompt_mode=args.prompt_mode,
            openai_api_base=args.openai_api_base,
            openai_api_key=args.openai_api_key,
            gemini_api_base=args.gemini_api_base,
            gemini_api_key=args.gemini_api_key,
            gemini_model_name=args.model_name,
            gemini_reasoning_enabled=args.gemini_reasoning,
        )
        elapsed_seconds = time.time() - start_time
        progress.set_postfix(steps=model_steps)

        env_model = _build_env(env_config, args.seed)
        env_model.set_state(initial_state)
        frames_model = _render_frames(env_model, model_actions)
        header_text = f"idx={row_idx}"
        if args.model in ("vlm", "openai", "gemini"):
            label_text = args.model_name
        else:
            label_text = "No help" if args.model == "no" else args.model.upper()
        frames_model = _annotate_frames(frames_model, label_text, header_text)

        output_path = os.path.join(gifs_dir, f"{row_idx}.gif")
        frames_model[0].save(
            output_path,
            save_all=True,
            append_images=frames_model[1:],
            duration=300,
            loop=0,
        )
        print(f"Saved {output_path} ({len(frames_model)} frames)")

        results.append(
            {
                "episode_id": int(row_idx),
                "env_config": row["env_config"],
                "initial_state": row["initial_state"],
                "actions": model_actions,
                "responses": responses,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_seconds": float(elapsed_seconds),
                "max_goal_matches_gt": max_goal_matches,
                "steps": int(model_steps),
                "done": bool(model_done),
            }
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=True, indent=2)

    existing_rows = _load_csv_rows(csv_path)
    if existing_rows and len(existing_rows) != len(episodes):
        raise ValueError("CSV row count does not match eval parquet rows")

    if existing_rows:
        rows = existing_rows
        fieldnames = list(rows[0].keys())
        if args.model in ("vlm", "openai", "gemini"):
            if args.model_name not in fieldnames:
                fieldnames.append(args.model_name)
        else:
            if args.model not in fieldnames:
                fieldnames.append(args.model)
    else:
        rows = [{"episode_id": str(i)} for i in range(len(episodes))]
        if args.model in ("vlm", "openai", "gemini"):
            fieldnames = ["episode_id", args.model_name]
        else:
            fieldnames = ["episode_id", args.model]

    for i, row in enumerate(rows):
        if args.model in ("vlm", "openai", "gemini"):
            row[args.model_name] = str(results[i]["steps"])
        else:
            row[args.model] = str(results[i]["steps"])

    _write_csv_rows(csv_path, rows, fieldnames)


if __name__ == "__main__":
    main()
