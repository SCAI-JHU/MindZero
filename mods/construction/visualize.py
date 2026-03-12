from PIL import Image

from env import ACTION_DELTAS, STAY


def save_gif_episode(
    env,
    human_agent,
    max_steps=100,
    action_helper=STAY,
    output_path="construction_demo.gif",
    render_kwargs=None,
    step_fn=None,
    pause_end=5,
):
    if render_kwargs is None:
        render_kwargs = {}
    if step_fn is None:
        step_fn = env.step

    obs = env._get_obs()
    frames = [Image.fromarray(env.render(mode="rgb_array", **render_kwargs))]

    for _ in range(max_steps):
        action_human = human_agent.sample_action(obs)
        actions = (action_human, action_helper)
        obs, done, info = step_fn(actions)
        frames.append(Image.fromarray(env.render(mode="rgb_array", **render_kwargs)))

        if done:
            for _ in range(pause_end):
                frames.append(Image.fromarray(env.render(mode="rgb_array", **render_kwargs)))
            break

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )
    print(f"Saved {output_path} ({len(frames)} frames)")


def save_ascii_episode(
    env,
    human_agent,
    max_steps=50,
    action_helper=STAY,
    output_path="construction_demo.txt",
    show_goal=True,
    step_fn=None,
):
    if step_fn is None:
        step_fn = env.step

    obs = env._get_obs()
    lines = [env.render(mode="ascii", show_goal=show_goal), ""]

    for _ in range(max_steps):
        action_human = human_agent.sample_action(obs)
        actions = (action_human, action_helper)
        obs, done, info = step_fn(actions)

        lines.append(f"--- After actions: Human={action_human}, Helper={action_helper} ---")
        lines.append(env.render(mode="ascii", show_goal=show_goal))
        lines.append("")

        if done:
            if info["goal_achieved"]:
                lines.append(f"✓ Goal achieved in {env.time_step} steps!")
            else:
                lines.append(f"✗ Timeout after {env.time_step} steps")
            break

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {output_path}")


def save_gif_from_actions(
    env,
    actions,
    output_path="construction_demo_replay.gif",
    render_kwargs=None,
    pause_end=5,
):
    if render_kwargs is None:
        render_kwargs = {}
    frames = [Image.fromarray(env.render(mode="rgb_array", **render_kwargs))]

    for action_pair in actions:
        decoded = _decode_action_pair(action_pair)
        obs, done, info = env.step(tuple(decoded))
        frames.append(Image.fromarray(env.render(mode="rgb_array", **render_kwargs)))
        if done:
            for _ in range(pause_end):
                frames.append(Image.fromarray(env.render(mode="rgb_array", **render_kwargs)))
            break

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )
    print(f"Saved {output_path} ({len(frames)} frames)")


def save_ascii_from_actions(
    env,
    actions,
    output_path="construction_demo_replay.txt",
    show_goal=True,
):
    lines = [env.render(mode="ascii", show_goal=show_goal), ""]

    for action_pair in actions:
        decoded = _decode_action_pair(action_pair)
        obs, done, info = env.step(tuple(decoded))
        lines.append(
            f"--- After actions: Human={decoded[0]}, Helper={decoded[1]} ---"
        )
        lines.append(env.render(mode="ascii", show_goal=show_goal))
        lines.append("")
        if done:
            if info["goal_achieved"]:
                lines.append(f"✓ Goal achieved in {env.time_step} steps!")
            else:
                lines.append(f"✗ Timeout after {env.time_step} steps")
            break

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {output_path}")


def _decode_action_pair(action_pair):
    if isinstance(action_pair[0], str):
        return action_pair
    delta_map = {tuple(v): k for k, v in ACTION_DELTAS.items()}
    human_delta = tuple(action_pair[0])
    helper_delta = tuple(action_pair[1])
    human_action = delta_map.get(human_delta, STAY)
    helper_action = delta_map.get(helper_delta, STAY)
    return [human_action, helper_action]

