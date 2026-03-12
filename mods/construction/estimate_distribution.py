import argparse
import json
import math
import re

from datasets import Dataset
from pydantic import BaseModel, Field, field_validator, model_validator

from .agent import HumanAgent
from .env import PICK, PUT, ConstructionEnv
from .utils import boltzmann_policy


class Object(BaseModel):
    color: str
    shape: str
    _valid_labels = None

    @model_validator(mode="before")
    def _normalize_fields(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Expected object with keys: color, shape.")
        expected = {"color", "shape"}
        extra = set(values.keys()) - expected
        missing = expected - set(values.keys())
        if extra or missing:
            raise ValueError("Expected keys: color, shape.")
        return values

    @model_validator(mode="after")
    def _validate_in_env(self):
        if self.__class__._valid_labels is not None:
            label = self.label()
            assert label in self.__class__._valid_labels, f"Unknown object label: {label}"
        return self

    def label(self):
        return f"{self.color} {self.shape}"


class GoalParticle(BaseModel):
    object1: Object
    object2: Object
    p: float = Field(..., ge=0, le=1, description="Probability of the goal proposal")

    @model_validator(mode="before")
    def _normalize_fields(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Expected object with keys: object1, object2, p.")
        expected = {"object1", "object2", "p"}
        extra = set(values.keys()) - expected
        missing = expected - set(values.keys())
        if extra or missing:
            raise ValueError("Expected keys: object1, object2, p.")
        for key in ("object1", "object2"):
            obj = values.get(key)
            if not isinstance(obj, dict):
                raise ValueError("object1/object2 must be objects with color and shape.")
            obj_expected = {"color", "shape"}
            obj_extra = set(obj.keys()) - obj_expected
            obj_missing = obj_expected - set(obj.keys())
            if obj_extra or obj_missing:
                raise ValueError("object1/object2 must only have color and shape.")
        return values

    @model_validator(mode="after")
    def _validate_distinct_objects(self):
        if self.object1.label() == self.object2.label():
            raise ValueError("object1 and object2 must be different objects.")
        return self


class GoalParticles(BaseModel):
    particles: list[GoalParticle]

    @model_validator(mode="before")
    def _normalize_list(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Expected object with key: particles.")
        expected = {"particles"}
        extra = set(values.keys()) - expected
        missing = expected - set(values.keys())
        if extra or missing:
            raise ValueError("Expected object with key: particles.")
        return values

    @field_validator("particles")
    def _ensure_len(cls, value):
        if len(value) != 8:
            raise ValueError("Expected exactly 8 goal particles.")
        seen = set()
        for particle in value:
            labels = sorted([particle.object1.label(), particle.object2.label()])
            key = tuple(labels)
            if key in seen:
                raise ValueError(f"Duplicate goal particle pair: {key}")
            seen.add(key)
        return value

    def normalize(self, min_prob=1e-12):
        for particle in self.particles:
            particle.p = max(particle.p, min_prob)
        partition = sum(particle.p for particle in self.particles)
        if partition <= 0:
            return
        for particle in self.particles:
            particle.p /= partition

    @classmethod
    def from_json_with_env(cls, data, env_config):
        object_labels = set(_object_labels_from_config(env_config))
        Object._valid_labels = object_labels
        dist = cls.model_validate(data)
        dist.normalize()
        Object._valid_labels = None
        return dist

    @classmethod
    def from_json_with_env_lenient(cls, data, env_config):
        object_labels = set(_object_labels_from_config(env_config))
        label_map = {}
        for label in object_labels:
            parts = label.split(" ")
            if len(parts) >= 2:
                label_map[label] = (parts[0], parts[1])

        particles = data.get("particles") if isinstance(data, dict) else None
        if not isinstance(particles, list):
            return None

        merged = {}
        Object._valid_labels = object_labels
        try:
            for item in particles:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate = GoalParticle.model_validate(item)
                except Exception:
                    continue
                label_a = candidate.object1.label()
                label_b = candidate.object2.label()
                if label_a not in object_labels or label_b not in object_labels:
                    continue
                prob = float(candidate.p)
                if prob < 0.0:
                    continue
                key = tuple(sorted([label_a, label_b]))
                merged[key] = merged.get(key, 0.0) + prob
        finally:
            Object._valid_labels = None

        if not merged:
            return None

        items = []
        for label_a, label_b in merged:
            if label_a not in label_map or label_b not in label_map:
                continue
            color_a, shape_a = label_map[label_a]
            color_b, shape_b = label_map[label_b]
            items.append(
                {
                    "object1": {"color": color_a, "shape": shape_a},
                    "object2": {"color": color_b, "shape": shape_b},
                    "p": merged[(label_a, label_b)],
                }
            )
        if not items:
            return None

        Object._valid_labels = object_labels
        particles = [GoalParticle.model_validate(item) for item in items]
        dist = cls.model_construct(particles=particles)
        dist.normalize()
        Object._valid_labels = None
        return dist


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


def estimate_goal(
    env,
    goal_pair,
    actions,
    temperature=0.01,
    gamma=0.95,
    action_cost=0.1,
    min_prob=1e-12,
):
    goal = tuple(sorted([int(goal_pair[0]), int(goal_pair[1])]))
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
    return env_config, initial_state, actions


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


def _extract_json(text):
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return None


def _parse_distribution(response, env_config):
    try:
        data = json.loads(response)
    except Exception:
        payload = _extract_json(response)
        if payload is None:
            return None
        try:
            data = json.loads(payload)
        except Exception:
            return None
    try:
        return GoalParticles.from_json_with_env(data, env_config)
    except Exception:
        return None


def _parse_distribution2(response, env_config):
    try:
        data = json.loads(response)
    except Exception:
        payload = _extract_json(response)
        if payload is None:
            return None
        try:
            data = json.loads(payload)
        except Exception:
            return None
    try:
        return GoalParticles.from_json_with_env_lenient(data, env_config)
    except Exception:
        return None


def compute_score(
    reward_inputs,
    key="neg_kl_backward",
    entropy_bonus=None,
    temperature=0.01,
    gamma=0.95,
    action_cost=0.1,
    min_prob=1e-12,
    distance=False,
):
    def safe_log(value):
        return math.log(max(value, min_prob))

    scores = []
    for reward_input in reward_inputs:
        env_config = reward_input["env_config"]
        if isinstance(env_config, str):
            env_config = json.loads(env_config)
        response = reward_input["response"]
        dist = _parse_distribution(response, env_config)
        initial_state = reward_input["initial_state"]
        if isinstance(initial_state, str):
            initial_state = json.loads(initial_state)
        actions = reward_input["actions"]
        if isinstance(actions, str):
            actions = json.loads(actions)

        object_labels = _object_labels_from_config(env_config)
        human_goal = initial_state.get("human_goal")
        if human_goal is not None and len(human_goal) == 2:
            if all(isinstance(item, int) for item in human_goal):
                obj_a, obj_b = human_goal
            else:
                label_a, label_b = human_goal
                obj_a = _resolve_object_index(
                    label_a, object_labels, env_config["object_colors"]
                )
                obj_b = _resolve_object_index(
                    label_b, object_labels, env_config["object_colors"]
                )
            env = _build_env(env_config)
            env.set_state(initial_state)

        if not dist:
            scores.append(
                {
                    "parse_ok": 0.0,
                    "num_candidates": 0,
                    "details": [],
                    "overall": float(math.log(min_prob)),
                    "n": 0,
                    "ent_p": 0.0,
                    "ent_q": 0.0,
                    "cross_ent_forward": 0.0,
                    "cross_ent_backward": 0.0,
                    "neg_kl_forward": -100.0,
                    "neg_kl_backward": -100.0,
                    "avg_p": 0.0,
                }
            )
            continue

        ps = []
        qs = []
        # details = []
        bad_distance = False
        for particle in dist.particles:
            label_a = particle.object1.label()
            label_b = particle.object2.label()
            obj_a = _resolve_object_index(
                label_a, object_labels, env_config["object_colors"]
            )
            obj_b = _resolve_object_index(
                label_b, object_labels, env_config["object_colors"]
            )
            if distance:
                dist_a = env.get_human_dist_to_block(obj_a)
                dist_b = env.get_human_dist_to_block(obj_b)
                if dist_a > dist_b:
                    bad_distance = True
                    break
            env = _build_env(env_config)
            env.set_state(initial_state)
            log_probs = estimate_goal(
                env,
                (obj_a, obj_b),
                actions,
                temperature=temperature,
                gamma=gamma,
                action_cost=action_cost,
                min_prob=min_prob,
            )
            log_sum = float(sum(log_probs))
            ps.append(math.exp(log_sum))
            qs.append(particle.p)
            # details.append(
            #     {
            #         "objects": [label_a, label_b],
            #         "p": particle.p,
            #         "sum": log_sum,
            #         "last": log_probs[-1],
            #         "all_log_probs": log_probs,
            #     }
            # )
        
        if distance and bad_distance:
            scores.append(
                {
                    "parse_ok": 0.0,
                    "num_candidates": 0,
                    "details": [],
                    "overall": float(math.log(min_prob)),
                    "n": 0,
                    "ent_p": 0.0,
                    "ent_q": 0.0,
                    "cross_ent_forward": 0.0,
                    "cross_ent_backward": 0.0,
                    "neg_kl_forward": -100.0,
                    "neg_kl_backward": -100.0,
                    "avg_p": 0.0,
                }
            )
            continue

        n = len(ps)
        ent_p = -sum(p * safe_log(p) for p in ps)
        ent_q = -sum(q * safe_log(q) for q in qs)
        cross_ent_forward = -sum(p * safe_log(q) for p, q in zip(ps, qs))
        cross_ent_backward = -sum(q * safe_log(p) for p, q in zip(ps, qs))
        kl_forward = cross_ent_forward - ent_p
        kl_backward = cross_ent_backward - ent_q
        avg_p = sum(ps) / max(1, n)

        if key in ("neg_kl_forward", "neg_kl_backward"):
            overall = -kl_forward if key == "neg_kl_forward" else -kl_backward
        elif key == "neg_kl_backward_entropy_bonus":
            if entropy_bonus is None:
                entropy_bonus = 0.0
            overall = -kl_backward + entropy_bonus * ent_q
        else:
            raise ValueError(f"Invalid key: {key}")

        scores.append(
            {
                "overall": overall,
                "parse_ok": 1.0,
                "num_candidates": len(dist.particles),
                # "details": details,
                "n": n,
                "ent_p": ent_p,
                "ent_q": ent_q,
                "cross_ent_forward": cross_ent_forward,
                "cross_ent_backward": cross_ent_backward,
                "neg_kl_forward": -kl_forward,
                "neg_kl_backward": -kl_backward,
                "avg_p": avg_p,
            }
        )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--row_idx", type=int, required=True)
    parser.add_argument("--response", type=str, default=None)
    parser.add_argument("--response_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--action_cost", type=float, default=0.1)
    parser.add_argument("--min_prob", type=float, default=1e-12)
    args = parser.parse_args()

    response = args.response
    if response is None and args.response_path:
        with open(args.response_path, "r", encoding="utf-8") as f:
            response = f.read()
    assert response is not None, "Provide --response or --response_path."

    env_config, initial_state, actions = _load_row(args.parquet_path, args.row_idx)
    score = compute_score(
        [
            {
                "response": response,
                "env_config": env_config,
                "initial_state": initial_state,
                "actions": actions,
            }
        ],
        temperature=args.temperature,
        gamma=args.gamma,
        action_cost=args.action_cost,
        min_prob=args.min_prob,
    )[0]
    print(json.dumps(score, ensure_ascii=True))


if __name__ == "__main__":
    main()

