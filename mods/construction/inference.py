import itertools
import math

from .agent import HumanAgent
from .env import PICK, PUT
from .utils import boltzmann_policy


class GoalDistribution:
    def __init__(self, goals, min_prob=0.0):
        self.goals = list(goals)
        self.min_prob = min_prob
        self.probs = {goal: 1.0 / len(self.goals) for goal in self.goals}

    def reset_uniform(self):
        uniform = 1.0 / len(self.goals)
        self.probs = {goal: uniform for goal in self.goals}

    def update(self, likelihoods):
        for goal in self.goals:
            self.probs[goal] *= likelihoods.get(goal, 0.0)
        if self.min_prob > 0.0:
            for goal in self.goals:
                self.probs[goal] = max(self.probs[goal], self.min_prob)
        self._normalize()

    def _normalize(self):
        total = sum(self.probs.values())
        if total == 0.0:
            self.reset_uniform()
            return
        for goal in self.goals:
            self.probs[goal] /= total

    def format(self, label_fn=None):
        ordered = sorted(self.goals, key=lambda g: self.probs[g], reverse=True)
        parts = []
        for goal in ordered:
            label = label_fn(goal) if label_fn is not None else goal
            parts.append(f"{label}:{self.probs[goal]:.3f}")
        return " | ".join(parts)


class ExactInference:
    def __init__(
        self,
        env,
        temperature=0.5,
        gamma=0.95,
        action_cost=0.1,
        min_prob=0.0,
        track_pick_put=False,
        print_each_step=False,
    ):
        self.env = env
        self.temperature = temperature
        self.gamma = gamma
        self.action_cost = action_cost
        self.track_pick_put = track_pick_put
        self.print_each_step = print_each_step
        self.goal_distribution = GoalDistribution(self._all_goals(), min_prob=min_prob)
        self.pick_distribution = None
        self.put_distribution = None
        if self.track_pick_put:
            self.pick_distribution = GoalDistribution(self._all_objects(), min_prob=0.0)
            self.put_distribution = GoalDistribution(self._all_objects(), min_prob=0.0)

        self._agents = {
            goal: HumanAgent(
                env,
                goal=goal,
                seed=None,
                priority=True,
                temperature=temperature,
                gamma=gamma,
                action_cost=action_cost,
            )
            for goal in self.goal_distribution.goals
        }

    def reset(self):
        self.goal_distribution.reset_uniform()
        if self.track_pick_put:
            self._sync_pick_put(self.env._get_obs())

    def step(self, actions):
        obs = self.env._get_obs()
        action_human, _ = actions
        likelihoods = {
            goal: self._action_likelihood(agent, obs, action_human)
            for goal, agent in self._agents.items()
        }
        self.goal_distribution.update(likelihoods)

        obs, done, info = self.env.step(actions)
        if self.track_pick_put:
            self._sync_pick_put(obs)
        if self.print_each_step:
            print(
                f"[t={self.env.time_step}] goal_dist: {self.goal_distribution.format()}"
            )
            if self.track_pick_put:
                pick_desc = self.pick_distribution.format(self._object_label)
                put_desc = self.put_distribution.format(self._object_label)
                print(f"    pick_goal: {pick_desc}")
                print(f"    put_goal: {put_desc}")
        return obs, done, info

    def _all_goals(self):
        return list(itertools.combinations(range(self.env.num_blocks), 2))

    def _all_objects(self):
        return list(range(self.env.num_blocks))

    def _object_label(self, obj_id):
        if hasattr(self.env, "object_labels"):
            return self.env.object_labels[int(obj_id)]
        return str(obj_id)

    def _set_distribution(self, distribution, weights):
        total = sum(weights.values())
        if total == 0.0:
            distribution.reset_uniform()
            return
        for goal in distribution.goals:
            distribution.probs[goal] = weights.get(goal, 0.0) / total

    def _sync_pick_put(self, obs):
        holding = obs["agents"][0]["holding"]
        if holding is not None:
            self.pick_distribution.probs = {
                obj: 1.0 if obj == holding else 0.0
                for obj in self.pick_distribution.goals
            }
            weights = {obj: 0.0 for obj in self.put_distribution.goals}
            total = 0.0
            for (a, b), prob in self.goal_distribution.probs.items():
                if holding == a:
                    weights[b] += prob
                    total += prob
                elif holding == b:
                    weights[a] += prob
                    total += prob
            if total == 0.0:
                candidates = [obj for obj in weights if obj != holding]
                uniform = 1.0 / len(candidates) if candidates else 0.0
                for obj in weights:
                    weights[obj] = uniform if obj != holding else 0.0
            self._set_distribution(self.put_distribution, weights)
            return

        block_pos = {b["id"]: b["pos"] for b in obs["blocks"]}
        carried_by = {b["id"]: b["carried_by"] for b in obs["blocks"]}
        human_pos = obs["agents"][0]["pos"]

        pick_weights = {obj: 0.0 for obj in self.pick_distribution.goals}
        put_weights = {obj: 0.0 for obj in self.put_distribution.goals}
        for (a, b), prob in self.goal_distribution.probs.items():
            if carried_by.get(a) is not None:
                dist_a = float("inf")
            else:
                dist_a = self.env.get_dist(human_pos, block_pos[a])
            if carried_by.get(b) is not None:
                dist_b = float("inf")
            else:
                dist_b = self.env.get_dist(human_pos, block_pos[b])

            temp = self.temperature if self.temperature > 0.0 else 1e-6
            weight_a = 0.0 if dist_a == float("inf") else math.exp(-dist_a / temp)
            weight_b = 0.0 if dist_b == float("inf") else math.exp(-dist_b / temp)
            denom = weight_a + weight_b
            if denom == 0.0:
                pick_a = 0.5
                pick_b = 0.5
            else:
                pick_a = weight_a / denom
                pick_b = weight_b / denom

            pick_weights[a] += prob * pick_a
            pick_weights[b] += prob * pick_b
            put_weights[b] += prob * pick_a
            put_weights[a] += prob * pick_b

        self._set_distribution(self.pick_distribution, pick_weights)
        self._set_distribution(self.put_distribution, put_weights)

    def _action_likelihood(self, agent, obs, action_human):
        agent_info = obs["agents"][0]
        holding = agent_info["holding"]
        if holding is not None and holding not in agent.goal:
            return 0.0

        if agent._should_pick(obs):
            return 1.0 if action_human == PICK else 0.0
        if agent._should_put(obs):
            return 1.0 if action_human == PUT else 0.0

        q_values = agent._compute_q(obs)
        if not q_values:
            return 0.0
        action_dist = boltzmann_policy(q_values, self.temperature)
        return action_dist.get(action_human, 0.0)

