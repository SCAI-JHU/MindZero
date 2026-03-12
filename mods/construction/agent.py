import numpy as np

from .env import MOVE_ACTIONS, PICK, PUT, STAY, TELEPORT
from .utils import boltzmann_policy, discounted_return, get_next_pos


class HumanAgent:
    def __init__(
        self,
        env,
        goal,
        seed=None,
        priority=True,
        temperature=0.5,
        gamma=0.95,
        action_cost=0.1,
        random_prob=0.0,
    ):
        self.env = env
        self.goal = goal
        self.width = env.width
        self.height = env.height
        self.rng = np.random.RandomState(seed)
        self.priority = priority
        self.temperature = temperature
        self.gamma = gamma
        self.action_cost = action_cost
        self.random_prob = random_prob
        self._hold_move_toggle = False

    def sample_action(self, obs):
        if getattr(self.env, "time_step", 0) == 0:
            self._hold_move_toggle = False
        if self._should_pick(obs):
            return PICK
        if self._should_put(obs):
            self._hold_move_toggle = False
            return PUT
        agent_info = obs["agents"][0 if self.priority else 1]
        holding = agent_info["holding"]
        if holding is not None and self._hold_move_toggle:
            self._hold_move_toggle = False
            return STAY
        if self.rng.random() < self.random_prob:
            action = self.rng.choice(MOVE_ACTIONS)
            self._hold_move_toggle = holding is not None
            return action
        q_values = self._compute_q(obs)
        action_dist = boltzmann_policy(q_values, self.temperature)
        action = self.rng.choice(list(action_dist.keys()), p=list(action_dist.values()))
        self._hold_move_toggle = holding is not None and action in MOVE_ACTIONS
        return action

    def _compute_q(self, obs):
        agent_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(agent_info["pos"])
        holding = agent_info["holding"]

        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}

        q_values = {}
        for action in MOVE_ACTIONS:
            next_pos = get_next_pos(pos, action, self.width, self.height)
            if next_pos == pos:
                continue

            if holding is None:
                v_next = self._compute_v_stage_1(next_pos, blocks)
            elif holding in self.goal:
                v_next = self._compute_v_stage_2(next_pos, holding, blocks)
            else:
                raise ValueError(f"Invalid holding: {holding}")

            q_values[action] = self.gamma * v_next - self.action_cost

        return q_values

    def _should_pick(self, obs):
        agent_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(agent_info["pos"])
        holding = agent_info["holding"]

        if holding is not None:
            return False

        for block in obs["blocks"]:
            if block["carried_by"] is None and tuple(block["pos"]) == pos:
                return block["id"] in self.goal
        return False

    def _should_put(self, obs):
        agent_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(agent_info["pos"])
        holding = agent_info["holding"]

        if holding is None:
            return False

        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}
        if any(
            b["carried_by"] is None and tuple(b["pos"]) == pos for b in obs["blocks"]
        ):
            return False

        target_a, target_b = self.goal
        if holding not in self.goal:
            return False
        other_target = target_b if holding == target_a else target_a
        target_pos = blocks[other_target]

        return self.env.min_dist_to_adjacent(pos, target_pos) == 0

    def _compute_v_stage_1(self, pos, blocks):
        target_a, target_b = self.goal

        dist_to_a = self.env.get_dist(pos, blocks[target_a])
        dist_to_b = self.env.get_dist(pos, blocks[target_b])

        if dist_to_a <= dist_to_b:
            nearest_target = blocks[target_a]
            other_target = blocks[target_b]
        else:
            nearest_target = blocks[target_b]
            other_target = blocks[target_a]

        dist_to_nearest = self.env.get_dist(pos, nearest_target)
        dist_between = self.env.get_dist(nearest_target, other_target)
        if dist_to_nearest == float("inf") or dist_between == float("inf"):
            total_dist = float("inf")
        else:
            total_dist = dist_to_nearest + max(0, dist_between - 1)

        return discounted_return(total_dist, self.gamma, self.action_cost)

    def _compute_v_stage_2(self, pos, holding, blocks):
        target_a, target_b = self.goal
        other_target = target_b if holding == target_a else target_a
        target_pos = blocks[other_target]

        min_dist = self.env.min_dist_to_adjacent(pos, target_pos)
        return discounted_return(min_dist, self.gamma, self.action_cost)


class HelperAgent:
    def __init__(
        self,
        env,
        seed=None,
        priority=False,
        temperature=0.5,
        gamma=0.95,
        action_cost=0.1,
        goal_mode="gt",
        bip_min_prob=1e-6,
        can_teleport=False,
        teleport_k=3,
    ):
        self.env = env
        self.width = env.width
        self.height = env.height
        self.rng = np.random.RandomState(seed)
        self.priority = priority
        self.temperature = temperature
        self.gamma = gamma
        self.action_cost = action_cost
        self.goal_mode = goal_mode
        self.bip_min_prob = bip_min_prob
        self._avoid_pick = set()
        self._stay_count = 0
        self.goal_distribution = None
        self.can_teleport = can_teleport
        self.teleport_k = teleport_k
        self._fixed_goal = None
        self._max_goal = None
        self._every_max_goal = None
        self._max_goal_count = 0

    def sample_action(self, obs, goal_dist=None):
        if getattr(self.env, "time_step", 0) == 0:
            self._avoid_pick.clear()
            self._stay_count = 0
            self._fixed_goal = None
            self._max_goal = None
            self._every_max_goal = None
            self._max_goal_count = 0
        helper_info = obs["agents"][0 if self.priority else 1]
        human_info = obs["agents"][1 if self.priority else 0]
        helper_pos = tuple(helper_info["pos"])
        human_pos = tuple(human_info["pos"])
        holding = helper_info["holding"]
        human_holding = human_info["holding"]
        if holding is not None and self.env.get_dist(helper_pos, human_pos) == 1:
            if not self._has_ground_object(obs, helper_pos):
                self._avoid_pick.add(holding)
                return PUT
        if holding is not None and self.env.get_dist(helper_pos, human_pos) == 2:
            return STAY
        updated_distribution = False
        action_history = obs.get("action_history") or obs.get("actions")
        last_human = action_history[-1][0] if action_history else None
        if self.goal_mode == "vlm":
            if goal_dist is None and self.goal_distribution is None:
                return STAY
            if last_human == STAY and self.goal_distribution is not None:
                goal_distribution = self.goal_distribution
            elif goal_dist is not None:
                self.goal_distribution = goal_dist
                goal_distribution = self.goal_distribution
                updated_distribution = True
            else:
                goal_distribution = self.goal_distribution
        else:
            skip_update = last_human == STAY
            if skip_update and self.goal_distribution is not None:
                goal_distribution = self.goal_distribution
            else:
                goal_distribution = self._get_goal_distribution(obs)
                self.goal_distribution = goal_distribution
                updated_distribution = True

        if self.can_teleport:
            max_goal = None
            if updated_distribution and goal_distribution:
                max_goal, _ = max(
                    goal_distribution, key=lambda item: (item[1], item[0])
                )
                self._every_max_goal = max_goal
            else:
                max_goal = self._max_goal
            if self._fixed_goal is None and updated_distribution:
                if max_goal is None:
                    self._max_goal = None
                    self._max_goal_count = 0
                elif max_goal == self._max_goal:
                    self._max_goal_count += 1
                else:
                    self._max_goal = max_goal
                    self._max_goal_count = 1 if max_goal is not None else 0
                if (
                    max_goal is not None
                    and human_holding in max_goal
                    and self._max_goal_count >= self.teleport_k
                ):
                    self._fixed_goal = max_goal
                    if holding is None:
                        other_obj = (
                            max_goal[1] if human_holding == max_goal[0] else max_goal[0]
                        )
                        return (TELEPORT, int(other_obj))
            if self._fixed_goal is not None:
                goal_distribution = [(self._fixed_goal, 1.0)]
        q_values = self._compute_q(obs, goal_distribution)
        if not q_values:
            return STAY
        action_dist = boltzmann_policy(q_values, self.temperature)
        action = self.rng.choice(list(action_dist.keys()), p=list(action_dist.values()))
        if action == STAY and self._stay_count >= 2:
            move_actions = [a for a in MOVE_ACTIONS if a in action_dist]
            if move_actions:
                action = self.rng.choice(move_actions)
        self._stay_count = self._stay_count + 1 if action == STAY else 0
        return action

    def teleport(self, goal_obj):
        assert 0 <= goal_obj < self.env.num_blocks, "goal_obj out of range"
        target_pos = tuple(self.env.block_positions[goal_obj])
        self.env.helper_pos = np.array(target_pos)
        if self.env.helper_holding is not None:
            self.env.block_positions[self.env.helper_holding] = (
                self.env.helper_pos.copy()
            )
        if hasattr(self.env, "helper_traj"):
            self.env.helper_traj.append(None)
            self.env.helper_traj.append(target_pos)

    def _get_goal_distribution(self, obs):
        if self.goal_mode == "gt":
            return self._goal_distribution_gt(obs)
        if self.goal_mode == "bip":
            return self._goal_distribution_bip(obs)
        return self._goal_distribution_uniform(obs)

    def _goal_distribution_gt(self, obs):
        if hasattr(self.env, "human_goal") and self.env.human_goal is not None:
            return [(tuple(self.env.human_goal), 1.0)]
        return self._goal_distribution_uniform(obs)

    def _goal_distribution_bip(self, obs):
        # * Expect obs to include action_history and initial_state for BIP replay.
        action_history = obs.get("action_history") or obs.get("actions")
        initial_state = obs.get("initial_state")
        if action_history is None or initial_state is None:
            return self._goal_distribution_uniform(obs)

        from .env import ConstructionEnv
        from .inference import ExactInference

        env_config = self.env.export_config()
        temp_env = ConstructionEnv(
            grid_size=tuple(env_config["grid_size"]),
            num_blocks=env_config["num_objects"],
            max_steps=env_config["max_steps"],
            blocked_positions=env_config["block_pos"],
            object_colors=env_config.get("object_colors"),
            object_shapes=env_config.get("object_shapes"),
        )
        temp_env.set_state(initial_state)
        inference = ExactInference(
            temp_env,
            temperature=self.temperature,
            gamma=self.gamma,
            action_cost=self.action_cost,
            min_prob=self.bip_min_prob,
            track_pick_put=False,
            print_each_step=False,
        )
        inference.reset()
        for action_pair in action_history:
            inference.step(tuple(action_pair))

        return list(inference.goal_distribution.probs.items())

    def _goal_distribution_uniform(self, obs):
        block_ids = [block["id"] for block in obs["blocks"]]
        goals = []
        for i in range(len(block_ids)):
            for j in range(i + 1, len(block_ids)):
                goals.append((block_ids[i], block_ids[j]))
        if not goals:
            return []
        prob = 1.0 / len(goals)
        return [(goal, prob) for goal in goals]

    def _normalize_goal_distribution(self, distribution):
        cleaned = []
        total = 0.0
        for goal, prob in distribution:
            if prob is None:
                continue
            prob = float(prob)
            if prob <= 0.0:
                continue
            goal = tuple(sorted(goal))
            cleaned.append((goal, prob))
            total += prob
        if total <= 0.0:
            return []
        return [(goal, prob / total) for goal, prob in cleaned]

    def _compute_q(self, obs, goal_distribution):
        total_prob = sum(prob for _, prob in goal_distribution)
        if total_prob <= 0:
            return {}
        q_values = {}
        for action in MOVE_ACTIONS + [STAY, PICK, PUT]:
            expected = 0.0
            for goal, prob in goal_distribution:
                expected += (prob / total_prob) * self._compute_action_value(
                    action, obs, goal
                )
            q_values[action] = expected
        return q_values

    def _compute_action_value(self, action, obs, goal):
        helper_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(helper_info["pos"])
        holding = helper_info["holding"]
        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}
        carriers = {b["id"]: b["carried_by"] for b in obs["blocks"]}

        helper_pick, helper_anchor = self._helper_targets(obs, goal)
        invalid_value = discounted_return(
            self.env.max_steps, self.gamma, self.action_cost
        )

        if helper_pick in self._avoid_pick:
            return invalid_value
        if holding is not None and holding != helper_pick:
            return invalid_value

        if action == PICK:
            if self._should_pick(obs, helper_pick):
                v_next = self._compute_v_stage_2(pos, helper_anchor, blocks)
                return self.gamma * v_next - self.action_cost
            return invalid_value

        if action == PUT:
            if self._should_put(obs, helper_pick, helper_anchor):
                return -self.action_cost
            return invalid_value

        if action == STAY:
            next_pos = pos
        else:
            next_pos = get_next_pos(pos, action, self.width, self.height)
            if next_pos == pos:
                return invalid_value

        if holding is None:
            if carriers.get(helper_pick) is not None:
                return invalid_value
            v_next = self._compute_v_stage_1(
                next_pos, helper_pick, helper_anchor, blocks
            )
        else:
            v_next = self._compute_v_stage_2(next_pos, helper_anchor, blocks)
        return self.gamma * v_next - self.action_cost

    def _helper_targets(self, obs, goal):
        target_a, target_b = goal
        human_info = obs["agents"][1 if self.priority else 0]
        human_pos = tuple(human_info["pos"])
        human_holding = human_info["holding"]
        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}

        if human_holding in goal:
            human_pick = human_holding
        else:
            dist_to_a = self.env.get_dist(human_pos, blocks[target_a])
            dist_to_b = self.env.get_dist(human_pos, blocks[target_b])
            human_pick = target_a if dist_to_a <= dist_to_b else target_b

        helper_pick = target_b if human_pick == target_a else target_a
        helper_anchor = target_a if helper_pick == target_b else target_b
        return helper_pick, helper_anchor

    def _should_pick(self, obs, helper_pick):
        helper_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(helper_info["pos"])
        holding = helper_info["holding"]
        if holding is not None:
            return False
        if helper_pick in self._avoid_pick:
            return False
        human_info = obs["agents"][1 if self.priority else 0]
        human_pos = tuple(human_info["pos"])
        human_holding = human_info["holding"]
        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}
        human_dist = self.env.get_dist(human_pos, blocks[helper_pick])
        for block in obs["blocks"]:
            if block["carried_by"] is None and tuple(block["pos"]) == pos:
                if block["id"] != helper_pick:
                    continue
                if human_holding is None and human_pos == pos:
                    return False
                if human_dist <= 3:
                    return False
                return True
        return False

    def _should_put(self, obs, helper_pick, helper_anchor):
        helper_info = obs["agents"][0 if self.priority else 1]
        pos = tuple(helper_info["pos"])
        holding = helper_info["holding"]
        if holding is None or holding != helper_pick:
            return False
        if any(
            b["carried_by"] is None and tuple(b["pos"]) == pos for b in obs["blocks"]
        ):
            return False
        blocks = {b["id"]: tuple(b["pos"]) for b in obs["blocks"]}
        target_pos = blocks[helper_anchor]
        return self.env.min_dist_to_adjacent(pos, target_pos) == 0

    def _compute_v_stage_1(self, pos, helper_pick, helper_anchor, blocks):
        dist_to_pick = self.env.get_dist(pos, blocks[helper_pick])
        dist_between = self.env.get_dist(blocks[helper_pick], blocks[helper_anchor])
        if dist_to_pick == float("inf") or dist_between == float("inf"):
            total_dist = self.env.max_steps
        else:
            total_dist = dist_to_pick + max(0, dist_between - 1)
        return discounted_return(total_dist, self.gamma, self.action_cost)

    def _compute_v_stage_2(self, pos, helper_anchor, blocks):
        target_pos = blocks[helper_anchor]
        min_dist = self.env.min_dist_to_adjacent(pos, target_pos)
        if min_dist == float("inf"):
            min_dist = self.env.max_steps
        return discounted_return(min_dist, self.gamma, self.action_cost)

    def _has_ground_object(self, obs, pos):
        for block in obs["blocks"]:
            if block["carried_by"] is None and tuple(block["pos"]) == pos:
                return True
        return False
