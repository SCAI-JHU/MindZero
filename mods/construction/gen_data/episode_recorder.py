import json

from env import ConstructionEnv


class EnvSaver:
    def __init__(self, env):
        self.env = env
        self.initial_state = None
        self.actions = []

    def start(self):
        self.initial_state = self.env.get_state()
        self.actions = []

    def step(self, actions, step_fn=None):
        if step_fn is None:
            obs, done, info = self.env.step(actions)
        else:
            obs, done, info = step_fn(actions)
        self.actions.append(list(actions))
        return obs, done, info

    def save(self, path):
        record = {
            "env_config": self.env.export_config(),
            "initial_state": self.initial_state,
            "actions": self.actions,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        config = record["env_config"]
        env = ConstructionEnv(
            grid_size=tuple(config["grid_size"]),
            num_blocks=config["num_objects"],
            max_steps=config["max_steps"],
            blocked_positions=config["block_pos"],
            object_colors=config.get("object_colors"),
            object_shapes=config.get("object_shapes"),
        )
        env.set_state(record["initial_state"])

        saver = cls(env)
        saver.initial_state = record["initial_state"]
        saver.actions = record["actions"]
        return saver

