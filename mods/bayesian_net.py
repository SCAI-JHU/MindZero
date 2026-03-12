import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI


@dataclass
class Node:
    name: str
    time: int
    value: str
    prob: float = None
    prompt: str = None

    def __repr__(self):
        return f"Node(name={self.name}, time={self.time}, value={self.value}, prob={self.prob})"


class LikelihoodEstimator:
    def __init__(self, aclient_kwargs, gen_kwargs):
        self.aclient = AsyncOpenAI(**aclient_kwargs)
        self.gen_kwargs = gen_kwargs
        self.sem = asyncio.Semaphore(64)

        prompts_dir = Path(__file__).parent.parent / "prompts" / "bayesian_net"
        self.jinja_env = Environment(
            loader=FileSystemLoader(prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_prompt(self, main_agent, node, parents, type):
        fill_data = {parent.name: parent.value for parent in parents}
        fill_data["main_agent"] = main_agent
        if type is not None:
            fill_data["type"] = type

        prompt = self.jinja_env.get_template(f"{node.name}.jinja").render(**fill_data)
        prompt = self.jinja_env.get_template("common.jinja").render(prompt=prompt, statement=node.value)

        return prompt

    async def __call__(self, main_agent, node, parents, type):
        async with self.sem:
            prompt = self._get_prompt(main_agent, node, parents, type)
            node.prompt = prompt

            response = await self.aclient.chat.completions.create(
                messages=[dict(role="user", content=prompt)],
                **self.gen_kwargs,
            )

            top_probs = {i.token: math.exp(i.logprob) for i in response.choices[0].logprobs.content[0].top_logprobs}
            node.prob = top_probs["A"] / (top_probs["A"] + top_probs["B"])

            return node.prob


class Net:
    def __init__(self, main_agent, nodes, types, likelihood_estimator):
        self.main_agent = main_agent

        self.nodes = nodes
        self.dependency_graph = {
            "s": ["s", "a"],
            "o": ["s"],
            "b": ["b", "o"],
            "a": ["b", "g"],
        }
        types = dict.fromkeys(self.dependency_graph.keys())
        types.update(types or dict())
        self.types = types

        self.estimator = likelihood_estimator

    def __getitem__(self, key):
        return self.nodes[key]

    @property
    def joint_prob(self):
        return math.prod(node.prob for node in self.nodes.values())

    async def infer(self):
        """
        Efficient inference by parallel LLM calls for likelihood estimation.
        """
        tasks = []
        for node in self.nodes.values():
            # * skip if given likelihood
            if node.prob is not None:
                continue

            parents = []
            for parent_name in self.dependency_graph[node.name]:
                if parent_name == node.name:
                    # * update using previous value
                    # * e.g. state, belief
                    parents.append(self[parent_name, node.time - 1])
                else:
                    parents.append(self[parent_name, node.time])

            tasks.append(
                self.estimator(
                    self.main_agent,
                    node,
                    parents,
                    self.types[node.name],
                )
            )

        await asyncio.gather(*tasks)


async def estimate_multiple():
    # base_url = os.getenv("REWARD_MODEL_BASE_URL")
    base_url = "http://n16:9991/v1"
    aclient = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    likelihood_estimator = LikelihoodEstimator(aclient)

    nodes = {
        ("b", 0): Node("b", 0, "NONE", 1),
        # ----- time 1 -----
        ("s", 1): Node("s", 1, "NONE", 1),
        ("b", 1): Node("b", 1, "xxx", None),
        ("o", 1): Node("o", 1, "xxx", None),
        ("a", 1): Node("a", 1, "Jayden entered the attic.", 1),
        # ----- time 2 -----
        ("s", 2): Node(
            "s", 2, "Jayden is in the attic. The hat is in the envelope in the attic. Hannah is in the attic.", 1
        ),
        ("b", 2): Node("b", 2, "xxx", None),
        ("o", 2): Node("o", 2, "xxx", None),
        ("a", 2): Node("a", 2, "Jayden exited the attic.", 1),
        # ----- time 3 -----
        ("s", 3): Node(
            "s",
            3,
            "Jayden is not in the attic. The hat is in the envelope in the attic. Hannah is in the attic.",
            1,
        ),
        ("b", 3): Node("b", 3, "xxx", None),
        ("o", 3): Node("o", 3, "xxx", None),
        ("a", 3): Node("a", 3, "Jayden entered the attic.", 1),
        # ----- time 4 -----
        ("s", 4): Node(
            "s",
            4,
            "Jayden is in the attic. The hat is in the container in the attic. Hannah is in the attic. The container is in the attic.",
            1,
        ),
        ("b", 4): Node("b", 4, "xxx", None),
        ("o", 4): Node("o", 4, "xxx", None),
        ("a", 4): Node("a", 4, "Jayden searched for the hat in the envelope in the attic.", 1),
    }
    nodes = {
        ("b", 0): Node("b", 0, "xxx", None),
        ("s", 1): Node("s", 1, "NONE", 1),
        ("b", 1): Node("b", 1, "xxx", None),
        ("o", 1): Node("o", 1, "xxx", None),
        ("a", 1): Node("a", 1, "Jayden entered the attic.", 1),
    }
    nets = [
        Net("Jayden", nodes, None, likelihood_estimator),
        # Net(nodes, "Jayden", None, likelihood_estimator),
    ]
    await asyncio.gather(*[net.infer() for net in nets])
    for i, net in enumerate(nets):
        print(f"Net {i} joint probability: {net.joint_prob}")


if __name__ == "__main__":
    asyncio.run(estimate_multiple())
