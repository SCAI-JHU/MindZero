r"""
- CHANGELOG:
  - 0107: free text mental states, hypothesis reduction, other changes
  - 0112: last step only, create choices, unify train and test format.
"""

import asyncio
import math
from pathlib import Path
from pprint import pprint

import pandas as pd
from mathruler.grader import extract_boxed_content

from mods.bayesian_net import LikelihoodEstimator, Net, Node
from mods.client_configs import ESTIMATOR_CONFIGS, PROPOSER_CONFIGS
from mods.likelihood_estimator import print_token_stats
from mods.test_and_save import TestDataset


# propose_config = PROPOSER_CONFIGS["qwen3-235b-fp8"]
# propose_config = PROPOSER_CONFIGS["qwen3-4b"]
propose_config = PROPOSER_CONFIGS["qwen3-235b-fp8"]
# propose_config = PROPOSER_CONFIGS["gpt-4o"]
estimate_config = ESTIMATOR_CONFIGS["qwen3-235b-fp8"]
# estimate_config = ESTIMATOR_CONFIGS["gpt-4o"]

data_version = "mmtom_0112_hiyouga/tom"


def get_choice_response(x):
    if x.lower() in ("a", "b"):
        return x.lower()

    x = extract_boxed_content(x)
    if x.lower() in ("a", "b"):
        return x.lower()
    else:
        return None


async def batch_joint_prob(reward_inputs):
    # ^ joint prob
    nets = []
    likelihood_estimator = LikelihoodEstimator(**estimate_config)
    for reward_input in reward_inputs:
        agent_name = reward_input["agent_name"]

        response = get_choice_response(reward_input["response"])

        # * format error
        if response is None:
            nets.append(None)
            continue

        pred_tom = reward_input["choices_value"][response]
        last_action = reward_input["last_action"]

        # * compute joint prob
        net = Net(
            agent_name,
            {
                ("a", 1): Node("a", 1, last_action, prob=None),
                ("b", 1): Node("b", 1, pred_tom["belief"], prob=1),
                ("g", 1): Node("g", 1, pred_tom["goal"], prob=1),
            },
            None,
            likelihood_estimator,
        )
        nets.append(net)

    # ^ async inference
    # await tqdm.gather(*[net.infer() for net in nets if net is not None], desc="batch_joint_prob")

    # ^ accuracy
    accs = []
    for reward_input in reward_inputs:
        response = get_choice_response(reward_input["response"])
        gt = reward_input["answer"] if "answer" in reward_input else reward_input["ground_truth"]

        # * format error
        if response is None:
            accs.append(False)
            continue

        accs.append(response == gt)

    return nets, accs


def compute_score(reward_inputs):
    scores = []
    nets, accs = asyncio.run(batch_joint_prob(reward_inputs))
    # ! debug: [node for node in nets[0].nodes.values()]
    # ! debug: [node.prompt for node in nets[0].nodes.values()]
    # ! debug: print(nets[0].nodes['o', 1].prompt)
    # ! debug: print(nets[0].nodes['b', 1].prompt)
    # ! debug: print(nets[0].nodes['a', 1].prompt)

    for i in range(len(reward_inputs)):
        p_joint = 0 if nets[i] is None else nets[i].joint_prob
        p_last = 0 if nets[i] is None else nets[i][("a", 1)].prob
        scores.append(
            dict(
                overall=math.log(max(p_joint, 1e-7)),
                p_joint=p_joint,
                p_last=p_last,
                acc=accs[i],
            )
        )

    return scores


if __name__ == "__main__":
    data_home = Path("../../ShunchiZhang/StructuredToM/data")

    data_path = (data_home / data_version / "test.parquet").as_posix()

    propose_model = propose_config["gen_kwargs"]["model"]
    estimate_model = estimate_config["gen_kwargs"]["model"]

    save_id = f"Q={propose_model}__P={estimate_model}"
    save_path = Path("inference") / data_version / f"{save_id}.json"

    # * inference
    dataset = TestDataset(
        data_path=data_path,
        debug_len=4,
        format_prompt="math_qwen3",
        **propose_config,
    )

    dataset.inference()
    save_keys = ["answer", "response", "input_tokens", "output_tokens"]
    details, summary = dataset.save(save_path, save_keys=save_keys)
    pprint(summary)
    print_token_stats(details)

    # * evaluate
    scores = compute_score(dataset.dataset)
    scores = pd.DataFrame(scores).to_dict()
    for k, v in scores.items():
        dataset.dataset = dataset.dataset.add_column(k, v.values())
    save_keys += list(scores.keys())
    details, summary = dataset.save(save_path, save_keys=save_keys)
    pprint(summary)

    jp = list(scores["p_joint"].values())
    ac = list(scores["acc"].values())
    print(" | ".join([f"Acc = {sum(ac[:300]) / 300:7.2%}", f"P = {sum(jp[:300]) / 300:7.2%}", "Belief"]))
    print(" | ".join([f"Acc = {sum(ac[300:]) / 300:7.2%}", f"P = {sum(jp[300:]) / 300:7.2%}", "Goal"]))
    print(" | ".join([f"Acc = {sum(ac) / 600:7.2%}", f"P = {sum(jp) / 600:7.2%}", "Overall"]))
