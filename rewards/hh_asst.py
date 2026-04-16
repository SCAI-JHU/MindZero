import asyncio
import copy
import json
import math
from collections import Counter
from pathlib import Path
from pprint import pprint

import pandas as pd
from json_repair import repair_json
from tqdm.asyncio import tqdm

import mods.autotom_prompts as prompts
from mods.client_configs import ESTIMATOR_CONFIGS, PROPOSER_CONFIGS
from mods.likelihood_estimator import LikelihoodEstimator, print_token_stats
from mods.test_and_save import TestDataset


# propose_config = PROPOSER_CONFIGS["qwen3-235b-fp8"]
# propose_config = PROPOSER_CONFIGS["qwen3-4b"]
# propose_config = PROPOSER_CONFIGS["llama3-3b"]
# propose_config = PROPOSER_CONFIGS["llama3-8b-fmt-step15"]
propose_config = PROPOSER_CONFIGS["llama3-3b-fmt-distill"]
# propose_config = PROPOSER_CONFIGS["qwen3-4b-ft"]
estimate_config = ESTIMATOR_CONFIGS["qwen3-235b-fp8"]

data_version = "vh_0126_hiyouga/asst"

EPS = 1e-10
N_PARTICLES = 10


def parse_proposals(x):
    try:
        obj = repair_json(x, return_objects=True)
        if isinstance(obj, list):
            obj = obj[-1]
        if isinstance(obj, dict):
            obj = dict(particles=obj["particles"])
        return prompts.GoalParticles.model_validate(obj)
    except Exception:
        return None


def safe_log(x):
    return math.log(max(x, EPS))


def safe_div(x, y):
    return x / max(y, EPS)


def metrics(ps, qs, prior_version):
    if ps is None and qs is None:
        return dict(
            n=0,
            ent_p=0,  # [0, \infty]
            ent_q=0,  # [0, \infty]
            neg_ce_fwd=safe_log(0),  # [-infty, 0]
            neg_ce_bwd=safe_log(0),  # [-infty, 0]
            neg_kl_fwd=safe_log(0),  # [-infty, 0]
            neg_kl_bwd=safe_log(0),  # [-infty, 0]
            avg_p=0,
        )
    else:
        n = len(ps)
        if isinstance(prior_version, int):
            avg_lkl = sum([x[0] for x in ps]) / n
            avg_prr = sum([x[1] for x in ps]) / n
            ps = [lkl * prr for lkl, prr in ps]

        ent_p = -sum(p * safe_log(p) for p in ps)
        ent_q = -sum(q * safe_log(q) for q in qs)
        ce_fwd = -sum(p * safe_log(q) for p, q in zip(ps, qs))
        ce_bwd = -sum(q * safe_log(p) for p, q in zip(ps, qs))
        kl_fwd = ce_fwd - ent_p
        kl_bwd = ce_bwd - ent_q

        avg_p = sum(ps) / n

        result = dict(
            n=n,
            ent_p=ent_p,
            ent_q=ent_q,
            neg_ce_fwd=-ce_fwd,
            neg_ce_bwd=-ce_bwd,
            neg_kl_fwd=-kl_fwd,
            neg_kl_bwd=-kl_bwd,
            avg_p=avg_p,
        )

    if isinstance(prior_version, int):
        result.update(
            dict(
                avg_lkl=avg_lkl,
                avg_prr=avg_prr,
            )
        )

    return result


async def forward_likelihood_once(particle, prompt_info, estimator, prior_version):
    if len(particle.objects) == 0:
        if isinstance(prior_version, int):
            return (0.0, 0.0)
        else:
            return 0.0

    prompt = prompts.forward_likelihood_all_time(**prompt_info, goal=particle.to_natlang())
    prob_lkl = await estimator(prompt)

    if isinstance(prior_version, int):
        prompt = prompts.prior_by_version[prior_version](goal=particle.to_natlang())
        prob_prr = await estimator(prompt)
        return (prob_lkl, prob_prr)
    else:
        return prob_lkl


async def forward_likelihood_batch(prompt_info, particles, human_done, estimator, prior_version):
    forward_particles = copy.deepcopy(particles)
    # try:
    #     forward_particles.minus_objects(human_done)
    # except Exception:
    #     print(f"{type(forward_particles) = }")
    #     for particle in forward_particles.particles:
    #         print(f"{particle.objects = }")
    #     raise

    tasks = [
        forward_likelihood_once(particle, prompt_info, estimator, prior_version)
        for particle in forward_particles.particles
    ]

    ps = await asyncio.gather(*tasks)
    return ps


async def compute_metrics_once(reward_input, estimator, prior_version):
    particles = parse_proposals(reward_input["response"])

    # * exceptions
    err_metrics = dict(**metrics(None, None, prior_version), err=1, fmt=0)

    if particles is None:
        return err_metrics

    particles.merge_duplicates()
    if len(particles.particles) != N_PARTICLES:
        return err_metrics

    # * format only
    if prior_version == "skip":
        return dict(**metrics(None, None, prior_version), err=0, fmt=1)

    # * normal cases
    prompt_info = json.loads(reward_input["prompt_info"])
    human_done = Counter(json.loads(reward_input["human_done"]))

    particles.normalize()
    ps = await forward_likelihood_batch(prompt_info, particles, human_done, estimator, prior_version)
    qs = [particle.p for particle in particles.particles]

    return dict(**metrics(ps, qs, prior_version), err=0, fmt=1)


async def compute_metrics_batch(reward_inputs, prior_version):
    estimator = LikelihoodEstimator(**estimate_config)
    tasks = [compute_metrics_once(reward_input, estimator, prior_version) for reward_input in reward_inputs]
    batch_metrics = await tqdm.gather(*tasks, desc="compute_metrics_batch")

    return batch_metrics


# ===============================================


def compute_score(reward_inputs, key, entropy_bonus, prior_version):
    batch_metrics = asyncio.run(compute_metrics_batch(reward_inputs, prior_version))

    scores = []
    for metrics in batch_metrics:
        metrics["overall"] = metrics[key]

        if entropy_bonus != 0:
            if key == "neg_ce_fwd":
                metrics["overall"] += entropy_bonus * metrics["ent_p"]
            if key == "neg_ce_bwd":
                metrics["overall"] += entropy_bonus * metrics["ent_q"]
            else:
                raise ValueError(f"Invalid key: {key}")

        scores.append(metrics)

    return scores


# * 1. entropy bonus? 2. forward or reverse KL?
def compute_score_ce_fwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_fwd", entropy_bonus=0.0)


def compute_score_kl_fwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_fwd", entropy_bonus=1.0)


def compute_score_ce_bwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=0.0)


def compute_score_kl_bwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=1.0)


# * 3. icml ablation configs
def compute_score_lkl(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=1.0, prior_version=None)


def compute_score_lkl_prr(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=1.0, prior_version=1)


def compute_score_lkl_prr_no_entropy(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=0.0, prior_version=1)


def compute_score_fmt_only(reward_inputs):
    return compute_score(reward_inputs, key="fmt", entropy_bonus=0.0, prior_version="skip")


# * 4. prior version
def compute_score_prr_v2(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=1.0, prior_version=2)


def compute_score_prr_v3(reward_inputs):
    return compute_score(reward_inputs, key="neg_ce_bwd", entropy_bonus=1.0, prior_version=3)


if __name__ == "__main__":
    data_home = Path("../../ShunchiZhang/StructuredToM/data")

    data_path = (data_home / data_version / "test.parquet").as_posix()  # 20
    # data_path = (data_home / data_version / "train.parquet").as_posix()  # 968

    propose_model = propose_config["gen_kwargs"]["model"]
    estimate_model = estimate_config["gen_kwargs"]["model"]

    reward_func = "_fmt_only"

    save_id = f"Q={propose_model}__P={estimate_model}__R={reward_func}"
    save_path = Path("inference") / data_version / f"{save_id}.json"
    print(f"Results will be saved to: {save_path.parent}")

    dataset = TestDataset(
        data_path=data_path,
        # debug_len=3,
        format_prompt="min_json",
        **propose_config,
    )

    dataset.inference()
    save_keys = ["problem", "answer", "response", "input_tokens", "output_tokens"]
    details, summary = dataset.save(save_path, save_keys=save_keys)
    pprint(summary)
    print_token_stats(details)

    scores = eval(f"compute_score{reward_func}")(dataset.dataset)
    scores = pd.DataFrame(scores).to_dict()
    for k, v in scores.items():
        dataset.dataset = dataset.dataset.add_column(k, v.values())
    save_keys += list(scores.keys())
    details, summary = dataset.save(save_path, save_keys=save_keys)
    pprint(summary)
