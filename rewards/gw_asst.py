from mods.construction.estimate_distribution import compute_score as _compute_score


def compute_score(reward_inputs, key="neg_kl_backward", entropy_bonus=0.0, distance=False):
    """Reward for GW 0125 assistant tasks based on goal distribution matching.

    Args:
        reward_inputs: list of dicts containing at least
            response/env_config/initial_state/actions.
        key: which metric to use as overall score.
        entropy_bonus: only used when key is "neg_kl_backward_entropy_bonus".
    """
    return _compute_score(
        reward_inputs,
        key=key,
        entropy_bonus=entropy_bonus if key == "neg_kl_backward_entropy_bonus" else None,
        distance=distance,
    )


def compute_score_kl_fwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_kl_forward")


def compute_score_kl_bwd(reward_inputs):
    return compute_score(reward_inputs, key="neg_kl_backward")


def compute_score_kl_bwd_distance(reward_inputs):
    return compute_score(reward_inputs, key="neg_kl_backward", distance=True)


def compute_score_kl_bwd_entropy_distance(reward_inputs, entropy_bonus=-1.0):
    return compute_score(
        reward_inputs,
        key="neg_kl_backward_entropy_bonus",
        entropy_bonus=entropy_bonus,
        distance=True,
    )
