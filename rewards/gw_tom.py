from pathlib import Path

import pandas as pd

from mod.client_configs import PROPOSER_CONFIGS
from mod.construction.estimate_likelihood import compute_score
from mod.test_and_save import TestDataset


propose_config = PROPOSER_CONFIGS["qwen3-4b-vl-gw0125"]

data_version = "gw_0125_hiyouga/tom"


def compute_score_sum(reward_inputs):
    return compute_score(reward_inputs, reward_type="sum")


def compute_score_last(reward_inputs):
    return compute_score(reward_inputs, reward_type="last")


if __name__ == "__main__":
    data_home = Path("../../ShunchiZhang/StructuredToM/data")

    # data_path = (data_home / data_version / "test.parquet").as_posix()
    data_path = (data_home / data_version / "test_easy_fix.parquet").as_posix()
    propose_model = propose_config["gen_kwargs"]["model"]

    save_id = f"{propose_model}"
    save_path = Path("inference") / data_version / f"{save_id}.json"
    print(f"Results will be saved to: {save_path.parent}")

    dataset = TestDataset(
        data_path=data_path,
        # debug_len=3,
        format_prompt="nonthink",
        **propose_config,
    )
    dataset.inference()
    save_keys = ["question_type", "answer", "response", "input_tokens", "output_tokens"]
    details, summary = dataset.save(save_path, save_keys=save_keys)

    scores = compute_score_sum(dataset.dataset)
    scores = pd.DataFrame(scores).to_dict()
    for k, v in scores.items():
        dataset.dataset = dataset.dataset.add_column(k, v.values())
    save_keys += list(scores.keys())
    details, summary = dataset.save(save_path, save_keys=save_keys)
