import json
import os

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PROPOSER_GEN_KWARGS = dict(
    temperature=0.0,
    top_p=1e-5,
)

ESTIMATOR_GEN_KWARGS = dict(
    temperature=0.0,
    top_p=1e-5,
    logprobs=True,
    top_logprobs=5,
    max_completion_tokens=4,
)


PROPOSER_CONFIGS = dict()
ESTIMATOR_CONFIGS = dict()


def scan_ports():
    vllm_base_urls, vllm_models = [], []
    nodes = [f"n{i:02d}" for i in range(1, 16 + 1)] + [
        f"c{i:03d}" for i in range(1, 11 + 1)
    ]
    for node in nodes:
        cmd = f"curl -s -X GET http://{node}:9991/v1/models"
        result = os.popen(cmd).read()
        try:
            result = json.loads(result)
            vllm_base_urls.append(f"http://{node}:9991/v1")
            vllm_models.append(result["data"][0]["id"])
        except:
            pass
    print(f"=== Scanning Results ===\n{list(zip(vllm_base_urls, vllm_models))}")
    return vllm_base_urls, vllm_models


def register():
    global PROPOSER_CONFIGS, ESTIMATOR_CONFIGS

    # vllm_base_urls, vllm_models = scan_ports()
    # for vllm_base_url, vllm_model in zip(vllm_base_urls, vllm_models):
    for vllm_base_url, vllm_model in [
        # * pretrained models
        ("http://n11:9991/v1", "llama3-8b"),
        ("http://n12:9991/v1", "llama3-3b"),
        ("http://c003:9991/v1", "qwen3-4b"),
        ("http://n04:9991/v1", "qwen3-4b-vl"),
        ("http://h15:9991/v1", "qwen3-235b-fp8"),
        ("http://n--:9991/v1", "qwen3-235b-fp8-vl"),
        # * finetuned models
        ("http://n--:9991/v1", "qwen3-4b-ft"),
        ("http://n10:9991/v1", "qwen3-4b-vl-gw0125"),
        ("http://n04:9991/v1", "llama3-8b-fmt-step15"),
        ("http://n12:9991/v1", "llama3-3b-fmt-distill"),
        ("https://nvl:9991/v1", "qwen3-4b-vh-prr-step40"),
        ("https://nvl:9991/v1", "qwen3-4b-vh-prr-step60"),
        ("https://nvl:9991/v1", "qwen3-4b-vh-lkl-step40"),
        ("https://nvl:9991/v1", "qwen3-4b-vh-lkl-step60"),
    ]:
        PROPOSER_CONFIGS[vllm_model] = dict(
            aclient_kwargs=dict(base_url=vllm_base_url, api_key="EMPTY"),
            gen_kwargs=dict(model=vllm_model, **PROPOSER_GEN_KWARGS),
        )
        ESTIMATOR_CONFIGS[vllm_model] = dict(
            aclient_kwargs=dict(base_url=vllm_base_url, api_key="EMPTY"),
            gen_kwargs=dict(model=vllm_model, **ESTIMATOR_GEN_KWARGS),
        )

    for openai_model in [
        "gpt-4o",
        "gpt-5.2",
    ]:
        PROPOSER_CONFIGS[openai_model] = dict(
            aclient_kwargs=dict(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY),
            gen_kwargs=dict(model=openai_model, **PROPOSER_GEN_KWARGS),
        )
        ESTIMATOR_CONFIGS[openai_model] = dict(
            aclient_kwargs=dict(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY),
            gen_kwargs=dict(model=openai_model, **ESTIMATOR_GEN_KWARGS),
        )


register()
