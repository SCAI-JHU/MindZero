import asyncio
import math

from openai import AsyncOpenAI


def print_token_stats(details):
    for key in ("input_tokens", "output_tokens"):
        tokens = [x[key] for x in details]
        mn = min(tokens)
        mx = max(tokens)
        avg = sum(tokens) / len(tokens)
        print(f"{key}: {mn} ~ {mx} (avg = {avg:.1f})")


class LikelihoodEstimator:
    def __init__(self, aclient_kwargs, gen_kwargs):
        self.aclient = AsyncOpenAI(**aclient_kwargs)
        self.gen_kwargs = gen_kwargs
        self.sem = asyncio.Semaphore(64)

    async def __call__(self, prompt):
        async with self.sem:
            response = await self.aclient.chat.completions.create(
                messages=[dict(role="user", content=prompt)],
                **self.gen_kwargs,
            )

            top_probs = {i.token: math.exp(i.logprob) for i in response.choices[0].logprobs.content[0].top_logprobs}
            prob = top_probs["A"] / (top_probs["A"] + top_probs["B"])

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return prob
