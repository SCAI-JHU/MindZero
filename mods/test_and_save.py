import asyncio
import base64
import json
from typing import Iterable

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from libs.EasyR1.verl.utils.dataset import RLHFDataset


class TestDataset(RLHFDataset):
    def __init__(
        self,
        data_path,
        format_prompt="math_qwen3",
        aclient_kwargs=None,
        gen_kwargs=None,
        debug_len=None,
    ):
        super().__init__(
            data_path=data_path,
            tokenizer=None,
            processor=None,
            prompt_key="problem",
            answer_key="answer",
            image_key="images",
            video_key="videos",
            image_dir=None,
            video_fps=None,
            max_prompt_length=None,
            truncation=None,
            format_prompt=f"examples/format_prompt/{format_prompt}.jinja",
            min_pixels=None,
            max_pixels=None,
            filter_overlong_prompts=False,
            filter_overlong_prompts_workers=None,
        )

        if isinstance(debug_len, int):
            indices = [i * len(self) // debug_len for i in range(debug_len)]
            self.dataset = self.dataset.select(indices)
        elif isinstance(debug_len, Iterable):
            self.dataset = self.dataset.select(debug_len)

        self.aclient = AsyncOpenAI(**aclient_kwargs)
        self.gen_kwargs = gen_kwargs
        self.sem = asyncio.Semaphore(64)

        self.responses = {}

    def __getitem__(self, index):
        example = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            images = example[self.image_key]
            assert len(images) == 1, "only support one image for now"
            url = f"data:image/png;base64,{base64.b64encode(images[0]['bytes']).decode('utf-8')}"
            messages[0]["content"][0] = dict(type="image_url", image_url=dict(url=url))

        return messages

    async def inference_one(self, i):
        async with self.sem:
            response = await self.aclient.chat.completions.create(
                messages=self[i],
                **self.gen_kwargs,
            )
            response = dict(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                response=response.choices[0].message.content,
            )
            return response

    async def inference_batch(self):
        tasks = [self.inference_one(i) for i in range(len(self))]
        responses = await tqdm.gather(*tasks, desc="TestDataset.inference")
        return responses

    def inference(self):
        if "response" in self.dataset.column_names:
            return

        responses = asyncio.run(self.inference_batch())
        input_tokens = [r["input_tokens"] for r in responses]
        output_tokens = [r["output_tokens"] for r in responses]
        responses = [r["response"] for r in responses]

        self.dataset = self.dataset.add_column("input_tokens", input_tokens)
        self.dataset = self.dataset.add_column("output_tokens", output_tokens)
        self.dataset = self.dataset.add_column("response", responses)

    def save(self, save_to, save_keys):
        save_to.parent.mkdir(parents=True, exist_ok=True)
        subset = self.dataset.select_columns(save_keys)

        # * details: case by case
        details = subset.to_list()
        with save_to.with_suffix(".details.json").open("w") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        # * summary: averaged metrics
        summary = dict()
        for k, v in subset.to_dict().items():
            try:
                nums = [float(_) for _ in v]
                avg = sum(nums) / len(nums)
                if 0 <= avg <= 1:
                    avg = f"{avg:.2%}"
                else:
                    avg = f"{avg:.2f}"
                summary[k] = avg
            except:
                pass

        with save_to.with_suffix(".summary.json").open("w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return details, summary
