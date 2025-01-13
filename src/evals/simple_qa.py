from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from .benchmark import Benchmark
import json
import hashlib


class SimpleQA(Benchmark):

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:

        self.args = args

        split_mapping = {
            "validation": "test",
            "test": "test",
        }

        self.split = split
        self.validation_set = set()
        self.validation_set_size = 645

        self.dataset = self.filtered_hf_dataset(
            path="basicv8vc/SimpleQA",
            name="default",
            split=split_mapping[split],
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def benchmark_filter(self, example: dict[str, Any]) -> bool:

        # Convert the dictionary to a JSON string with sorted keys for consistency
        data_string = json.dumps(example, sort_keys=True)
        # Create a hash of the JSON string
        unique_id = hashlib.sha256(data_string.encode()).hexdigest()

        if len(self.validation_set) < self.validation_set_size:
            self.validation_set.add(unique_id)
        if self.split == "validation":
            if unique_id in self.validation_set:
                return True

        elif self.split == "test":
            if unique_id not in self.validation_set:
                return True

        return False

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following question:

            {record["problem"]}
        """
        ).strip()

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n\n"
        prompt += f"OUTPUT ANSWER FORMAT: Provide your final answer as succinctly as possible E.g. a single number, date, or a few words."

        return Sample(
            input=prompt,
            target=str(record["answer"]),
            metadata={"answer": record["answer"]},
        )

    @task
    def match_task(self):
        return Task(
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.llm_match(),
            config=GenerateConfig(temperature=0.5),
        )
