from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from .benchmark import Benchmark
import json
import hashlib


import random


class GPQA(Benchmark):

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:

        self.args = args

        split_mapping = {
            "validation": "train",
            "test": "train",
        }

        self.split = split
        self.validation_set = set()

        self.dataset = self.filtered_hf_dataset(
            path="Idavidrein/gpqa",
            name="gpqa_diamond",
            split=split_mapping[split],
            sample_fields=self._record_to_sample,
            shuffle=False,  # deliberately False to validation & test sets are not mixed
            seed=self.args.random_seed,
            limit=limit,
        )

    def benchmark_filter(self, example: dict[str, Any]) -> bool:

        # Convert the dictionary to a JSON string with sorted keys for consistency
        data_string = json.dumps(example, sort_keys=True)
        # Create a hash of the JSON string
        unique_id = hashlib.sha256(data_string.encode()).hexdigest()

        if len(self.validation_set) < 32:
            self.validation_set.add(unique_id)
        if self.split == "validation":
            if unique_id in self.validation_set:
                return True

        elif self.split == "test":
            if unique_id not in self.validation_set:
                return True

        return False

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a multiple-choice question into a `Sample` object.
        """

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following multiple choice question.

            {record["Question"]}
        """
        ).strip()

        choices = [
            record["Correct Answer"],
            record["Incorrect Answer 1"],
            record["Incorrect Answer 2"],
            record["Incorrect Answer 3"],
        ]

        random.shuffle(choices)

        # Append the choices, labeling each with a letter starting at 'A'
        choices_prompt = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
        )
        print("choices_prompt", choices_prompt)

        # Combine question and choices into a single prompt
        prompt = (
            f"{question_prompt}\n{choices_prompt}\n"  # Removed the extra line break
        )

        prompt += f"OUTPUT ANSWER FORMAT: Provide your final answer as a single letter in the range A-{chr(65 + len(choices) - 1)}."

        # Determine the correct answer letter
        correct_answer_letter = chr(65 + choices.index(record["Correct Answer"]))

        return Sample(
            input=prompt,
            target=correct_answer_letter,
            metadata={"correct_answer": record["Correct Answer"]},
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
