from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from .benchmark import Benchmark
import json
from .metrics import ci_lower, ci_upper, median
from inspect_ai.scorer import (
    accuracy,
    Score,
    scorer,
)
from chat import get_structured_json_response_from_gpt


class DROP(Benchmark):

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
            "test": "validation",
        }

        self.dataset = self.filtered_hf_dataset(
            path="ucinlp/drop",
            name="default",
            split=split_mapping[split],
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a multiple-choice question into a `Sample` object.
        """

        passage = record["passage"]
        question = record["question"]
        target = json.dumps(record["answers_spans"]["spans"])

        # Construct the main prompt including the question
        prompt = dedent(
            f"""
            Given the following passage and question, succintly provide the answer to the question.

            Passage:
            {passage}

            Question:
            {question}
        """
        ).strip()

        prompt += f"\n\nOUTPUT ANSWER FORMAT: Provide your final answer as a string containing one or many words or numbers, or a short sentance."

        return Sample(
            input=prompt,
            target=target,
            metadata={
                "section_id": record["section_id"],
                "query_id": record["query_id"],
            },
        )

    @task
    def match_task(self):
        return Task(
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.span_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def span_match():
        async def score(state, target):

            target_list = json.loads(target.text)
            # target_list into bullet points
            target_prompt = "\n".join(f"* {item}" for item in target_list)

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
                },
                {
                    "role": "user",
                    "content": f"""
                Given this task:
                {state.input.split("OUTPUT ANSWER FORMAT")[0]}
                with the correct answer comprising all of the following bullet points:
                {target_prompt}

                How many of the {len(target_list)} bullet points did the user correctly include in their answer:

                Answer:
                {state.output.completion}.
                """,
                },
            ]
            response_format = {
                "thinking": "One sentence of step-by-step reasoning.",
                "number_of_correct_bullet_points": f"A single integer between 0 and {len(target_list)}.",
            }

            response = await get_structured_json_response_from_gpt(
                messages,
                response_format,
                model="gpt-4o-mini",
                temperature=0.5,
                retry=0,
            )

            number_of_correct = response.get("number_of_correct_bullet_points", 0)

            # convert string to int robustly (get the first integer in the string, the integer could have more than one digit)
            try:
                number_of_correct = int("".join(filter(str.isdigit, number_of_correct)))
                if number_of_correct < 0 or number_of_correct > len(target_list):
                    number_of_correct = 0
            except ValueError:
                number_of_correct = 0

            # Calculate Precision, Recall, and F1
            predicted_items = len(
                state.output.completion.split("\n")
            )  # Split on newlines for bullet points
            total_gold_items = len(target_list)

            # Avoid division by zero
            precision = (
                number_of_correct / predicted_items if predicted_items > 0 else 0
            )
            recall = number_of_correct / total_gold_items if total_gold_items > 0 else 0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0
            )

            return Score(
                name="span_match",
                value=f1,
                answer=state.output.completion,
                explanation=f"Thinking: {response.get('thinking')}, Precision: {precision}, Recall: {recall}, F1: {f1}",
            )

        return score
