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
        self.split = split

        self.args = args

        split_mapping = {
            "validation": "train",
            "test": "validation",
        }

        self.dataset = self.filtered_hf_dataset(
            path="ucinlp/drop",
            name="default",
            split=split,
            split_mapping=split_mapping,
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

            You will be asked to read a passage and answer a question. An example passage and Q&A is provided below.
            Passage:
            Non-nationals make up more than half of the population of Bahrain, with immigrants making up
            about 55% of the overall population. Of those, the vast majority come from South and Southeast Asia:
            according to various media reports and government statistics dated between 2005-2009 roughly 290,000
            Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians.
            Question: What two nationalities had the same number of people living in Bahrain between
            2005-2009?
            Answer [Not Given]: Pakistanis and Filipinos

            Your task is as follows:
            Given the following passage and question, succintly provide the answer to the question.

            Passage:
            {passage}

            Question:
            {question}
        """
        ).strip()

        prompt += f"\n\nOUTPUT ANSWER FORMAT: Provide your final answer as comma separated string of answers."

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
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="span_match",
                    value=0,
                    answer=state.output.completion,
                    explanation=f"Error in model response.",
                )

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
                },
                {
                    "role": "user",
                    "content": f"""
                Given this task:
                {state.input}
                with the correct answer being {target.text}.

                Did use respond with something meaningless rather than their guess at the answer? YES or NO.
                Here is their answer: {state.output.completion}.

                Examples of meaningless responses include:
                - "I don't know"
                - "I can't answer this"
                - "Final answer format" # But the answer is not provided
                - "Structured response" # But the answer is not provided
                - "Valid response" # But the answer is not provided
                - "No valid responses" # There is always a valid response
                - "No valid answer" # There is always a valid answer
                - "Please provide your question" # Any questions to you should be marked as meaningless
                - "Please provide your answer" # Any questions to you should be marked as meaningless
                """,
                },
            ]
            response_format = {
                "thinking": "One sentence of step-by-step reasoning.",
                "is_meaningless": "One word, YES or NO.",
            }

            response = await get_structured_json_response_from_gpt(
                messages,
                response_format,
                model="gpt-4o-mini",
                temperature=0.5,
                retry=0,
            )

            if "yes" in response.get("is_meaningless", "").lower():
                return Score(
                    name="span_match",
                    value=0,
                    answer=state.output.completion,
                    explanation=f"Meaningless respnose {response.get('thinking','')}",
                )

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
                state.output.completion.split(",")
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

            if f1 > 1:
                f1 = 0

            return Score(
                name="span_match",
                value=f1,
                answer=state.output.completion,
                explanation=f"Thinking: {response.get('thinking')}, Precision: {precision}, Recall: {recall}, F1: {f1}",
            )

        return score
