from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.scorer import match
import random
from inspect_ai.dataset import Dataset, hf_dataset
from typing import Any, Literal, Union
from textwrap import dedent
from inspect_ai._eval.eval import eval


class EvaluateMMLU:

    def __init__(
        self,
        split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
        shuffle: bool = False,
        subjects: Union[list[str], str] = [],
    ) -> Dataset:

        dataset = hf_dataset(
            path="cais/mmlu",
            name="all",
            split=split,
            sample_fields=self.record_to_sample,
            shuffle=shuffle,
            seed=42,
        )

        # filter dataset if requested
        subjects = subjects if isinstance(subjects, list) else [subjects]
        if len(subjects) > 0:
            self.dataset = dataset.filter(
                name=f"{dataset.name}-{'-'.join(subjects)}",
                predicate=lambda sample: sample.metadata is not None
                and sample.metadata.get("subject") in subjects,
            )
        else:
            self.dataset = dataset

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a multiple-choice question into a `Sample` object.

        Parameters
        ----------
        record : dict[str, Any]
            A dictionary containing:
            - "question": str
            - "choices": list[str]
            - "answer": int (the index of the correct choice)
            - "subject": str (the category or subject of the question)

        Returns
        -------
        Sample
            A `Sample` object with:
            - input: The constructed multiple-choice question prompt.
            - target: The correct answer letter (e.g., "A", "B", "C", etc.).
            - metadata: A dictionary containing the subject of the question.
        """

        # Construct the main prompt including the question
        question_prompt = dedent(
            f"""
            Answer the following multiple choice question.

            {record["question"]}
        """
        ).strip()

        # Append the choices, labeling each with a letter starting at 'A'
        choices_prompt = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(record["choices"])
        )

        # Combine question and choices into a single prompt
        prompt = f"{question_prompt}\n{choices_prompt}\n\n"
        prompt += f"Provide your answer as a single letter in the range A-{chr(65 + len(record['choices']) - 1)}."

        # Determine the correct answer letter
        correct_answer_letter = chr(65 + record["answer"])

        return Sample(
            input=prompt,
            target=correct_answer_letter,
            metadata={"subject": record["subject"]},
        )

    @solver
    def match_solver(self) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            # set state.output.completion to a letter from A-D
            state.output.completion = random.choice(["A", "B", "C", "D"])

            return state

        return solve

    @task
    def match_task(self):
        return Task(
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=match(),
        )


if __name__ == "__main__":
    e = EvaluateMMLU()
    eval(e.match_task(), model="openai/gpt-3.5-turbo", limit=1000)
