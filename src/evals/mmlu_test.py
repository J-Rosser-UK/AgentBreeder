"""
Measuring Massive Multitask Language Understanding

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300

Based on: https://github.com/openai/simple-evals/blob/main/mmlu_eval.py

# eval all subjects w/ 500 randomly selected samples
inspect eval inspect_evals/mmlu --limit 500

# add chain of thought
inspect eval inspect_evals/mmlu --limit 500 -T cot=true

# eval selected subjects
inspect eval inspect_evals/mmlu -T subjects=anatomy
inspect eval inspect_evals/mmlu -T subjects=astronomy
inspect eval inspect_evals/mmlu -T subjects=anatomy,astronomy
"""

from typing import Any, Literal, Union

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import exact
from inspect_ai.solver import (
    Choices,
    Generate,
    Solver,
    TaskState,
    generate,
    multiple_choice,
    solver,
)
from inspect_ai.solver._multiple_choice import (
    answer_character,
    answer_index,
    answer_options,
    unshuffle_choices,
)
from inspect_ai.solver._task_state import Choices, TaskState

from inspect_ai.scorer._metric import CORRECT, INCORRECT, Score
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import Scorer, scorer
from inspect_ai.scorer._target import Target

import random


MMLU_MULTISHOT_PROMPT_TEMPLATE = r"""
The following are multiple choice questions about {subject_name}.

{subject_dev_questions}

{test_question}
""".strip()

MMLU_MULTISHOT_QUESTION_TEMPLATE = r"""
{question}
{choices}
Answer: {answer}
""".strip()


def _choices_are_shuffled(choices: Choices) -> bool:
    return any(i != choice.original_position for i, choice in enumerate(choices))


def _score_target(target: Target, choices: Choices) -> tuple[list[int], list[str]]:
    target_positions = [
        answer_index(target_character) for target_character in target.text
    ]

    choice_positions = [i for i, choice in enumerate(choices) if choice.correct is True]

    answers = [answer_character(choice) for choice in choice_positions]

    return target_positions, answers


def _shuffled_explanation(choices: Choices) -> str:
    generated_answers = [
        answer_character(i)
        for i, choice in enumerate(choices)
        if choice.correct is True
    ]

    return f"Choices were shuffled before generating a response, the following was sent to the model:\n\n{answer_options(choices)}\nShuffled answer:\nANSWER: {', '.join(generated_answers)}"


@scorer(metrics=[accuracy(), stderr()])
def choice() -> Scorer:
    """
    Scorer for multiple choice answers, required by the `multiple_choice` solver.

    This assumes that the model was called using a template ordered with letters
    corresponding to the answers, so something like:

        What is the capital of France?

        A) Paris
        B) Berlin
        C) London

    The target for the dataset will then have a letter corresponding to the
    correct answer, e.g. the `Target` would be `"A"` for the above question. If
    multiple choices are correct, the `Target` can be an array of these letters.
    """

    async def score(state: TaskState, target: Target) -> Score:

        print("TARGET", target.text)
        print("COMPLETION", state.output.completion)
        choices = state.choices
        print(choices._choices)

        if _choices_are_shuffled(choices):
            explanation = _shuffled_explanation(choices)
            # Unshuffle the choices so that we can score them correctly against
            # the target
            choices = unshuffle_choices(choices)
        else:

            explanation = state.output.completion

        target_positions, answers = _score_target(target, choices)

        generated_selected_choices = [
            i for i, choice in enumerate(choices) if choice.correct is True
        ]

        print("TARGET POSITIONS", target_positions)
        print("GENERATED CHOICES", generated_selected_choices)
        target_matches_choices = generated_selected_choices == sorted(target_positions)

        return Score(
            value=CORRECT if target_matches_choices else INCORRECT,
            answer=", ".join(answers),
            explanation=explanation,
        )

    return score


@task
def mmlu_0_shot(subjects: str | list[str] = [], cot: bool = False) -> Task:
    """
    Inspect Task implementation for MMLU, with 0-shot prompting.

    Args:
        subjects (str | list[str]): Subjects to filter to
        cot (bool): Whether to use chain of thought
    """
    return Task(
        # (shuffle so that --limit draws from multiple subjects)
        dataset=get_mmlu_dataset("test", shuffle=True, subjects=subjects),
        solver=random_choice_solver(),
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )


# @task
# def mmlu_5_shot(subjects: str | list[str] = []) -> Task:
#     """
#     Inspect Task implementation for MMLU, with 5-shot prompting.

#     Args:
#         subjects (str | list[str]): Subjects to filter to
#     """
#     return Task(
#         # (shuffle so that --limit draws from multiple subjects)
#         dataset=get_mmlu_dataset("test", shuffle=True, subjects=subjects),
#         solver=[
#             random_choice_solver(),
#             # To perfectly match the MMLU paper, we should be getting the logprobs
#             # for the tokens 'A', 'B', 'C' and 'D' and pick the greatest one. We
#             # instead get a simple completion, given logprobs are not supported
#             # by many models supported by Inspect.
#             generate(max_tokens=1),
#         ],
#         scorer=exact(),
#         config=GenerateConfig(temperature=0),
#     )


@solver
def mmlu_5_shot_solver() -> Solver:
    """A custom solver for MMLU 5-shot.

    It uses the dev dataset to generate a prompt with 5 example answered questions.

    It expects the TaskState to have a metadata field "subject" that is the
    subject of the question, e.g. "astronomy".

    Returns:
        Solver: A solver that uses the dev dataset to generate a prompt with
          5 example answered questions.
    """
    dev_dataset = get_mmlu_dataset("dev", shuffle=False)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        print("QUESTION", state.input)
        print("CHOICES", state.choices._choices)
        # e.g. "abstract_algebra" -> "abstract algebra"
        subject_name = state.metadata["subject"].replace("_", " ")

        subject_dev_dataset = dev_dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata["subject"] == state.metadata["subject"]
        )
        subject_dev_questions = "\n\n".join(
            [
                format_mmlu_question(
                    question=str(sample.input),
                    choices=Choices(
                        sample.choices if sample.choices is not None else []
                    ),
                    answer=str(sample.target),
                )
                for sample in subject_dev_dataset
            ]
        )

        test_question = format_mmlu_question(
            question=str(state.input),
            choices=state.choices,
        )

        state.user_prompt.text = MMLU_MULTISHOT_PROMPT_TEMPLATE.format(
            subject_name=subject_name,
            subject_dev_questions=subject_dev_questions,
            test_question=test_question,
        )

        return state

    return solve


def format_mmlu_question(question: str, choices: Choices, answer: str = "") -> str:
    r"""Format an MMLU-style prompt for a question, choices and answer.

    Args:
        question (str): Question to pose.
        choices (Choices): Choices to select from.
        answer (str, optional): Correct answer. Defaults to "". If this is not
          provided the prompt will still include space for an answer.

    Returns:
        Prompt for question and choices.

    Example:
    >>> format_mmlu_question("What is 5 squared?", Choices(["20", "25", "50", "55"]), "B")
    "What is 5 squared?\n(A) 20\n(B) 25\n(C) 50\n(D) 55\nAnswer: B"
    """
    return MMLU_MULTISHOT_QUESTION_TEMPLATE.format(
        question=question, choices=format_mmlu_choices(choices), answer=answer
    )


def format_mmlu_choices(choices: Choices) -> str:
    r"""Format choices in the style of the MMLU paper.

    Returns the `choices` formatted as a multiple choice question in style of the
    MMLU paper (with round parentheses around letters). This is necessary as by
    default Inspect will only put parentheses after letters.

    Args:
        choices (Choices): Choices to select from.

    Returns:
        Formatted choices.

    Example:
    >>> format_mmlu_choices(Choices(["20", "25", "50", "55"]))
    "(A) 20\n(B) 25\n(C) 50\n(D) 55"
    """
    indexes = list(range(len(choices)))

    return "\n".join(
        [f"({chr(65 + i)}) {choices[j].value}" for i, j in enumerate(indexes)]
    )


def get_mmlu_dataset(
    split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
    shuffle: bool = False,
    subjects: Union[list[str], str] = [],
) -> Dataset:
    dataset = hf_dataset(
        path="cais/mmlu",
        name="all",
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
        seed=42,
    )

    # filter dataset if requested
    subjects = subjects if isinstance(subjects, list) else [subjects]
    if len(subjects) > 0:
        return dataset.filter(
            name=f"{dataset.name}-{'-'.join(subjects)}",
            predicate=lambda sample: sample.metadata is not None
            and sample.metadata.get("subject") in subjects,
        )
    else:
        return dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["choices"],
        # converts 0 -> A, 1 -> B, etc.
        target=("ABCD"[record["answer"]]),
        metadata={"subject": record["subject"]},
    )


@solver
def random_choice_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:

        # print(state.choices._choices)

        # At random set random.choicestate.choices._choices
        # Ensure there are choices available
        if state.choices and len(state.choices) > 0:

            print(random.choice(state.choices))
            # Randomly select a choice
            selected = random.choice(state.choices)
            # Add the selected choice as the assistant's answer
            selected.correct = True

        print(state.choices._choices)

        return state

    return solve
