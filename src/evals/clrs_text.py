from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import TaskState
import random
from inspect_ai.dataset import Dataset, MemoryDataset
from typing import Any, Literal, Union
from textwrap import dedent
import numpy as np
import re
import ast
from typing import cast
from inspect_ai.scorer import Metric, Score, metric, accuracy, scorer
from inspect_ai.dataset import Dataset, hf_dataset
from inspect_ai._eval.eval import eval
import os
import json
from pathlib import Path
from typing import Any

from inspect_ai._util.appdirs import inspect_cache_dir
from inspect_ai._util.error import pip_dependency_error
from inspect_ai._util.file import safe_filename
from inspect_ai._util.hash import mm3_hash
from inspect_ai._util.version import verify_required_version

from inspect_ai.dataset._dataset import (
    Dataset,
    FieldSpec,
    MemoryDataset,
    RecordToSample,
)
from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn
from .metrics import ci_lower, ci_upper, median
from .inspect_base import InspectBase
import uuid


class EvaluateCLRS(InspectBase):

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:

        self.args = args

        split_mapping = {
            "validation": "test_1",
            "test": "test_2",
        }

        dataset = easy_hf_dataset(
            path="tomg-group-umd/CLRS-Text-test",
            name="default",
            split=split_mapping[split],
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

        # Print length before filtering
        print("Original dataset size:", len(dataset))

        self.dataset = dataset

    def _parse_question(self, question_str):
        """
        Parses the question string to extract algorithm name, key, initial trace, and other components.
        """
        lines = question_str.strip().split("\n")
        algo_line = lines[0]
        algo_name = algo_line.rstrip(":").strip()

        # Extract key and initial_trace using regex
        key_match = re.search(r"key:\s*\[([^\]]+)\]", question_str)
        initial_trace_match = re.search(r"initial_trace:\s*\[([^\]]+)\]", question_str)

        key = [float(num) for num in key_match.group(1).split()] if key_match else []
        initial_trace = (
            [float(num) for num in initial_trace_match.group(1).split()]
            if initial_trace_match
            else []
        )

        # Extract any other components if necessary
        # For example, 'trace | pred:'

        return {
            "algo_name": algo_name,
            "key": key,
            "initial_trace": initial_trace,
            # Add more fields as needed
        }

    def _parse_answer(self, answer_str):
        """
        Parses the answer string into a list of lists.
        """

        # Split the answer into individual array strings
        array_strings = answer_str.split("],")
        arrays = []
        for array_str in array_strings:
            # Remove any leading/trailing brackets and whitespace
            clean_str = array_str.replace("[", "").replace("]", "").strip()
            # Split into individual numbers and convert to floats
            numbers = [float(num) for num in clean_str.split()]
            arrays.append(numbers)
        return arrays

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing CLRS task data into a Sample object.
        """
        print(record)
        # {'question': 'heapsort:\nkey: [0.077 0.512 0.713 0.055 0.905 0.639], initial_trace: [0.077 0.512 0.713 0.055 0.905 0.639]\ntrace | pred:\n', 'answer': '[0.077 0.512 0.713 0.055 0.905 0.639], [0.077 0.512 0.713 0.055 0.905 0.639], [0.077 0.512 0.713 0.055 0.905 0.639], [0.077 0.512 0.713 0.055 0.905 0.639], [0.077 0.905 0.713 0.055 0.512 0.639], [0.077 0.905 0.713 0.055 0.512 0.639], [0.905 0.077 0.713 0.055 0.512 0.639], [0.905 0.512 0.713 0.055 0.077 0.639], [0.905 0.512 0.713 0.055 0.077 0.639], [0.639 0.512 0.713 0.055 0.077 0.905], [0.713 0.512 0.639 0.055 0.077 0.905], [0.713 0.512 0.639 0.055 0.077 0.905], [0.077 0.512 0.639 0.055 0.713 0.905], [0.639 0.512 0.077 0.055 0.713 0.905], [0.639 0.512 0.077 0.055 0.713 0.905], [0.055 0.512 0.077 0.639 0.713 0.905], [0.512 0.055 0.077 0.639 0.713 0.905], [0.512 0.055 0.077 0.639 0.713 0.905], [0.077 0.055 0.512 0.639 0.713 0.905], [0.077 0.055 0.512 0.639 0.713 0.905], [0.055 0.077 0.512 0.639 0.713 0.905] | [0.055 0.077 0.512 0.639 0.713 0.905]\n\n', 'algo_name': 'heapsort', 'length': 6}

        parsed_question = self._parse_question(record["question"])

        trace = self._parse_answer(record["answer"].split("|")[0].strip())
        answer = [
            ast.literal_eval(record["answer"].split("|")[1].strip().replace(" ", ", "))
        ]
        parsed_answer = trace + answer
        # Build the full task prompt
        task_prompt = dedent(
            f"""
            ## Task Overview:
            You are solving a CLRS-text reasoning task based on the {parsed_question['algo_name']} algorithm.

            ### Problem Description:
            - **Algorithm**: {parsed_question['algo_name']}
            - **Key (Values to Sort)**: {parsed_question['key']}
            - **Initial Trace (Starting State)**: {parsed_question['initial_trace']}

            ### Trace Steps:
            Please provide the step-by-step trace of the {parsed_question['algo_name']} algorithm applied to the given key and initial trace.

            ### Expected Output Format: "[[]]" (String of List of Lists)

            ### Instructions:
            Analyze the initial trace using the {parsed_question['algo_name']} algorithm and provide the sequence of states (trace) as the array is being sorted.
            """
        ).strip()

        print("task_prompt", task_prompt)

        print("final answer", answer)

        # Return formatted sample
        return Sample(
            input=task_prompt,
            target=str(parsed_answer),
            metadata={
                "task_id": str(uuid.uuid4()),
                "algo_name": record["algo_name"],
                "length": record["length"],
            },
        )

    @task
    def match_task(self, system, i, N):
        return Task(
            name=f"{i} of {N} {system.system_name}",
            dataset=self.dataset,
            solver=self.match_solver(system),
            scorer=self.trace_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def trace_match():
        async def score(state, target):

            try:
                answer_trace = ast.literal_eval(state.output.completion)
                target_trace = ast.literal_eval(target.text)
            except Exception as e:
                return Score(
                    name="llm_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation=f"Error: {e}",
                )
            if not isinstance(answer_trace, list) or not isinstance(target_trace, list):
                return Score(
                    name="llm_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation="Both answer and target should be lists.",
                )
            # Check for exact match
            if answer_trace == target_trace:
                return Score(
                    name="llm_match",
                    value=1.0,
                    answer=state.output.completion,
                    explanation="Exact match with the target trace.",
                )

            # Function to check if `sub` is a subsequence of `main`
            def is_subsequence(sub, main):
                it = iter(main)
                return all(elem in it for elem in sub)

            # Check if answer_trace is a subsequence of target_trace
            subseq = is_subsequence(answer_trace, target_trace)

            # Check if the final element matches
            final_element_matches = (
                len(answer_trace) > 0
                and len(target_trace) > 0
                and answer_trace[-1] == target_trace[-1]
            )

            # Determine the score based on the conditions
            if subseq and final_element_matches and len(answer_trace) >= 2:
                return Score(
                    name="llm_match",
                    value=len(answer_trace) / len(target_trace),
                    answer=state.output.completion,
                    explanation="Subsequence with the correct final element.",
                )
            else:
                return Score(
                    name="llm_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation="Incorrect trace or final element does not match.",
                )

        return score


def easy_hf_dataset(
    path: str,
    split: str,
    name: str | None = None,
    data_dir: str | None = None,
    revision: str | None = None,
    sample_fields: FieldSpec | RecordToSample | None = None,
    auto_id: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    trust: bool = False,
    cached: bool = True,
    **kwargs: Any,
) -> Dataset:

    # Ensure required HuggingFace datasets version
    FEATURE = "Hugging Face Datasets"
    PACKAGE = "datasets"
    VERSION = "2.16.0"
    try:
        import datasets
    except ImportError:
        raise pip_dependency_error(FEATURE, [PACKAGE])
    verify_required_version(FEATURE, PACKAGE, VERSION)

    # Resolve data-to-sample function
    data_to_sample = record_to_sample_fn(sample_fields)

    # Generate cache directory for dataset
    dataset_hash = mm3_hash(f"{path}{name}{data_dir}{split}{kwargs}")
    datasets_cache_dir = inspect_cache_dir("hf_datasets")
    dataset_cache_dir = os.path.join(
        datasets_cache_dir, f"{safe_filename(path)}-{dataset_hash}"
    )

    # Load dataset from cache or HuggingFace Hub
    if os.path.exists(dataset_cache_dir) and cached and revision is None:
        dataset = datasets.load_from_disk(dataset_cache_dir)
    else:
        print(f"Loading dataset {path} from Hugging Face...")
        dataset = datasets.load_dataset(
            path=path,
            name=name,
            data_dir=data_dir,
            split=split,
            revision=revision,
            trust_remote_code=trust,
            **kwargs,
        )
        dataset.save_to_disk(dataset_cache_dir)

    # Filter dataset for grid size <= 5x5
    def filter_algo(example):
        if example.get("algo_name") == "quicksort":
            if example.get("length") <= 4:
                return True
        return False

    dataset = dataset.filter(filter_algo)

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    # Limit if requested
    if limit:
        dataset = dataset.select(range(limit))

    # Return filtered dataset
    return MemoryDataset(
        samples=data_to_samples(dataset.to_list(), data_to_sample, auto_id),
        name=Path(path).stem if Path(path).exists() else path,
        location=path,
    )
