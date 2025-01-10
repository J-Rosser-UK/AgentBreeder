import re
import ast
import uuid
import logging
from textwrap import dedent
from typing import Any, Literal, Union

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, accuracy, scorer

from .metrics import ci_lower, ci_upper, median
from .benchmark import Benchmark


class CLRSText(Benchmark):
    """
    Class for evaluating CLRS-text tasks. Loads a filtered dataset and provides
    functionality to parse questions, answers, and prepare them as samples.
    """

    def __init__(
        self,
        args: Any = None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ):
        """
        Initialize the CLRSText class.

        Args:
            args (Any, optional): Additional arguments or configurations.
            split (Union[Literal["validation"], Literal["test"]], optional):
                Which dataset split to load ("validation" or "test"). Defaults to "validation".
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            limit (int, optional): Maximum number of records to load. Defaults to 1000.
        """
        self.args = args
        split_mapping = {
            "validation": "test_1",
            "test": "test_2",
        }

        dataset = self.filtered_hf_dataset(
            path="tomg-group-umd/CLRS-Text-test",
            name="default",
            split=split_mapping[split],
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed if self.args else None,
            limit=limit,
        )

        logging.info(f"Dataset size: {len(dataset)}")
        self.dataset = dataset

    def benchmark_filter(self, example: dict[str, Any]) -> bool:
        """
        Example filter for dataset entries used in benchmarking.
        Filters records with algo_name == 'quicksort' and length <= 4.

        Args:
            example (dict[str, Any]): A record from the dataset.

        Returns:
            bool: True if record passes the filter, False otherwise.
        """
        if example.get("algo_name") == "quicksort":
            if example.get("length") <= 10:
                return True
        return False

    def _parse_question(self, question_str: str) -> dict[str, Any]:
        """
        Parse the question string to extract the algorithm name, key, and initial trace.

        Args:
            question_str (str): The question text containing algo_name, key, and initial trace.

        Returns:
            dict[str, Any]: A dictionary with keys 'algo_name', 'key', and 'initial_trace'.
        """
        lines = question_str.strip().split("\n")
        algo_line = lines[0]
        algo_name = algo_line.rstrip(":").strip()

        # Use regex to capture the key and initial trace
        key_match = re.search(r"key:\s*\[([^\]]+)\]", question_str)
        initial_trace_match = re.search(r"initial_trace:\s*\[([^\]]+)\]", question_str)

        key = [float(num) for num in key_match.group(1).split()] if key_match else []
        initial_trace = (
            [float(num) for num in initial_trace_match.group(1).split()]
            if initial_trace_match
            else []
        )

        return {
            "algo_name": algo_name,
            "key": key,
            "initial_trace": initial_trace,
        }

    def _parse_answer(self, answer_str: str) -> list[list[float]]:
        """
        Parse the answer string into a list of lists of floats.

        Args:
            answer_str (str): The raw answer string (e.g. "[0.1 0.2], [0.3 0.4]").

        Returns:
            list[list[float]]: Parsed list of lists of floats.
        """
        # Split the answer into separate array segments
        array_strings = answer_str.split("],")
        arrays = []
        for array_str in array_strings:
            # Clean brackets/spaces, split, and convert to float
            clean_str = array_str.replace("[", "").replace("]", "").strip()
            numbers = [float(num) for num in clean_str.split()]
            arrays.append(numbers)
        return arrays

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record to a Sample object with a structured prompt and target.

        Args:
            record (dict[str, Any]): The raw record from the dataset.

        Returns:
            Sample: An inspect_ai Sample with input prompt, target, and metadata.
        """
        # Extract question info
        parsed_question = self._parse_question(record["question"])

        # The 'answer' field is split by '|' to separate the trace from final steps
        trace_part, final_part = record["answer"].split("|")
        trace = self._parse_answer(trace_part.strip())

        # Parse final answer step(s) as a Python literal, adjusting formatting
        final_parsed = ast.literal_eval(final_part.strip().replace(" ", ", "))
        parsed_answer = trace + [final_parsed]

        # Prepare the prompt
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

            ### Two-Shot Examples (from test_5):
            Initial trace: [0.426, 0.178, 0.964, 0.423, 0.343, 0.824, 0.558, 0.952]
            Expected output: [[0.426, 0.178, 0.964, 0.423, 0.343, 0.824, 0.558, 0.952], [0.426, 0.178, 0.964, 0.423, 0.343, 0.824, 0.558, 0.952], [0.426, 0.178, 0.423, 0.964, 0.343, 0.824, 0.558, 0.952], [0.426, 0.178, 0.423, 0.343, 0.964, 0.824, 0.558, 0.952], [0.426, 0.178, 0.423, 0.343, 0.824, 0.964, 0.558, 0.952], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.964, 0.952], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.824, 0.558, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.558, 0.824, 0.952, 0.964], [0.426, 0.178, 0.423, 0.343, 0.558, 0.824, 0.952, 0.964], [0.178, 0.426, 0.423, 0.343, 0.558, 0.824, 0.952, 0.964], [0.178, 0.426, 0.423, 0.343, 0.558, 0.824, 0.952, 0.964], [0.178, 0.343, 0.423, 0.426, 0.558, 0.824, 0.952, 0.964], [0.178, 0.343, 0.423, 0.426, 0.558, 0.824, 0.952, 0.964], [0.178, 0.343, 0.423, 0.426, 0.558, 0.824, 0.952, 0.964]]

            Initial trace: [0.626, 0.83, 0.081, 0.745, 0.09]
            Expected output: [[0.626, 0.83, 0.081, 0.745, 0.09], [0.081, 0.83, 0.626, 0.745, 0.09], [0.081, 0.83, 0.626, 0.745, 0.09], [0.081, 0.09, 0.626, 0.745, 0.83], [0.081, 0.09, 0.626, 0.745, 0.83], [0.081, 0.09, 0.626, 0.745, 0.83], [0.081, 0.09, 0.626, 0.745, 0.83], [0.081, 0.09, 0.626, 0.745, 0.83], [0.081, 0.09, 0.626, 0.745, 0.83]]
            
            ### Expected Output Format: "[[]]" (String of List of Lists)

            ### Instructions:
            Analyze the initial trace using the {parsed_question['algo_name']} algorithm 
            and provide the sequence of states (trace) in the expected output format
            as the array is being sorted.
            """
        ).strip()

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
    def match_task(self, system: Any, i: int, N: int) -> Task:
        """
        Create a Task object for scoring traces using the 'trace_match' scorer.

        Args:
            system (Any): The system or solver to be used.
            i (int): Current index in a batch of tasks.
            N (int): Total number of tasks in the batch.

        Returns:
            Task: An inspect_ai Task that will be run with the specified scorer and config.
        """
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
        """
        Custom scorer function that checks whether the model's output trace matches
        the target trace, either exactly or as a subsequence with a correct final element.

        Returns:
            Callable: A coroutine that performs scoring and returns a Score object.
        """

        async def score(state, target):
            try:
                answer_trace = ast.literal_eval(state.output.completion)
                target_trace = ast.literal_eval(target.text)
            except Exception as e:
                return Score(
                    name="trace_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation=f"Error: {e}",
                )

            # Ensure both answer and target are lists
            if not isinstance(answer_trace, list) or not isinstance(target_trace, list):
                return Score(
                    name="trace_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation="Both answer and target should be lists.",
                )

            # Exact match check
            if answer_trace == target_trace:
                return Score(
                    name="trace_match",
                    value=1.0,
                    answer=state.output.completion,
                    explanation="Exact match with the target trace.",
                )

            longest_subseq = CLRSText.longest_consecutive_subsequence(
                answer_trace, target_trace
            )

            # Check final element correctness
            final_element_matches = (
                len(answer_trace) > 0
                and len(target_trace) > 0
                and answer_trace[-1] == target_trace[-1]
            )

            # Score based on subsequence presence and final element
            if final_element_matches and len(answer_trace) >= 2:

                rouge_s_precision = len(longest_subseq) / len(answer_trace)
                rouge_s_recall = len(longest_subseq) / len(target_trace)

                return Score(
                    name="trace_match",
                    value=2
                    * (rouge_s_precision * rouge_s_recall)
                    / (rouge_s_precision + rouge_s_recall),
                    answer=state.output.completion,
                    explanation="Subsequence with the correct final element.",
                )

            return Score(
                name="trace_match",
                value=0.0,
                answer=state.output.completion,
                explanation="Incorrect trace or final element does not match.",
            )

        return score

    @staticmethod
    def longest_consecutive_subsequence(sub, main):
        """
        Find the longest consecutive subsequence in 'sub' that is a subsequence of 'main'.

        Parameters:
        - sub: List of elements representing the subsequence to check.
        - main: List of elements representing the main sequence.

        Returns:
        - The longest consecutive subsequence from 'sub' that is a subsequence of 'main'.
        Returns an empty list if no such subsequence exists.
        """
        max_len = 0
        longest_sub = []

        def is_subsequence(s, m):
            """
            Check if 'sub' is a subsequence of 'main'.
            """
            it = iter(m)
            return all(elem in it for elem in s)

        # Iterate over all possible start indices in 'sub'
        for start in range(len(sub)):
            # Iterate over possible end indices, starting from the end of 'sub' down to 'start'
            for end in range(len(sub), start, -1):
                # Current candidate subsequence
                candidate = sub[start:end]

                # If the candidate length is less than or equal to current max, no need to check further
                if len(candidate) <= max_len:
                    break  # Optimization: no longer subsequences possible from this start

                # Check if the candidate is a subsequence of 'main'
                if is_subsequence(candidate, main):
                    # Update max_len and longest_sub if a longer subsequence is found
                    if len(candidate) > max_len:
                        max_len = len(candidate)
                        longest_sub = candidate
                    break  # No need to check shorter subsequences from this start

        return longest_sub

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def sorted_array_match():
        """
        Custom scorer function that checks whether the model's output trace matches
        the target trace, either exactly or as a subsequence with a correct final element.

        Returns:
            Callable: A coroutine that performs scoring and returns a Score object.
        """

        async def score(state, target):
            try:
                answer_trace = ast.literal_eval(state.output.completion)
                target_trace = ast.literal_eval(target.text)
            except Exception as e:
                return Score(
                    name="trace_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation=f"Error: {e}",
                )

            # Ensure both answer and target are lists
            if not isinstance(answer_trace, list) or not isinstance(target_trace, list):
                return Score(
                    name="trace_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation="Both answer and target should be lists.",
                )

            # Exact match check
            if answer_trace[-1] == target_trace[-1]:
                return Score(
                    name="trace_match",
                    value=1.0,
                    answer=state.output.completion,
                    explanation="Exact match with the target trace.",
                )

            return Score(
                name="trace_match",
                value=0.0,
                answer=state.output.completion,
                explanation="Incorrect trace or final element does not match.",
            )

        return score
