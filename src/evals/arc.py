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
from typing import cast
from inspect_ai.scorer import Metric, Score, metric, accuracy, scorer

from inspect_ai._eval.eval import eval

from typing import Any

from .metrics import ci_lower, ci_upper, median
from .benchmark import Benchmark


class ARC(Benchmark):

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:

        self.args = args

        split_mapping = {
            "validation": "training",
            "test": "training",
        }

        # Load dataset
        dataset = self.filtered_hf_dataset(
            path="dataartist/arc-agi",
            name="default",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

        # Print length before filtering
        print("Original dataset size:", len(dataset))

        self.dataset = dataset

    def benchmark_filter(self, example):
        for g in example.get("train", []) + example.get("test", []):
            grid = g.get("input", {})

            if len(grid) > 5:
                return False
            else:
                for row in grid:
                    if len(row) > 5:
                        return False
        return True

    @staticmethod
    def _grid_2_str(grid: list[list[int]]) -> str:
        """Helper function to format grid into a string representation."""
        return "\n".join([" ".join(map(str, row)) for row in grid])

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing ARC task data into a Sample object.
        """

        task_id = record["id"]

        # Construct examples section
        examples = []
        for i, example in enumerate(record["train"]):
            input_grid = ARC._grid_2_str(example["input"])
            output_grid = ARC._grid_2_str(example["output"])
            examples.append(
                f"### Example {i}:\nInput:\n{input_grid}\nOutput:\n{output_grid}\n"
            )

        examples = "\n".join(examples)

        # Construct test section
        test_input = record["test"][0]["input"]

        # Build the full task prompt
        task_prompt = dedent(
            f"""
            ## Task Overview:
            You are solving an ARC (Abstraction and Reasoning Corpus) task.

            ### Instructions:
            Given some example input and output grids, determine the transformation rule and return
            a transformation function which only uses built-in python libraries to perform the
            transformation on the Test Input.

            ## Examples:
            {examples}

            ## Test Input:
            {ARC._grid_2_str(test_input)}

            

            Analyze the transformation rules and Test Input and return the transformation function in the format:
```
def transform(grid: list[list[int]]) -> list[list[int]]:
    # Your code here
    return transformed_grid
```

            Do not simply return the grid as the output and ensure the return value is "transformed_grid". You must provide a function that can transform the input grid in question.
            """
        ).strip()

        # print("Task rpompt", task_prompt)

        # Return formatted sample
        return Sample(
            input=task_prompt,
            target=str(
                ARC._grid_2_str(list(record["test"][0]["output"]))
            ),  # Expected test output
            metadata={"task_id": task_id, "test_input": test_input},
        )

    @task
    def match_task(self):
        return Task(
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.percentage_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def percentage_match():
        async def score(state, target):
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="percentage_match",
                    value=0,
                    answer=state.output.completion,
                    explanation=f"Error in model response.",
                )
            try:

                transformation_code_snippet = state.output.completion

                print("transformation_code_snippet", transformation_code_snippet)

                match = re.search(r"def\s+(\w+)\s*\(", transformation_code_snippet)

                function_name = match.group(1)  # Return the captured function name
                # Dynamically execute the code and extract the function namespace
                namespace = {}
                exec(transformation_code_snippet, namespace)
                transform = namespace[function_name]

                # Run the function on the test input
                test_input = state.metadata["test_input"]
                prediction_grid = transform(test_input)

                target_str = target.text

                prediction_str = str(ARC._grid_2_str(prediction_grid))

                pct = ARC._get_percentage_match(
                    ARC._parse_grid(target_str),
                    ARC._parse_grid(prediction_str),
                )
                # hard match
                if pct == 1.0:
                    pct = 1.0
                else:
                    pct = 0.0
            except Exception as e:
                print("Error during ARC scoring:", e)
                pct = 0.0
                prediction_str = f"Error: {e}"

            return Score(
                name="percentage_match",
                value=pct,
                answer=prediction_str + f"\n Code: {transformation_code_snippet}",
                explanation=f"Percentage match: {pct*100:.2f}%",
            )

        return score

    @staticmethod
    def _parse_grid(grid_str: str) -> list[list[int]]:
        """
        Parse a multi-line string of integers into a 2D list of ints.
        Example:
            "1 0\n1 0" -> [[1, 0], [1, 0]]
        """
        lines = grid_str.strip().split("\n")
        grid = []
        for line in lines:
            row = line.strip().split()
            # convert each token in row to int
            grid.append([int(x) for x in row])
        return grid

    @staticmethod
    def _get_percentage_match(arr1: list[list[int]], arr2: list[list[int]]) -> float:
        """
        Compare two 2D integer arrays for exact matching of each cell.
        Returns a number in [0,1] indicating fraction of cells matching.
        """
        # print("arr1", arr1)
        # print("arr2", arr2)

        # Validate input: check if both arrays are non-empty and rectangular
        def is_valid_grid(arr: list[list[int]]) -> bool:
            return bool(arr) and all(
                isinstance(row, list) and len(row) == len(arr[0]) for row in arr
            )

        if not is_valid_grid(arr1) or not is_valid_grid(arr2):
            return 0.0  # Return 0 if either array is empty or non-rectangular

        # Handle mismatched shapes safely
        rows1, cols1 = len(arr1), len(arr1[0])
        rows2, cols2 = len(arr2), len(arr2[0])

        # Find the common overlapping region
        rows = min(rows1, rows2)
        cols = min(cols1, cols2)

        score = 0
        for i in range(rows):
            for j in range(cols):
                if arr1[i][j] == arr2[i][j]:
                    score += 1

        # Compute fraction of matching cells based on the size of the smaller region
        total = rows * cols

        # Prevent division by zero
        return float(score) / total if total > 0 else 0.0
