from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.scorer import match
from inspect_ai.model import GenerateConfig
import random
from inspect_ai.dataset import Dataset, MemoryDataset
from typing import Any, Literal, Union
from textwrap import dedent
import re

from inspect_ai._eval.eval import eval
import os
import importlib.util
import uuid
from base import initialize_session
import contextlib

from inspect_ai.scorer import Score, scorer
from inspect_ai.scorer import (
    accuracy,
    stderr,
)  # or any other built-in metrics you'd like


import os
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
    def filter_grid_size(example):
        for g in example.get("train", []) + example.get("test", []):
            grid = g.get("input", {})

            if len(grid) > 5:
                return False
            else:
                for row in grid:
                    if len(row) > 5:
                        return False
        return True

    dataset = dataset.filter(filter_grid_size)

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


class EvaluateARC:

    def __init__(
        self,
        args=None,
        split: Union[
            Literal["test"], Literal["dev"], Literal["validation"]
        ] = "evaluation",
        shuffle: bool = False,
        seed: int = 42,
    ) -> Dataset:

        self.args = args

        # Load dataset
        dataset = easy_hf_dataset(
            path="dataartist/arc-agi",
            name="default",
            split=split,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=seed,
        )

        # Print length before filtering
        print("Original dataset size:", len(dataset))

        self.dataset = dataset

    def _format_grid(self, grid: list[list[int]]) -> str:
        """Helper function to format grid into a string representation."""
        return "\n".join([" ".join(map(str, row)) for row in grid])

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing ARC task data into a Sample object.

        Parameters
        ----------
        record : dict[str, Any]
            A dictionary containing:
            - "train": list[dict] - List of example pairs (input, output).
            - "test": list[dict] - Test input grid.
            - "id": str - Identifier for the task.

        Returns
        -------
        Sample
            A Sample object with:
            - input: The formatted ARC task description.
            - target: The correct output grid as a string (for evaluation purposes).
            - metadata: A dictionary containing the task ID.
        """

        task_id = record["id"]

        # Construct examples section
        examples = []
        for i, example in enumerate(record["train"]):
            input_grid = self._format_grid(example["input"])
            output_grid = self._format_grid(example["output"])
            examples.append(
                f"### Example {i}:\nInput:\n{input_grid}\nOutput:\n{output_grid}\n"
            )

        examples = "\n".join(examples)

        # Construct test section
        test_input = self._format_grid(record["test"][0]["input"])

        # Build the full task prompt
        task_prompt = dedent(
            f"""
            ## Task Overview:
            You are solving an ARC (Abstraction and Reasoning Corpus) task.

            ### Instructions:
            Given some example input and output grids, determine the transformation rule and apply it to predict the output grid for the given test input.

            ## Examples:
            {examples}

            ## Test Problem:
            Input:
            {test_input}

            Analyze the transformation rules based on the provided examples and determine what the output should be for the test problem.
            """
        ).strip()

        print("Task rpompt", task_prompt)

        # Return formatted sample
        return Sample(
            input=task_prompt,
            target=str(
                self._format_grid(record["test"][0]["output"])
            ),  # Expected test output
            metadata={"task_id": task_id},
        )

    @solver
    def match_solver(self, framework) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            session, Base = initialize_session(self.args.db_name)

            # Create the agent framework in temporary code
            current_directory = os.path.dirname(os.path.abspath(__file__))
            parent_directory = os.path.dirname(current_directory)
            cleaned_name = re.sub(r"[^A-Za-z0-9 ]+", "", framework.framework_name)
            temp_file = (
                f"""{parent_directory}/temp/agent_system_temp_"""
                + f"""
                {cleaned_name}_{framework.framework_id}_{uuid.uuid4()}.py""".strip()
            )

            forward_function = framework.framework_code

            if "return self.forward" in forward_function:
                return 0

            try:

                # Write the complete AgentSystem class to the file, including the forward function
                with open(temp_file, "w") as f:
                    f.write("import random\n")
                    f.write("import pandas\n")
                    f.write("import asyncio\n\n")
                    f.write(f"from base import Agent, Meeting, Chat, Wrapper\n\n")
                    f.write(f"from sqlalchemy.orm import Session\n\n")
                    f.write("class AgentSystem:\n")
                    f.write("    def __init__(self, session: Session):\n")
                    f.write("        self.Agent = Wrapper(Agent, session)\n")
                    f.write("        self.Meeting = Wrapper(Meeting, session)\n")
                    f.write("        self.Chat = Wrapper(Chat, session)\n")
                    f.write("        self.session = session\n\n")
                    f.write("    " + forward_function.replace("\n", "\n    "))
                    f.write("\n\n")
                    f.write("if __name__ == '__main__':\n")
                    f.write("    " + "from base import initialize_session\n")
                    f.write("    " + "session, Base = initialize_session()\n")
                    f.write("    " + "agent_system = AgentSystem(session)\n")
                    f.write(
                        "    "
                        + """task = "What should I have for dinner?"""
                        + """A: soup B: burgers C: pizza D: pasta"\n"""
                    )
                    f.write(
                        "    " + "output = asyncio.run(agent_system.forward(task))\n"
                    )
                    f.write("    " + "print(output)\n")

                # Import the AgentSystem class from the temp file
                spec = importlib.util.spec_from_file_location(
                    "agent_system_temp", temp_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AgentSystem = module.AgentSystem

                agentSystem = AgentSystem(session)

                task = state.input

                state.output.completion = await agentSystem.forward(task)

            except Exception as e:

                print("Error during evaluation:", e)

            finally:

                # delete file at the end
                os.remove(temp_file)

                session.close()

            return state

        return solve

    @task
    def match_task(self, framework, i, N):
        return Task(
            name=f"{i} of {N} {framework.framework_name}",
            dataset=self.dataset,
            solver=self.match_solver(framework),
            scorer=self.percentage_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    @scorer(metrics=[accuracy(), stderr()])
    def percentage_match():
        async def score(state, target):
            solution_str = target.text
            solution_grid = EvaluateARC._parse_grid(solution_str)

            prediction_str = state.output.completion
            try:
                prediction_grid = EvaluateARC._parse_grid(prediction_str)
            except Exception:
                prediction_grid = []

            pct = EvaluateARC._get_percentage_match(solution_grid, prediction_grid)

            return Score(
                name="percentage_match",
                value=pct,
                answer=prediction_str,
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
        print("arr1", arr1)
        print("arr2", arr2)

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

    def evaluate(self, framework, i=1, N=1, limit=1000):

        results = eval(
            self.match_task(framework, i, N),
            model="openai/gpt-3.5-turbo",
            limit=limit,
            log_dir="./logs",
            log_format="eval",
            score=True,
        )

        accuracy = -2
        print(results)
        for res in results:
            if res.results and res.results.scores:
                print("Final metrics for the entire dataset:")
                for score in res.results.scores:
                    print(f"Score: {score.name}")
                    for metric_name, metric in score.metrics.items():
                        print(f"  {metric_name}: {metric.value}")
                        if metric_name == "accuracy":
                            accuracy = metric.value

        return accuracy
