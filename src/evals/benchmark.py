from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import solver, Solver, TaskState, Generate
import random
from inspect_ai.dataset import Dataset, hf_dataset
from typing import Any, Literal, Union
from textwrap import dedent
import re
import ast
import asyncio
from inspect_ai._eval.eval import eval
import os
import importlib.util
import uuid
from .model import CustomModel, CustomModelAPI
import json
import contextlib
import logging
from inspect_ai.scorer import Score, scorer

from inspect_ai.scorer import accuracy  # or any other built-in metrics you'd like
from chat import get_structured_json_response_from_gpt
from .metrics import ci_lower, ci_upper, median
from abc import ABC, abstractmethod

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

from pathlib import Path
from typing import Any


class AgentSystemException(Exception):
    """Custom exception for errors in the agent system."""

    pass


class Benchmark(ABC):

    def evaluate(self, systems, limit=10):

        # Run the evaluation while hiding any print outputs
        # with open(os.devnull, "w") as devnull:
        #     with contextlib.redirect_stdout(devnull):

        temp_files = []
        models = []
        for system in systems:
            AgentSystem, temp_file = Benchmark.get_callable(
                system.system_id, system.system_name, system.system_code
            )
            temp_files.append(temp_file)

            custom_api = CustomModelAPI(
                model_name=system.system_name + "||" + system.system_id,
                config=GenerateConfig(),  # Example config
                system=system,
                temp_file=temp_file,
                agent_system=AgentSystem,
            )

            models.append(CustomModel(api=custom_api, config=GenerateConfig()))

        from .salad_data import SaladData

        sd = SaladData()
        results = eval(
            [self.match_task(), sd.match_task()],
            model=models,
            limit=limit,
            log_dir=f"./logs/{self.__class__.__name__}-{str(systems[0].population_id)}",  # specify where logs are stored
            log_format="eval",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
        )

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print("Error removing temp file:", e)

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        model_metrics = {}  # dictionary to hold info for each model

        for res in results:

            # 1) Get the model name and task name
            model_name = str(getattr(res.eval, "model", ""))
            task_name = res.eval.task

            # 2) Initialize defaults (or None) for each metric
            accuracy = None
            ci_lower = None
            ci_upper = None
            median = None

            # 3) Check if results and scores exist
            if res.results and res.results.scores:
                for score in res.results.scores:
                    if score.metrics:
                        # 4) For each metric, check if it exists and store its value
                        if "accuracy" in score.metrics:
                            accuracy = score.metrics["accuracy"].value
                        if "ci_lower" in score.metrics:
                            ci_lower = score.metrics["ci_lower"].value
                        if "ci_upper" in score.metrics:
                            ci_upper = score.metrics["ci_upper"].value
                        if "median" in score.metrics:
                            median = score.metrics["median"].value

            # 5) Save the metrics in a dictionary, keyed by the model name
            if not model_metrics.get(model_name):
                model_metrics[model_name] = {task_name: {}}

            if not model_metrics[model_name].get(task_name):
                model_metrics[model_name][task_name] = {}

            model_metrics[model_name][task_name] = {
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "median": median,
            }

        return model_metrics

    @abstractmethod
    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        pass

    @staticmethod
    def get_callable(system_id, system_name, system_code) -> tuple:

        try:
            forward_function = system_code
            # Create the agent system in temporary code
            current_directory = os.path.dirname(os.path.abspath(__file__))
            parent_directory = os.path.dirname(current_directory)
            cleaned_name = re.sub(r"[^A-Za-z0-9 ]+", "", system_name)
            temp_file = (
                f"""{parent_directory}/temp/agent_system_temp_"""
                + f"""
                {cleaned_name}_{system_id}_{uuid.uuid4()}.py""".strip()
            )

            # Write the complete AgentSystem class to the file, including the forward function
            with open(temp_file, "w") as f:
                f.write("import random\n")
                f.write("import pandas\n")
                f.write("import numpy as np\n")
                f.write("import asyncio\n\n")
                f.write(f"from base import Agent, Meeting, Chat\n\n")
                f.write("class AgentSystem:\n")
                f.write("    " + forward_function.replace("\n", "\n    "))
                f.write("\n\n")
                f.write("if __name__ == '__main__':\n")
                f.write("    " + "agent_system = AgentSystem()\n")
                f.write(
                    "    "
                    + """task = "What should I have for dinner?"""
                    + "    "
                    + """A: soup B: burgers C: pizza D: pasta"\n"""
                )
                f.write("    " + "output = asyncio.run(agent_system.forward(task))\n")
                f.write("    " + "print(output)\n")

            # Import the AgentSystem class from the temp file
            spec = importlib.util.spec_from_file_location(
                "agent_system_temp", temp_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AgentSystem = module.AgentSystem

        except Exception as e:
            print("Error during evaluation:", e)
            return None, temp_file

        return AgentSystem, temp_file

    @solver
    def match_solver(self) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            state = await generate(state=state)

            # print("generate", state.output.completion)

            return state

        return solve

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def llm_match():
        async def score(state, target):
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="llm_match",
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

                Did this user correctly answer the question? YES or NO. Here is their answer: {state.output.completion}.
                """,
                },
            ]
            response_format = {
                "thinking": "One sentence of step-by-step reasoning.",
                "is_match": "One word, YES or NO.",
            }

            response = await get_structured_json_response_from_gpt(
                messages,
                response_format,
                model="gpt-4o-mini",
                temperature=0.5,
                retry=0,
            )

            if "yes" in response.get("is_match", "").lower():
                accuracy = 1
            else:
                accuracy = 0

            return Score(
                name="llm_match",
                value=accuracy,
                answer=state.output.completion,
                explanation=f"Is answer correct? {response['is_match']}",
            )

        return score

    @abstractmethod
    @task
    def match_task(self):
        pass

    def benchmark_filter(self, example):
        return True

    def filtered_hf_dataset(
        self,
        path: str,
        split: str,
        split_mapping: dict,
        name: Union[str, list] | None = None,
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
        """
        Sometimes the datasets contain examples that are too challenging to evaluate on.
        This function simplifies the dataset by filtering out examples that are too complex.
        """

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

        if isinstance(name, str):
            name = [name]

        final_dataset_mapping = {"validation": [], "test": []}

        for split_k, split_v in split_mapping.items():
            for n in name:
                # Generate cache directory for dataset
                dataset_hash = mm3_hash(f"{path}{n}{data_dir}{split_v}{kwargs}")
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
                        name=n,
                        data_dir=data_dir,
                        split=split_v,
                        revision=revision,
                        trust_remote_code=trust,
                        **kwargs,
                    )

                    dataset.save_to_disk(dataset_cache_dir)

                dataset = dataset.filter(self.benchmark_filter)

                final_dataset_mapping[split_k].extend(dataset.to_list())

        if list(split_mapping.values())[0] == list(split_mapping.values())[1]:
            print(list(split_mapping.values())[0], list(split_mapping.values())[1])
            # Assign 20% of the validation set to the validation set and 80% to the test set

            validation_save = list(final_dataset_mapping["validation"])
            final_dataset_mapping["validation"] = final_dataset_mapping["validation"][
                : int(len(final_dataset_mapping["validation"]) * 0.2)
            ]

            final_dataset_mapping["test"] = validation_save[
                int(len(validation_save) * 0.2) :
            ]
        print("Validation length", len(final_dataset_mapping["validation"]))
        print("Test length", len(final_dataset_mapping["test"]))

        # assert that no elements in the validation set are in the test set
        for i, element in enumerate(final_dataset_mapping["validation"]):
            if element in final_dataset_mapping["test"]:
                print(f"Element {i} is in both validation and test sets")

        final_dataset = final_dataset_mapping[split]

        if limit:
            # repeat to ensure dataset is at least 'limit' long
            final_dataset = (
                final_dataset * (limit // len(final_dataset))
                + final_dataset[: limit % len(final_dataset)]
            )

            final_dataset = final_dataset[:limit]

        if shuffle:
            random.shuffle(final_dataset)

        # Return filtered dataset
        return MemoryDataset(
            samples=data_to_samples(final_dataset, data_to_sample, auto_id),
            name=Path(path).stem if Path(path).exists() else path,
            location=path,
        )
