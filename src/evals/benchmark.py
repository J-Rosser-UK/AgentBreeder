from inspect_ai import Task, task
from inspect_ai.dataset import Sample
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
from base import initialize_session
import contextlib
import logging
from inspect_ai.scorer import Score, scorer
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

    @abstractmethod
    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        pass

    async def forward(
        self, forward_function: str, temp_file: str, session, task: str
    ) -> None:
        output = ""
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
                f.write("    " + "for session in initialize_session():\n")
                f.write("    " + "    " + "agent_system = AgentSystem(session)\n")
                f.write(
                    "    "
                    + "    "
                    + """task = "What should I have for dinner?"""
                    + "    "
                    + """A: soup B: burgers C: pizza D: pasta"\n"""
                )
                f.write(
                    "    "
                    + "    "
                    + "output = asyncio.run(agent_system.forward(task))\n"
                )
                f.write("    " + "    " + "print(output)\n")

            # Import the AgentSystem class from the temp file
            spec = importlib.util.spec_from_file_location(
                "agent_system_temp", temp_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AgentSystem = module.AgentSystem

            agentSystem = AgentSystem(session)

            # Set a timeout of 3 minutes (180 seconds)
            output = await asyncio.wait_for(agentSystem.forward(task), timeout=180)
            output = str(output)

        except TimeoutError:
            logging.info(f"Time expired for {temp_file}")
            print(f"Time expired for {temp_file}")

            output = "Time Expired"

        except Exception as e:
            print("Error during evaluation:", e)
            output = f"Error: {str(e)}"
            session.rollback()

            if "sqlite3.OperationalError" in str(e):
                # wait between 10 and 60 seconds
                await asyncio.sleep(random.randint(10, 60))
                try:
                    output = await asyncio.wait_for(
                        agentSystem.forward(task), timeout=180
                    )
                except:
                    print("Double error during evaluation:", e)
                    output = f"Double error: {str(e)}"
                    session.rollback()

        finally:
            # delete file at the end
            session.commit()
            os.remove(temp_file)
            session.close()

        return output

    @solver
    def match_solver(self, system) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            for session in initialize_session():

                # Create the agent system in temporary code
                current_directory = os.path.dirname(os.path.abspath(__file__))
                parent_directory = os.path.dirname(current_directory)
                cleaned_name = re.sub(r"[^A-Za-z0-9 ]+", "", system.system_name)
                temp_file = (
                    f"""{parent_directory}/temp/agent_system_temp_"""
                    + f"""
                    {cleaned_name}_{system.system_id}_{uuid.uuid4()}.py""".strip()
                )

                forward_function = system.system_code

                if "return self.forward" in forward_function:
                    return 0

                state.output.completion = await self.forward(
                    forward_function, temp_file, session, state.input
                )

            return state

        return solve

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def llm_match():
        async def score(state, target):
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
    def match_task(self, system, i, N):
        pass

    def evaluate(self, system, i=1, N=1, limit=10):

        # Run the evaluation while hiding any print outputs
        # with open(os.devnull, "w") as devnull:
        #     with contextlib.redirect_stdout(devnull):
        results = eval(
            self.match_task(system, i, N),
            model="openai/gpt-3.5-turbo",  # this doesn't matter and isn't used
            limit=limit,
            log_dir="./logs",  # specify where logs are stored
            log_format="eval",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
        )

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        accuracy = -2.0
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
                        elif metric_name == "ci_lower":
                            ci_lower = metric.value
                        elif metric_name == "ci_upper":
                            ci_upper = metric.value
                        elif metric_name == "median":
                            median = metric.value

        return accuracy, ci_lower, ci_upper, median

    async def forward_pass(self, next_response_code, temp_file, session):

        sample = random.choice(self.dataset)

        output = await self.forward(
            next_response_code, temp_file, session, sample.input
        )

        # check state.output.completion is a string that doesnt load to a dictionary or list, its literally just a string
        result = None
        try:
            result = ast.literal_eval(output)

        except Exception as e:
            pass

        # If it evaluates successfully, check its type
        if result:
            if isinstance(result, (dict, tuple)):
                raise AgentSystemException(
                    "The final output of the forward function should be a string, not a dictionary, or tuple."
                )

        if "Time Expired" in output:
            raise AgentSystemException(
                "Time expired, please ensure the forward function executes within the time limit of 3 minutes."
            )

        if "def transform" in output:
            try:
                # Attempt to compile the string as Python code
                compiled_code = compile(output, "<string>", "exec")

            except SyntaxError as s:
                print("code error", s, output)
                raise AgentSystemException(
                    "When outputting code as a string, the final output of the forward function should be a well-formed Python code snippet. This code is returning a syntax error {s}."
                )

        return output

    def benchmark_filter(self, example):
        return True

    def filtered_hf_dataset(
        self,
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

        dataset = dataset.filter(self.benchmark_filter)

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
