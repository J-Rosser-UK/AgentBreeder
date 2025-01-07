from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, Solver, TaskState, Generate
import random
from inspect_ai.dataset import Dataset, hf_dataset
from typing import Any, Literal, Union
from textwrap import dedent
import re
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


class InspectBase(ABC):

    @abstractmethod
    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        pass

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

                # Set a timeout of 3 minutes (180 seconds)
                task = state.input
                state.output.completion = await asyncio.wait_for(
                    agentSystem.forward(task), timeout=180
                )

            except TimeoutError:
                logging.info(f"Time expired for {temp_file}")
                print(f"Time expired for {temp_file}")

                state.output.completion = "Time Expired"

            except Exception as e:
                print("Error during evaluation:", e)
                state.output.completion = f"Error: {str(e)}"

            finally:
                # delete file at the end
                os.remove(temp_file)
                session.close()

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
    def match_task(self, framework, i, N):
        pass

    def evaluate(self, framework, i=1, N=1, limit=10):

        # Run the evaluation while hiding any print outputs
        # with open(os.devnull, "w") as devnull:
        #     with contextlib.redirect_stdout(devnull):
        results = eval(
            self.match_task(framework, i, N),
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
