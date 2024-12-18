from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.scorer import match
from inspect_ai.model import GenerateConfig
import random
from inspect_ai.dataset import Dataset, hf_dataset
from typing import Any, Literal, Union
from textwrap import dedent

# from .quiet_eval import eval

from inspect_ai._eval.eval import eval
import os
import importlib.util
import uuid
from base import initialize_session
import contextlib


class EvaluateMMLU:

    def __init__(
        self,
        args=None,
        split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
        shuffle: bool = False,
        subjects: Union[list[str], str] = [],
    ) -> Dataset:

        self.args = args

        dataset = hf_dataset(
            path="cais/mmlu",
            name="all",
            split=split,
            sample_fields=self._record_to_sample,
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

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
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
    def match_solver(self, framework) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            session, Base = initialize_session(self.args.db_name)

            # Create the agent framework in temporary code
            current_directory = os.path.dirname(os.path.abspath(__file__))
            parent_directory = os.path.dirname(current_directory)
            temp_file = (
                f"""{parent_directory}/temp/agent_system_temp_"""
                + f"""
                {framework.framework_name}_{framework.framework_id}_{uuid.uuid4()}.py""".strip()
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
                    f.write("    " + "session, Base = initialize_session\n")
                    f.write("    " + "agent_system = AgentSystem()\n")
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
            scorer=match(),
            config=GenerateConfig(temperature=0.5),
        )

    def evaluate(self, framework, i=1, N=1, limit=1000):

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
