import os
import importlib.util

from base import Framework
from tqdm import tqdm
from sqlalchemy.orm import Session
from .mmlu import EvaluateMMLU
from textwrap import dedent
import asyncio


class AgentSystemException(Exception):
    """Custom exception for errors in the agent system."""

    pass


class Evaluator:

    def __init__(self, args):
        """
        Initializes the Evaluator class.

        Args:
            args: Arguments object containing configurations for the evaluator, including
            dataset file paths and model settings.
        """
        self.args = args

    def inspect_evaluate(self, frameworks_for_evaluation: list[Framework]):

        for framework in tqdm(frameworks_for_evaluation):
            e = EvaluateMMLU(self.args)
            accuracy = e.evaluate(framework, limit=self.args.n_evals)
            framework.update(
                ci_sample_size=self.args.n_evals,
                framework_fitness=accuracy,
            )

    def run_forward_pass(
        self, forward_function: str, temp_file: str, session: Session
    ) -> None:
        """
        Evaluates a forward function of an agent framework by executing it in a temporary
        environment.

        Args:
            forward_function (str): The forward function code to evaluate.
            temp_file (str): Path to the temporary file where the framework code will be written.

        Raises:
            AgentSystemException: If there is an error during evaluation.
        """

        try:

            # Write the complete AgentSystem class to the file, including the forward function
            with open(temp_file, "w") as f:
                f.write("import random\n")
                f.write("import pandas\n\n")
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
                f.write("    " + "output = agent_system.forward(task)\n")
                f.write("    " + "print(output)\n")

            # Import the AgentSystem class from the temp file
            spec = importlib.util.spec_from_file_location(
                "agent_system_temp", temp_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AgentSystem = module.AgentSystem

            agentSystem = AgentSystem(session)

            task = dedent(
                """
                Answer the following multiple choice question.

                Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

                (A) 0
                (B) 4
                (C) 2
                (D) 6

                Provide your answer as a single letter in the range A-D.
            """
            )

            forward_pass_output = asyncio.run(agentSystem.forward(task))
            print(forward_pass_output)
            if forward_pass_output not in ["A", "B", "C", "D"]:
                raise AgentSystemException(
                    f"Invalid answer format: {forward_pass_output}. Please adjust the prompts and/or framework to return a single letter answer such as A, B, C, or D."
                )

            # delete file at the end
            os.remove(temp_file)

        except Exception as e:
            raise AgentSystemException(f"Error evaluating framework: {e}")
