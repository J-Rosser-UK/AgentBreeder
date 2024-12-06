import os
import random
import pandas

import importlib.util
from pydantic import BaseModel
import uuid
import logging
from base import Framework, initialize_session
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sqlalchemy.orm import Session


class MultipleChoiceQuestion(BaseModel):
    question_id: uuid.UUID
    question: str
    A: str
    B: str
    C: str
    D: str
    correct_answer_letter: str
    subject: str


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
        self.multiple_choice_questions = self.load_eval_dataset()

    def load_eval_dataset(self) -> list[MultipleChoiceQuestion]:
        """
        Loads and prepares a dataset of multiple-choice questions for evaluation.

        Returns:
            list[MultipleChoiceQuestion]: A list of formatted multiple-choice question objects.
        """

        df = pandas.read_csv(self.args.data_filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(examples)

        multiple_choice_questions = []
        for example in examples:
            multiple_choice_questions.append(
                MultipleChoiceQuestion(
                    question_id=uuid.uuid4(),
                    question=example["Question"],
                    A=example["A"],
                    B=example["B"],
                    C=example["C"],
                    D=example["D"],
                    correct_answer_letter=example["Answer"],
                    subject=example["Subject"],
                )
            )

        return multiple_choice_questions

    def format_question(self, multiple_choice_question: MultipleChoiceQuestion) -> str:
        """
        Formats a multiple-choice question into a prompt string.

        Args:
            multiple_choice_question (MultipleChoiceQuestion): The question object to format.

        Returns:
            str: A formatted string containing the multiple-choice question and options.
        """

        QUERY_TEMPLATE_MULTICHOICE = """
        Answer the following multiple choice question.

        <<Question>>

        (A) <<A>>
        (B) <<B>>
        (C) <<C>>
        (D) <<D>>
        """.strip()

        # Start by replacing the question
        prompt = QUERY_TEMPLATE_MULTICHOICE.replace(
            "<<Question>>", multiple_choice_question.question
        )

        # Replace each option placeholder iteratively
        for letter in ["A", "B", "C", "D"]:
            prompt = prompt.replace(
                f"<<{letter}>>", getattr(multiple_choice_question, letter)
            )

        return prompt

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

            mcq = MultipleChoiceQuestion(
                question_id=uuid.uuid4(),
                question="Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
                A="0",
                B="4",
                C="2",
                D="6",
                correct_answer_letter="B",
                subject="maths",
            )

            task = self.format_question(mcq)
            forward_pass_output = agentSystem.forward(task)
            print(forward_pass_output)
            if forward_pass_output not in ["A", "B", "C", "D"]:
                raise AgentSystemException(
                    f"Invalid answer format: {forward_pass_output}. Please adjust the prompts and/or framework to return a single letter answer such as A, B, C, or D."
                )

            # delete file at the end
            os.remove(temp_file)

        except Exception as e:
            raise AgentSystemException(f"Error evaluating framework: {e}")

    def async_evaluate(self, frameworks_for_evaluation: list[Framework]):
        """
        Asynchronously evaluates a batch of frameworks by comparing their forward functions
        against a dataset of questions.

        Args:
            frameworks_for_evaluation (list[Framework]): A list of frameworks to evaluate.
        """

        framework_id_map = {}
        framework_question_pairs: list[dict] = []
        for framework in frameworks_for_evaluation:
            framework.update(framework_fitness=-1)
            for n in range(5):
                framework_question_pairs.append(
                    {
                        "framework_id": framework.framework_id,
                        "framework_name": framework.framework_name,
                        "forward_function": framework.framework_code,
                        "multiple_choice_question": random.choice(
                            self.multiple_choice_questions
                        ),
                        "result": None,
                    }
                )
                framework_id_map[framework.framework_id] = framework

        # use concurrent futures to multithread this and return updated framework_question_pairs
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self._thread_eval, pair)
                for pair in framework_question_pairs
            ]
            for future in tqdm(futures, desc="Evaluating Async Frameworks"):
                pair = future.result()

        framework_results_pairs = {}

        for pair in framework_question_pairs:
            framework_id = pair["framework_id"]
            if framework_id not in framework_results_pairs:
                framework_results_pairs[framework_id] = []
            framework_results_pairs[framework_id].append(pair["result"])

        # Update the frameworks with the results
        for framework_id, results_list in framework_results_pairs.items():
            framework = framework_id_map[framework_id]

            confidence_level = 0.95
            confidence_interval_string, ci_lower, ci_upper, median = (
                self.bootstrap_confidence_interval(
                    results_list, confidence_level=confidence_level
                )
            )

            framework.update(
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_median=median,
                ci_sample_size=len(results_list),
                ci_confidence_level=confidence_level,
                framework_fitness=median,
            )

            logging.info(
                f"""Framework: {framework.framework_name},
                Confidence Interval: {confidence_interval_string}"""
            )

    def _thread_eval(self, pair):
        """
        Evaluates a single framework and multiple-choice question pair in a thread.

        Args:
            pair (dict): A dictionary containing framework details, forward function, and
            the question to evaluate.
        """

        session, Base = initialize_session()

        # Create the agent framework in temporary code
        current_directory = os.path.dirname(os.path.abspath(__file__))
        temp_file = f"""
            {current_directory}/temp/agent_system_temp_
            {pair['framework_name']}_{pair['framework_id']}_{uuid.uuid4()}.py""".strip()

        result = self.evaluate_forward_function_on_one_question(
            session,
            pair["multiple_choice_question"],
            pair["forward_function"],
            temp_file,
        )

        pair["result"] = result

        # delete file at the end
        os.remove(temp_file)

        session.close()

    def evaluate_forward_function_on_one_question(
        self, session, multiple_choice_question, forward_function, temp_file
    ) -> int:
        """
        Evaluates the forward function of a framework on a single multiple-choice question.

        Args:
            session: Database session for framework interactions.
            multiple_choice_question (MultipleChoiceQuestion): The question object to evaluate.
            forward_function (str): The forward function code to test.
            temp_file (str): Path to the temporary file where the framework code will be written.

        Returns:
            int: 1 if the framework's forward function produces the correct answer, 0 otherwise.
        """

        if "return self.forward" in forward_function:
            return 0

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

            task = self.format_question(multiple_choice_question)
            agent_framework_answer = agentSystem.forward(task)

            if agent_framework_answer == multiple_choice_question.correct_answer_letter:
                return 1
            else:
                return 0

        except Exception:
            return 0

    def bootstrap_confidence_interval(
        self, data, num_bootstrap_samples=100000, confidence_level=0.95
    ):
        """
        Calculates the bootstrap confidence interval for the mean accuracy of framework evaluations.

        Args:
            data (list or array): 1D list or array of accuracy data points.
            num_bootstrap_samples (int): Number of bootstrap samples to generate.
            confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).

        Returns:
            tuple: A formatted string of the confidence interval and median, along with lower and upper bounds of the interval and the median value.
        """
        # Convert data to a numpy array for easier manipulation
        data = np.array(data)

        # List to store the means of bootstrap samples
        bootstrap_means = []

        # Generate bootstrap samples and compute the mean for each sample
        for _ in range(num_bootstrap_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            # Compute the mean of the bootstrap sample
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_means.append(bootstrap_mean)

        # Convert bootstrap_means to a numpy array for percentile calculation
        bootstrap_means = np.array(bootstrap_means)

        # Compute the lower and upper percentiles for the confidence interval
        lower_percentile = (1.0 - confidence_level) / 2.0
        upper_percentile = 1.0 - lower_percentile
        ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
        ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

        # Compute the median of the bootstrap means
        median = np.median(bootstrap_means)

        # Convert to percentages and format to one decimal place
        ci_lower_percent = ci_lower * 100
        ci_upper_percent = ci_upper * 100
        median_percent = median * 100

        # Return the formatted string with confidence interval and median
        confidence_interval_string = f"""
            95% Bootstrap Confidence Interval: (
            {ci_lower_percent:.1f}%,
            {ci_upper_percent:.1f}%),
            Median: {median_percent:.1f}%
        """.strip()

        return confidence_interval_string, ci_lower, ci_upper, median
