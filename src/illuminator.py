from base import Population, Framework, Cluster
from chat import get_structured_json_response_from_gpt
import logging
from rich import print
from sqlalchemy.orm import joinedload


class Illuminator:

    def __init__(self, args):
        """
        Initializes the Illuminator class.

        Args:
            args: Arguments object containing configurations for the illuminator,
            including model settings.
        """

        self.args = args

    def illuminate(
        self, session, population: Population, frameworks_for_evaluation: Framework
    ):
        """
        Evaluates frameworks in a population to determine which surrogates should
        proceed to further evaluation.

        Args:
            population (Population): The population object containing the current
            generation and associated frameworks.
            frameworks_for_evaluation (Framework): A list of frameworks to be illuminated
            for evaluation.

        Returns:
            list[Framework]: A list of frameworks deemed suitable for further evaluation
            based on the illumination process.
        """

        generation = population.generations[-1]

        illuminated_frameworks_for_evaluation = []

        print(len(frameworks_for_evaluation))

        for framework in frameworks_for_evaluation:

            if not framework.cluster_id:
                continue

            cluster_frameworks = (
                session.query(Framework)
                .filter_by(cluster_id=framework.cluster_id)
                .all()
            )

            print("Cluster frameworks length: ", len(cluster_frameworks))

            new_framework = {
                "framework_id": framework.framework_id,
                "framework_name": framework.framework_name,
                "framework_thought_process": framework.framework_thought_process,
                "framework_code": framework.framework_code,
            }

            cluster_frameworks: list[Framework] = [
                {
                    "framework_id": fw.framework_id,
                    "framework_name": fw.framework_name,
                    "framework_thought_process": fw.framework_thought_process,
                    "framework_code": fw.framework_code,
                    "framework_fitness": fw.framework_fitness,
                }
                for fw in cluster_frameworks
                if str(fw.framework_id) != str(framework.framework_id)
            ]
            if len(cluster_frameworks) > 1:
                print("---------------------------")
                print(f"New framework: {new_framework['framework_name']}")
                print(
                    f"Cluster frameworks: {[cluster_framework['framework_name'] for cluster_framework in cluster_frameworks]}"
                )
                print("---------------------------")

            if len(cluster_frameworks) <= 1:
                illuminated_frameworks_for_evaluation.append(framework)
                continue

            response_format = {
                "thinking": "Your step by step thinking.",
                "Will it outperform the other frameworks?": "A single letter, Y or N.",
            }

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant. Make sure to return in a
                    WELL-FORMED JSON object.""",
                },
                {
                    "role": "user",
                    "content": f"""
                    Given the following multi-agent frameworks defined in code, do you think
                    there is a 50% or greater chance that the new framework will outperform
                    or equally perform against the best of them in a simulated environment?

                    Here are the frameworks in the cluster:
                    {cluster_frameworks}

                    Here is the new framework:
                    {new_framework}

                    Please use the following response format:
                    {response_format}

                    """.strip(),
                },
            ]

            try:
                Y_or_N = get_structured_json_response_from_gpt(
                    messages, response_format, model=self.args.model, temperature=0.5
                )
                if Y_or_N["Will it outperform the other frameworks?"] == "Y":
                    illuminated_frameworks_for_evaluation.append(framework)
            except Exception as e:
                logging.error(f"Error in illumination: {e}")
                continue

        return illuminated_frameworks_for_evaluation
