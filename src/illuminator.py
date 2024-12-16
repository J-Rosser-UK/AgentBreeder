from base import Population, Framework, Cluster
from chat import get_structured_json_response_from_gpt
import logging
from rich import print
from sqlalchemy.orm import joinedload
import asyncio

import asyncio
from base import Population, Framework, Cluster
from chat import get_structured_json_response_from_gpt
import logging
from rich import print
from sqlalchemy.orm import joinedload
import asyncio
from base import initialize_session


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
        self,
        population: Population,
        frameworks_for_evaluation: list[Framework],
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
        session, Base = initialize_session(self.args.db_name)

        generation = population.generations[-1]
        illuminated_frameworks_for_evaluation = []

        print(len(frameworks_for_evaluation))

        # Run the evaluations synchronously by using asyncio.run
        illuminated_frameworks_for_evaluation = asyncio.run(
            self._run_illuminations(session, frameworks_for_evaluation)
        )

        illuminated_frameworks_for_evaluation_ids = [
            str(framework.framework_id)
            for framework in illuminated_frameworks_for_evaluation
        ]

        session.close()

        return illuminated_frameworks_for_evaluation_ids

    async def _illuminate_framework(self, session, framework):
        """
        Evaluate a single framework asynchronously.
        """
        framework.update(framework_fitness=-1)

        if not framework.cluster_id:
            return None

        cluster_frameworks = (
            session.query(Framework).filter_by(cluster_id=framework.cluster_id).all()
        )

        print("Cluster frameworks length: ", len(cluster_frameworks))

        new_framework = {
            "framework_id": framework.framework_id,
            "framework_name": framework.framework_name,
            "framework_thought_process": framework.framework_thought_process,
            "framework_code": framework.framework_code,
        }

        cluster_frameworks_filtered = [
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

        if len(cluster_frameworks_filtered) > 1:
            print("---------------------------")
            print(f"New framework: {new_framework['framework_name']}")
            print(
                f"Cluster frameworks: {[cluster_framework['framework_name'] for cluster_framework in cluster_frameworks_filtered]}"
            )
            print("---------------------------")

        if len(cluster_frameworks_filtered) <= 1:
            return framework

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
                {cluster_frameworks_filtered}

                Here is the new framework:
                {new_framework}

                Please use the following response format:
                {response_format}

                """.strip(),
            },
        ]

        try:
            Y_or_N = await get_structured_json_response_from_gpt(
                messages, response_format, model=self.args.model, temperature=0.5
            )
            if Y_or_N["Will it outperform the other frameworks?"] == "Y":
                return framework
        except Exception as e:
            logging.error(f"Error in illumination: {e}")
            return None

    async def _run_illuminations(self, session, frameworks_for_evaluation):
        tasks = [
            self._illuminate_framework(session, fw) for fw in frameworks_for_evaluation
        ]
        results = await asyncio.gather(*tasks)
        return [fw for fw in results if fw is not None]
