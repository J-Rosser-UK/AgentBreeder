from base import Population, System, Cluster
from chat import get_structured_json_response_from_gpt
import logging
from rich import print
from sqlalchemy.orm import joinedload
import asyncio

import asyncio
from base import Population, System, Cluster
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
        systems_for_evaluation: list[System],
    ):
        """
        Evaluates systems in a population to determine which surrogates should
        proceed to further evaluation.

        Args:
            population (Population): The population object containing the current
            generation and associated systems.
            systems_for_evaluation (System): A list of systems to be illuminated
            for evaluation.

        Returns:
            list[System]: A list of systems deemed suitable for further evaluation
            based on the illumination process.
        """
        session, Base = initialize_session(self.args.db_name)

        generation = population.generations[-1]
        illuminated_systems_for_evaluation = []

        print(len(systems_for_evaluation))

        # Run the evaluations synchronously by using asyncio.run
        illuminated_systems_for_evaluation = asyncio.run(
            self._run_illuminations(session, systems_for_evaluation)
        )

        illuminated_systems_for_evaluation_ids = [
            str(system.system_id) for system in illuminated_systems_for_evaluation
        ]

        session.close()

        return illuminated_systems_for_evaluation_ids

    async def _illuminate_system(self, session, system):
        """
        Evaluate a single system asynchronously.
        """
        system.update(system_fitness=-1)

        if not system.cluster_id:
            return None

        cluster_systems = (
            session.query(System)
            .filter_by(cluster_id=system.cluster_id)
            .order_by(System.system_fitness.desc())
            .all()
        )[:5]

        print("Cluster systems length: ", len(cluster_systems))

        new_system = {
            "system_id": system.system_id,
            "system_name": system.system_name,
            "system_thought_process": system.system_thought_process,
            "system_code": system.system_code,
        }

        cluster_systems_filtered = [
            {
                "system_id": fw.system_id,
                "system_name": fw.system_name,
                "system_thought_process": fw.system_thought_process,
                "system_code": fw.system_code,
                "system_fitness": fw.system_fitness,
            }
            for fw in cluster_systems
            if str(fw.system_id) != str(system.system_id)
        ]

        if len(cluster_systems_filtered) > 1:
            print("---------------------------")
            print(f"New system: {new_system['system_name']}")
            print(
                f"Cluster systems: {[(cluster_system['system_name'], cluster_system['system_fitness']) for cluster_system in cluster_systems_filtered]}"
            )
            print("---------------------------")

        if len(cluster_systems_filtered) <= 1:
            return system

        response_format = {
            "thinking": "Your step by step thinking.",
            "Will it outperform the other systems?": "A single letter, Y or N.",
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
                Given the following multi-agent systems defined in code, do you think
                there is a 50% or greater chance that the new system will outperform
                or equally perform against the best of them in a simulated environment?

                Here are the top 5 multi-agent systems in the cluster:
                {cluster_systems_filtered}

                Here is the new system:
                {new_system}

                Please use the following response format:
                {response_format}

                """.strip(),
            },
        ]
        Y_or_N = "Error before generated"
        try:
            Y_or_N = await get_structured_json_response_from_gpt(
                messages, response_format, model=self.args.model, temperature=0.5
            )
            if Y_or_N["Will it outperform the other systems?"] == "Y":
                return system
        except Exception as e:
            logging.error(f"Error in illumination: {e}, Y_or_N: {Y_or_N}")
            return None

    async def _run_illuminations(self, session, systems_for_evaluation):
        tasks = [self._illuminate_system(session, fw) for fw in systems_for_evaluation]
        results = await asyncio.gather(*tasks)
        return [fw for fw in results if fw is not None]
