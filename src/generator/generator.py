import random
from base import (
    System,
    Population,
    initialize_session,
)


# from rich import print
from descriptor import Descriptor
from evals import Validator
from prompts.mutation_prompts import multi_agent_system_mutation_prompts
from icecream import ic
from .mutator import Mutator
from evals import Validator
import json
import asyncio
import logging
import datetime

from prompts.meta_agent_base import get_base_prompt_with_archive


class Generator:

    def __init__(self, args, population) -> None:
        """
        Initializes the Generator class.

        Args:
            args: Arguments object containing configurations for the generator.

            population_id: The ID of the population to operate on.
        """
        self.args = args
        self.population = population
        self.mutation_operators = multi_agent_system_mutation_prompts
        self.batch_size = 1
        self.descriptor = Descriptor()

        self.validator = Validator(args)
        self.base_prompt = None
        self.base_prompt_response_format = None

    async def generate_mutant(
        self,
        parents,
    ):

        try:
            mutator = Mutator(
                self.args,
                self.mutation_operators,
                self.validator,
                self.base_prompt,
                self.base_prompt_response_format,
            )
            # Create a new Generator instance per task
            mutant_system = await mutator.mutate(parents)

        except Exception as e:
            logging.error(f"Error generating mutant: {e}")
            mutant_system = None

        print("MUTANT", mutant_system)
        return mutant_system

    # The async part of the logic
    async def run_generation(self, session):

        print(self.population.population_id)

        parents = []

        for _ in range(self.args.n_mutations):

            system_1 = random.choice(self.population.elites).to_dict()

            system_2 = random.choice(self.population.elites).to_dict()
            parents.append((system_1, system_2))

        self.base_prompt, self.base_prompt_response_format = (
            get_base_prompt_with_archive(self.args, session)
        )

        generation_timestamp = datetime.datetime.utcnow()
        tasks = [
            self.generate_mutant(
                parents[i],
            )
            for i in range(self.args.n_mutations)
        ]
        results = await asyncio.gather(*tasks)
        print(results)

        for system in results:
            if system:

                system = System(
                    session=session,
                    system_name=system["system_name"],
                    system_code=system["system_code"],
                    system_thought_process=system["system_thought_process"],
                    system_first_parent_id=system["system_first_parent_id"],
                    system_second_parent_id=system["system_second_parent_id"],
                    population=self.population,
                    generation_timestamp=generation_timestamp,
                )
                self.population.systems.append(system)

                system.update(system_descriptor=self.descriptor.generate(system))

        return results
