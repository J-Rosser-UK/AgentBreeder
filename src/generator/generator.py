import random
from base import (
    System,
    Population,
    initialize_session,
    initialize_session,
)
from prompts.mutation_base import get_init_archive

# from rich import print
from descriptor import Descriptor
from evals import Validator
from prompts.mutation_prompts import multi_agent_system_mutation_prompts
from icecream import ic
from .mutator import Mutator
from evals import Validator
from descriptor import Clusterer
import asyncio
import logging
import datetime


async def generate_mutant(args, population_id, generation_timestamp):
    try:
        session, Base = initialize_session()
        generator = Generator(
            args, generation_timestamp
        )  # Create a new Generator instance per task
        mutant_system = await generator.generate(session, population_id)
    except Exception as e:
        logging.error(f"Error generating mutant: {e}")
        mutant_system = None
        session.rollback()
    finally:
        session.close()
    return mutant_system


# The async part of the logic
async def run_generation(args, population_id):
    generation_timestamp = datetime.datetime.utcnow()
    tasks = [
        generate_mutant(args, population_id, generation_timestamp)
        for _ in range(args.n_mutations)
    ]
    results = await asyncio.gather(*tasks)
    return results


class Generator:

    def __init__(self, args, generation_timestamp) -> None:
        """
        Initializes the Generator class.

        Args:
            args: Arguments object containing configurations for the generator.
            session: The session object for managing database interactions.
            population_id: The ID of the population to operate on.
        """
        self.args = args
        self.mutation_operators = multi_agent_system_mutation_prompts
        self.batch_size = 1
        self.descriptor = Descriptor()
        self.generation_timestamp = generation_timestamp
        self.evaluator = Validator(args)

    async def generate(self, session, population_id) -> System:
        """
        Generates a mutated system from the population by sampling and applying mutations.

        Returns:
            System: The mutated system object. Returns None if the mutation fails.
        """

        # self.crossover = Crossover(args)
        self.population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        print(self.population.population_id)

        self.systems = self.population.systems  # error on this line

        self.mutator = Mutator(
            self.args,
            session,
            self.population,
            self.mutation_operators,
            self.evaluator,
            self.generation_timestamp,
        )

        mutant_system = await self.mutator.mutate()

        if mutant_system:
            mutant_system.update(
                system_descriptor=self.descriptor.generate(mutant_system)
            )

            self.population.systems.append(mutant_system)

        return mutant_system


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of systems for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    try:
        session, Base = initialize_session()

        archive = get_init_archive()

        population = Population(session=session, population_benchmark=args.benchmark)
        descriptor = Descriptor()
        evaluator = Validator(args)
        clusterer = Clusterer()

        generation_timestamp = datetime.datetime.utcnow()

        for system in archive:
            system = System(
                session=session,
                system_name=system["name"],
                system_code=system["code"],
                system_thought_process=system["thought"],
                population=population,
                generation_timestamp=generation_timestamp,
            )
            population.systems.append(system)

        for system in population.systems:
            system.update(system_descriptor=descriptor.generate(system))

        population_id = str(population.population_id)

        systems_for_evaluation = (
            session.query(System)
            .filter_by(population_id=population_id, system_fitness=None)
            .all()
        )

        # evaluator.async_evaluate(illuminated_systems_for_evaluation)
        evaluator.validate(systems_for_evaluation)

        # Re-load the population object in this session
        population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        # Recluster the population
        clusterer.cluster(population)

    except:
        session.rollback()
        raise
    finally:
        # be sure to close it!
        session.close()

    return population_id
