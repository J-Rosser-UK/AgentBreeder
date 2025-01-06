import random
from base import (
    Framework,
    Population,
    initialize_session,
    initialize_session,
)
from prompts.mutation_base import get_init_archive

# from rich import print
from descriptor import Descriptor
from evals import Evaluator
from prompts.ma_mutation_prompts import multi_agent_system_mutation_prompts
from icecream import ic
from .mutator import Mutator
from evals import Evaluator
from descriptor import Clusterer
import asyncio
import logging


async def generate_mutant(args, population_id):
    try:
        session, Base = initialize_session(args.db_name)
        generator = Generator(args)  # Create a new Generator instance per task
        mutant_framework = await generator.generate(session, population_id)
    except Exception as e:
        logging.error(f"Error generating mutant: {e}")
        mutant_framework = None
    finally:
        session.close()
    return mutant_framework


# The async part of the logic
async def run_generation(args, population_id):
    tasks = [generate_mutant(args, population_id) for _ in range(args.n_mutations)]
    results = await asyncio.gather(*tasks)
    return results


class Generator:

    def __init__(self, args) -> None:
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
        self.evaluator = Evaluator(args)

    async def generate(self, session, population_id) -> Framework:
        """
        Generates a mutated framework from the population by sampling and applying mutations.

        Returns:
            Framework: The mutated framework object. Returns None if the mutation fails.
        """

        # self.crossover = Crossover(args)
        self.population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        print(self.population.population_id)

        self.frameworks = self.population.frameworks  # error on this line

        self.mutator = Mutator(
            self.args,
            session,
            self.population,
            self.mutation_operators,
            self.evaluator,
        )

        mutant_framework = await self.mutator.mutate()

        if mutant_framework:
            mutant_framework.update(
                framework_descriptor=self.descriptor.generate(mutant_framework)
            )

            self.population.frameworks.append(mutant_framework)

        return mutant_framework


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of frameworks for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    session, Base = initialize_session(args.db_name)

    archive = get_init_archive()

    population = Population(session=session)
    descriptor = Descriptor()
    evaluator = Evaluator(args)
    clusterer = Clusterer()

    for framework in archive:
        framework = Framework(
            session=session,
            framework_name=framework["name"],
            framework_code=framework["code"],
            framework_thought_process=framework["thought"],
            population=population,
        )
        population.frameworks.append(framework)

    for framework in population.frameworks:
        framework.update(framework_descriptor=descriptor.generate(framework))

    population_id = str(population.population_id)

    frameworks_for_evaluation = (
        session.query(Framework)
        .filter_by(population_id=population_id, framework_fitness=None)
        .all()
    )

    # evaluator.async_evaluate(illuminated_frameworks_for_evaluation)
    evaluator.inspect_evaluate(frameworks_for_evaluation)

    # Re-load the population object in this session
    population = session.query(Population).filter_by(population_id=population_id).one()
    # Recluster the population
    clusterer.cluster(population)

    session.close()

    return population_id
