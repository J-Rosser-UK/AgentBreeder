import random
from base import (
    System,
    Population,
    initialize_session,
)
from prompts.initial_population import (
    COT,
    COT_SC,
    Reflexion,
    LLM_debate,
    Take_a_step_back,
    QD,
    Role_Assignment,
)

# from rich import print
from descriptor import Descriptor
from evals import Validator

from evals import Validator
from descriptor import Clusterer
import asyncio
import logging
import datetime


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of systems for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    for session in initialize_session():

        archive = [
            COT,
            COT_SC,
            Reflexion,
            LLM_debate,
            Take_a_step_back,
            QD,
            Role_Assignment,
        ]

        population = Population(session=session, population_benchmark=args.benchmark)
        descriptor = Descriptor()
        validator = Validator(args)
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

        systems_for_validation = (
            session.query(System)
            .filter_by(population_id=population_id, system_fitness=None)
            .all()
        )

        validator.validate(systems_for_validation)

        # Re-load the population object in this session
        population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        # Recluster the population
        clusterer.cluster(population)

    return population_id
