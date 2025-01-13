import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import initialize_population_id, Generator
from descriptor import Clusterer
from base import initialize_session, Population, System
from evals import Validator
import os
import uuid
import asyncio
import datetime
import warnings
from sqlalchemy.exc import SAWarning
from illuminator import Illuminator
from sqlalchemy.orm import joinedload
import time


# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def main(args, population_id=None):
    random.seed(args.random_seed)

    validator = Validator(args)
    clusterer = Clusterer()
    illuminator = Illuminator(args)

    # Initialize population_id only if it doesn't exist
    if not population_id:
        population_id = initialize_population_id(args)
        print(f"Population ID: {population_id}")
    else:
        for session in initialize_session():

            # Re-load the population object in this session
            population = (
                session.query(Population).filter_by(population_id=population_id).one()
            )
            # Recluster the population
            clusterer.cluster(population)

            # Only choose systems which haven't been validated yet (e.g. system_fitness=None)
            systems_for_validation = (
                session.query(System)
                .filter_by(population_id=population_id, system_fitness=-1)
                .order_by(System.system_timestamp.desc())
                .all()[:10]
            )
            validator.validate(systems_for_validation)

            print(f"Reloaded population ID: {population.population_id}")

    for session in initialize_session():
        # Begin Bayesian Illumination...
        for g in tqdm(range(args.n_generation), desc="Generations"):

            # Re-load the population object in this session
            population = (
                session.query(Population).filter_by(population_id=population_id).one()
            )

            generator = Generator(args, population)

            # Generate a new batch of mutants
            asyncio.run(generator.run_generation(session))

            # Recluster the population
            clusterer.cluster(population)

            # Only choose systems which haven't been validated yet (e.g. system_fitness=None)
            systems_for_validation = (
                session.query(System)
                .filter_by(population_id=population_id, system_fitness=None)
                .all()
            )

            validator.validate(systems_for_validation)

            session.commit()

    return population_id  # Return the population ID for restarts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_generation", type=int, default=5)
    parser.add_argument("--n_mutations", type=int, default=20)
    parser.add_argument("--n_evals", type=int, default=20)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("-p", "--population_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="drop")

    args = parser.parse_args()

    population_id = args.population_id
    if population_id == "None":
        population_id = None

    if population_id == "last":
        for session in initialize_session():
            population = (
                session.query(Population)
                .order_by(Population.population_timestamp.desc())
                .limit(1)
                .one()
            )
            population_id = population.population_id

    while True:
        # try:
        population_id = main(args, population_id)
        break
        #     break  # Exit the loop if successful
        # except Exception as e:
        #     logging.error(f"An error occurred: {e}")
        #     logging.info("Restarting the process...")
        #     time.sleep(5)  # Optional: Add a small delay before restarting
