import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import initialize_population_id, generate_mutant, run_generation
from descriptor import Clusterer
from base import initialize_session, Population, Framework
from evals import Evaluator
import os
import uuid
import asyncio

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
    random.seed(args.shuffle_seed)

    evaluator = Evaluator(args)
    clusterer = Clusterer()
    illuminator = Illuminator(args)

    # Initialize population_id only if it doesn't exist
    if not population_id:
        population_id = initialize_population_id(args)
        print(f"Population ID: {population_id}")
    else:
        session, Base = initialize_session(args.db_name)

        # Re-load the population object in this session
        population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        # # Recluster the population
        # clusterer.cluster(population)

        frameworks_for_evaluation = (
            session.query(Framework).filter_by(population_id=population_id).all()
        )

        illuminated_frameworks_for_evaluation_ids: list[str] = illuminator.illuminate(
            population, frameworks_for_evaluation
        )

        # Perform the query correctly
        illuminated_frameworks_for_evaluation = (
            session.query(Framework)  # Start the query
            .filter(
                Framework.framework_id.in_(illuminated_frameworks_for_evaluation_ids)
            )  # Apply the filter
            .all()  # Fetch all results
        )

        print(
            "fw for eval",
            len(frameworks_for_evaluation),
            "ilfw for eval",
            len(illuminated_frameworks_for_evaluation),
        )

        print(f"Reloaded population ID: {population.population_id}")

        session.close()

    # Begin Bayesian Illumination...
    for _ in tqdm(range(args.n_generation), desc="Generations"):

        # Generate a new batch of mutants
        asyncio.run(run_generation(args, population_id))

        session, Base = initialize_session(args.db_name)

        # Re-load the population object in this session
        population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )

        # Recluster the population
        clusterer.cluster(population)

        frameworks_for_evaluation = (
            session.query(Framework)
            .filter_by(population_id=population_id, framework_fitness=None)
            .all()
        )

        illuminated_frameworks_for_evaluation_ids: list[str] = illuminator.illuminate(
            population, frameworks_for_evaluation
        )

        # Perform the query correctly
        illuminated_frameworks_for_evaluation = (
            session.query(Framework)  # Start the query
            .filter(
                Framework.framework_id.in_(illuminated_frameworks_for_evaluation_ids)
            )  # Apply the filter
            .all()  # Fetch all results
        )

        print(
            "F",
            len(frameworks_for_evaluation),
            "I",
            len(illuminated_frameworks_for_evaluation),
        )

        # evaluator.async_evaluate(illuminated_frameworks_for_evaluation)
        evaluator.inspect_evaluate(illuminated_frameworks_for_evaluation_ids)

        session.close()

    return population_id  # Return the population ID for restarts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data_filename",
        type=str,
        default=f"{current_directory}/data/mmlu_sample_3.csv",
    )
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--n_generation", type=int, default=100)
    parser.add_argument("--n_mutations", type=int, default=1)
    parser.add_argument("--n_evals", type=int, default=20)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--population_id", type=str, default="None")
    parser.add_argument("--db_name", type=str, default="illuminator.db")

    args = parser.parse_args()

    population_id = args.population_id
    if population_id == "None":
        population_id = None

    while True:
        # try:
        population_id = main(args, population_id)
        #     break  # Exit the loop if successful
        # except Exception as e:
        #     logging.error(f"An error occurred: {e}")
        #     logging.info("Restarting the process...")
        #     time.sleep(5)  # Optional: Add a small delay before restarting
