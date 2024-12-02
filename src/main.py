import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import initialize_population_id, generate_mutant
from descriptor import Clusterer
from base import initialize_session, Population, Framework
from evaluator import Evaluator
import os

import warnings
from sqlalchemy.exc import SAWarning
from illuminator import Illuminator


# Disable logging for httpx
# logging.getLogger("httpx").disabled = True

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
    else:
        session, Base = initialize_session()
        frameworks_for_evaluation = (
            session.query(Framework)
            .filter_by(population_id=population_id, framework_fitness=None)
            .all()
        )

        # Uncomment to evaluate any frameworks if evaluation was paused or stopped
        # frameworks_for_evaluation.extend(session.query(Framework).filter_by(
        #     population_id=population_id,
        #     framework_fitness=-1
        # ).limit(20))

        print("Frameworks for evaluation: ", len(frameworks_for_evaluation))

        evaluator.async_evaluate(frameworks_for_evaluation)
        session.close()

    # Begin Bayesian Illumination...
    for i in tqdm(range(args.n_generation), desc="Generations"):
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda _: generate_mutant(args, population_id),
                        range(args.n_mutations),
                    ),
                    desc="Mutations",
                    total=args.n_mutations,
                )
            )

        session, Base = initialize_session()

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

        illuminated_frameworks_for_evaluation = illuminator.illuminate(
            population, frameworks_for_evaluation
        )

        print(
            "Illuminated frameworks for evaluation: ",
            len(illuminated_frameworks_for_evaluation),
        )

        evaluator.async_evaluate(illuminated_frameworks_for_evaluation)

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
    parser.add_argument("--n_mutations", type=int, default=10)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--population_id", type=str, default="dd615513-d3a0-43fd-bee3-31d3e87b7cc4"
    )

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
