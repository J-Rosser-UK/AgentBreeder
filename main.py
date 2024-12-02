import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import Generator, initialize_population_id, generate_mutant
from descriptor import Clusterer
from visualizer import Visualizer
from base import initialize_session, Population, Framework
from evaluator import Evaluator
import time  # Added for restart delay
import warnings
from sqlalchemy.exc import SAWarning
import logging

import logging

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def main(args, population_id=None):
    random.seed(args.shuffle_seed)

    evaluator = Evaluator(args)
    clusterer = Clusterer()
    visualizer = Visualizer()

    # Initialize population_id only if it doesn't exist
    if not population_id:
        population_id = initialize_population_id(args)
    else:
        session, Base = initialize_session()
        frameworks_for_evaluation = session.query(Framework).filter_by(
            population_id=population_id,
            framework_fitness=None
        ).all()

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
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     list(tqdm(executor.map(lambda _: generate_mutant(args, population_id), range(args.n_mutations)), desc="Mutations", total=args.n_mutations))
        for m in range(args.n_mutations):
            generate_mutant(args, population_id)

        session, Base = initialize_session()

        
        # Re-load the population object in this session
        population = session.query(Population).filter_by(
            population_id=population_id
        ).one()

        # Recluster the population
        clusterer.cluster(population)

        frameworks_for_evaluation = session.query(Framework).filter_by(
            population_id=population_id,
            framework_fitness=None
        ).all()


        illuminated_frameworks_for_evaluation = evaluator.illuminate(population, frameworks_for_evaluation)

        print("Illuminated frameworks for evaluation: ", len(illuminated_frameworks_for_evaluation))

        evaluator.async_evaluate(illuminated_frameworks_for_evaluation)

        session.close()

    return population_id  # Return the population ID for restarts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/data/mmlu_sample_3.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/home/j/Documents/AgentBreeder/results')
    parser.add_argument('--dataset_name', type=str, default="mmlu")
    parser.add_argument('--n_generation', type=int, default=100)
    parser.add_argument('--n_mutations', type=int, default=10)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-mp', '--num_mutation_prompts', default=2)
    parser.add_argument('-ts', '--num_thinking_styles', default=4)
    parser.add_argument('-e', '--num_evals', default=10)
    parser.add_argument('-n', '--simulations', default=10)

    args = parser.parse_args()

    population_id = None
    while True:
        # try:
        population_id = main(args, population_id)
        #     break  # Exit the loop if successful
        # except Exception as e:
        #     logging.error(f"An error occurred: {e}")
        #     logging.info("Restarting the process...")
        #     time.sleep(5)  # Optional: Add a small delay before restarting
