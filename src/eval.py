import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import initialize_population_id, generate_mutant, run_generation
from descriptor import Clusterer
from base import initialize_session, Population, System
from evals import Validator
import os
import uuid
import asyncio
import json
import warnings
from sqlalchemy.exc import SAWarning
from illuminator import Illuminator
from sqlalchemy.orm import joinedload
import time
from evals import CLRSText, MMLU

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def main(args):
    random.seed(args.random_seed)

    evaluator = MMLU(args=args, split="test", shuffle=True, limit=args.n_evals)

    for session in initialize_session():

        # Only choose systems which haven't been evaluated yet (e.g. system_fitness=None)
        systems_for_evaluation = (
            session.query(System).filter_by(system_id=args.system_id).all()
        )

        print(
            "Number of systems for evaluation",
            len(systems_for_evaluation),
        )

        model_metrics = evaluator.evaluate(
            systems_for_evaluation,
            limit=args.n_evals,
        )

        print(json.dumps(model_metrics, indent=4))

    return args.system_id  # Return the population ID for restarts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_evals", type=int, default=10)
    parser.add_argument("--population_id", type=str, default="None")
    parser.add_argument("--system_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="mmlu")

    args = parser.parse_args()

    system_id = main(args)
