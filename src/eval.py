import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from generator import initialize_population_id, generate_mutant, run_generation
from descriptor import Clusterer
from base import initialize_session, Population, System
from evals import Evaluator
import os
import uuid
import asyncio

import warnings
from sqlalchemy.exc import SAWarning
from illuminator import Illuminator
from sqlalchemy.orm import joinedload
import time
from evals import EvaluateCLRSText, EvaluateMMLU

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


def main(args):
    random.seed(args.random_seed)

    evaluator = EvaluateMMLU(args=args, split="test", shuffle=True, limit=args.n_evals)

    try:
        session, Base = initialize_session()

        # Only choose systems which haven't been evaluated yet (e.g. system_fitness=None)
        systems_for_evaluation = (
            session.query(System).filter_by(system_id=args.system_id).all()
        )

        print(
            "Number of systems for evaluation",
            len(systems_for_evaluation),
        )

        for i, system in enumerate(systems_for_evaluation):

            accuracy, ci_lower, ci_upper, median = evaluator.evaluate(
                system,
                i + 1,
                len(systems_for_evaluation),
                limit=args.n_evals,
            )

            print(accuracy, ci_lower, ci_upper, median)

    except:
        session.rollback()
        raise
    finally:
        # be sure to close it!
        session.close()

    return args.system_id  # Return the population ID for restarts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_evals", type=int, default=100)
    parser.add_argument("--population_id", type=str, default="None")
    parser.add_argument("--system_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="mmlu")

    args = parser.parse_args()

    system_id = main(args)