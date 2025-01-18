import argparse

import logging

from base import initialize_session, Population, System

import os
import uuid
import asyncio
import json
import warnings
from sqlalchemy.exc import SAWarning

import time
from evals import CLRSText, MMLU, ARC, GPQA, DROP, MGSM, SaladData, SimpleQA, Math500
import pandas as pd
import numpy as np
from rich import print
from evals.validator import Validator

# Disable logging for httpx
logging.getLogger("httpx").disabled = True

# Suppress all SAWarnings
warnings.filterwarnings("ignore", category=SAWarning)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--current_dir", type=str, default=current_directory)
    parser.add_argument("--log_timestamp", type=str, default=log_timestamp_str)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_generation", type=int, default=10)
    parser.add_argument("--n_mutations", type=int, default=10)
    parser.add_argument("--n_evals", type=int, default=50)
    parser.add_argument("--debug_max", type=int, default=3)
    parser.add_argument("--pareto", type=bool, default=True)
    parser.add_argument("--safety", type=bool, default=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("-p", "--population_id", type=str, default="None")
    parser.add_argument("--benchmark", type=str, default="None")

    args = parser.parse_args()

    eval_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))

    # try:
    populations = []
    for session in initialize_session():

        population_ids = [
            "f3aded39-55e9-4aa6-814e-6d75c999ab9a",
            "9d7e54bd-c567-4791-9062-5338a216535d",
            "c3719b4f-3d3e-4eea-933c-73a2f1b2e335",
        ]

        for population_id in population_ids:

            args.pareto = None

            population = (
                session.query(Population).filter_by(population_id=population_id).one()
            )

            args.benchmark = population.population_benchmark

            print(population_id, population.population_benchmark)

            systems = session.query(System).filter_by(population_id=population_id).all()

            # do chunks of 20 systems at a time

            # reverse the systems list:
            systems = systems[::-1]

            for i in range(0, len(systems), 20):
                systems_chunk = systems[i : i + 20]

                validator = Validator(args)

                validator.validate(systems_chunk)
