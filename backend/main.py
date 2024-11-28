"""
because the
work we build on comes from that community, here we adopt the
language and metaphors of evolutionary computation. In that
parlance, a solution is an organism or phenotype or individual, the
organism is described by a genome or genotype, and the actions
performed by that organism are the organism’s behavior. The performance or quality of a solution is called its fitness, and the equation, simulation, etc. that returns that fitness value is the fitness
function. The way of stochastically producing new solutions is
to take an existing solution and mutate its genome, meaning to
change the genome in some random way, and or to produce a
new solution descriptor by sampling portions of two parent descriptors, a process called crossover. Solutions that produce new
offspring organisms are those that are selected, and such selection
is typically biased towards solutions with higher fitness2
.

"""


from base import Agent, Meeting, Chat, Population, Framework
import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import logging
import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm
import time
import importlib.util
from pydantic import BaseModel
import uuid
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from prompts.mutation_base import get_init_archive

from tqdm import tqdm
import os
from base import initialize_session

from bayesian_illumination import Generator
from descriptor import Clusterer, Visualizer


def main(args):

    random.seed(args.shuffle_seed)

    mutant_generator = Generator(args)

    clusterer = Clusterer()

    visualizer = Visualizer()

    # Begin Bayesian Illumination...
    for i in tqdm(range(args.n_generation), desc="Generations"):

        with ThreadPoolExecutor(max_workers=18) as executor:
            list(tqdm(executor.map(lambda _: mutant_generator(), range(args.n_mutations)), desc="Mutations", total=args.n_mutations))

        # Recluster the population
        clusterer.cluster(mutant_generator.population)

        visualizer.plot(mutant_generator.population)



if __name__ == "__main__":

    session, Base = initialize_session()

    # Initialize a logging
    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/backend/data/mmlu_sample_3.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/home/j/Documents/AgentBreeder/backend/results')
    parser.add_argument('--dataset_name', type=str, default="mmlu")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--n_mutations', type=int, default=5)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
    parser.add_argument('-ts', '--num_thinking_styles', default=4)     
    parser.add_argument('-e', '--num_evals', default=10)     
    parser.add_argument('-n', '--simulations', default=10)  

    args = parser.parse_args()


    

    main(args)