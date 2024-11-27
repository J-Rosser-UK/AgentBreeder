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


from agent import Agent, Meeting, Chat, Experiment, Framework
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

from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt

from tqdm import tqdm



import os
from agent import initialize_session
from eval import load_eval_dataset, evaluate_framework


# from pb import init_run, run_for_n

from population import initialize_mutation_thinking_population


def initialize_gen_0_frameworks(experiment:Experiment)->list[Framework]:
    """Initialize the first generation of frameworks for the experiment."""

    archive = get_init_archive()

    frameworks = []
    for framework in archive:
        frameworks.append(Framework(
            framework_name=framework['name'],
            framework_code=framework['code'],
            framework_thought_process=framework['thought'],
            framework_generation=0,
            experiment=experiment
        ))

    return frameworks



def main(args):

    experiment = Experiment()

    random.seed(args.shuffle_seed)

    file_path = os.path.join(args.save_dir, f"{args.dataset_name}_{args.model}_results_run_archive.json")
    
    frameworks = initialize_gen_0_frameworks(experiment)

    multiple_choice_questions = load_eval_dataset(args)

    with ThreadPoolExecutor(max_workers=18) as executor:
        median_percents = list(tqdm(executor.map(lambda framework: evaluate_framework(framework, multiple_choice_questions, args), frameworks), total=len(frameworks)))

    print(f"Total accuracy for archive: {sum(median_percents) / len(median_percents) * 100}%")

    

    population = initialize_mutation_thinking_population(args)


    # Run MAP-Elites algorithm

    # population = initialize_population(args)

    # mutation_operators = initialize_mutation_operators(args)

    # for i in range(args.n_generation):

    #     # Randomly select an elite from the map
    #     x = random_selection(population)

    #     # Randomly select a mutation operator
    #     m = random_selection(mutation_operators)

    #     # Mutate the elite to greate a mutant
    #     x_mutated = mutate(x, m)

    #     # Record the feature descriptors of the mutant
    #     x_mutated.b = feature_descriptor(x_mutated)

    #     # Evaluate the fitness of the mutant
    #     x_mutated.f = evaluate_fitness(x_mutated) 

    #     # Update the map with the mutant
    #     population = update_map(population, x_mutated)





if __name__ == "__main__":

    session, Base = initialize_session()

    # Initialize a logging
    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/backend/AgentBreeder/data/mmlu_sample_3.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/home/j/Documents/AgentBreeder/backend/AgentBreeder/results')
    parser.add_argument('--dataset_name', type=str, default="mmlu")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
    parser.add_argument('-ts', '--num_thinking_styles', default=4)     
    parser.add_argument('-e', '--num_evals', default=10)     
    parser.add_argument('-n', '--simulations', default=10)  

    args = parser.parse_args()


    

    main(args)