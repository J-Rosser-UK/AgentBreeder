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

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from bayesian_illumination import Generator, initialize_population_id, generate_mutant
from descriptor import Clusterer, Visualizer
from base import initialize_session, Population, Framework

from evaluator import Evaluator

def main(args):

    random.seed(args.shuffle_seed)

    population_id = initialize_population_id(args)

    clusterer = Clusterer()

    visualizer = Visualizer()

    evaluator = Evaluator(args)

    # Begin Bayesian Illumination...
    for i in tqdm(range(args.n_generation), desc="Generations"):
        

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(lambda _: generate_mutant(args, population_id), range(args.n_mutations)), desc="Mutations", total=args.n_mutations))

        
        
        
        session, Base = initialize_session()

        # population_id = "de6326a9-95a7-4e19-bff6-5bb191ab1c25"

        # Get a list of frameworks where the framework_fitness is None
        frameworks_for_evaluation = session.query(Framework).filter_by(
            population_id=population_id,
            framework_fitness=None
        ).all()

        print("Frameworks for evaluation: ", len(frameworks_for_evaluation))


        evaluator.async_evaluate(frameworks_for_evaluation)


        # Re-load the population object in this session
        population = session.query(Population).filter_by(
            population_id=population_id
        ).one()
        
        # Recluster the population
        clusterer.cluster(population)

        # visualizer.plot(population)

        session.close()



if __name__ == "__main__":

    
    # Initialize a logging
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
    parser.add_argument('--n_generation', type=int, default=10)
    parser.add_argument('--n_mutations', type=int, default=10)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
    parser.add_argument('-ts', '--num_thinking_styles', default=4)     
    parser.add_argument('-e', '--num_evals', default=10)     
    parser.add_argument('-n', '--simulations', default=10)  

    args = parser.parse_args()
    main(args)