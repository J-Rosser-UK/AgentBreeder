from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse
import asyncio

from dotenv import load_dotenv
from rich import print

from openai_wrapper import OpenAIClientWrapper

load_dotenv() # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=2)     
parser.add_argument('-ts', '--num_thinking_styles', default=4)     
parser.add_argument('-e', '--num_evals', default=10)     
parser.add_argument('-n', '--simulations', default=10)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       

args = vars(parser.parse_args())

total_evaluations = args['num_mutation_prompts']*args['num_thinking_styles']*args['num_evals']

# set num_workers to total_evaluations so we always have a thread
co = OpenAIClientWrapper(api_key=os.environ['OPENAI_API_KEY'])
# co = cohere.Client(api_key=os.environ['COHERE_API_KEY'],  num_workers=total_evaluations, max_retries=5, timeout=30) #override the 2 min timeout with 30s. 

tp_set:list[str] = mutation_prompts[:int(args['num_mutation_prompts'])]       # mutation prompts set

mutator_set:list[str ]= thinking_styles[:int(args['num_thinking_styles'])]      # thinking style set

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population of mutations and thinking...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, co, int(args['num_evals']))

print(p)
print(len(p.units))
assert 1 == 2

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, model=co, num_evals=int(args['num_evals']))

print("%"*80)
print("done processing! final gen:")
print(p.units)
