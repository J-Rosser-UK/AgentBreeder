
from agent import Agent, Meeting, Chat, Experiment, Framework
import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm
import time
import importlib.util
from pydantic import BaseModel
import uuid

from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt

client = openai.OpenAI()


import os

from agent import initialize_session
from eval import load_eval_dataset, evaluate_framework



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

    
    evaluate_framework(frameworks[0], multiple_choice_questions, args)

    for g in range(1, args.n_generation+1):

        print(f"Generation {g}")

        for framework in frameworks:

        #     print(f"Framework: {framework.framework_name}")

        #     prompt = get_prompt(framework.framework_name, framework.framework_code, framework.framework_thought_process)

        #     # Get the reflexion prompt
        #     reflexion_prompt = get_reflexion_prompt(framework.framework_name, framework.framework_code, framework.framework_thought_process)

        #     # Get the response from the LLM
        #     response = get_json_response_from_gpt_reflect(
        #         messages=[
        #             {"role": "system", "content": prompt},
        #             {"role": "user", "content": reflexion_prompt}
        #         ],
        #         model=args.model
        #     )

        #     # Save the response
        #     framework.framework_code = response['code']
        #     framework.framework_thought_process = response['thought']
        #     framework.framework_generation = g

        #     time.sleep(1)

        # # Save the frameworks
        # experiment.frameworks.extend(frameworks)

        # # Get the next generation of frameworks
        # frameworks = get_next_generation(frameworks)
        
            pass
        pass

    # print(frameworks)








if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/backend/ADAS_mmlu/code/mmlu_sample_3.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/home/j/Documents/AgentBreeder/backend/ADAS_mmlu/architectures/results')
    parser.add_argument('--dataset_name', type=str, default="mmlu")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-mini')

    args = parser.parse_args()


    session, Base = initialize_session()

    main(args)