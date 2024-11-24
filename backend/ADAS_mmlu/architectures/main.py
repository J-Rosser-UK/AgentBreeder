
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

from utils import bootstrap_confidence_interval

import os

from agent import initialize_session



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


def format_question(multiple_choice_question):
    
    QUERY_TEMPLATE_MULTICHOICE = """
    Answer the following multiple choice question.

    <<Question>>

    (A) <<A>>
    (B) <<B>>
    (C) <<C>>
    (D) <<D>>
    """.strip()

    # Start by replacing the question
    prompt = QUERY_TEMPLATE_MULTICHOICE.replace("<<Question>>", multiple_choice_question.question)

    # Replace each option placeholder iteratively
    for letter in ['A', 'B', 'C', 'D']:
        prompt = prompt.replace(f"<<{letter}>>", getattr(multiple_choice_question, letter))

    return prompt

    

def evaluate_framework(framework, multiple_choice_questions, args):
   
    # Create the agent framework in temporary code
    current_directory = os.path.dirname(os.path.abspath(__file__))
    temp_file = f"{current_directory}/agent_system_temp.py"
    forward_function = framework.framework_code

    # Write the complete AgentSystem class to the file, including the forward function
    with open(temp_file, "w") as f:
        f.write("import random\n")
        f.write("import pandas\n\n")
        f.write(f"from agent import Agent, Meeting, Chat\n\n")
        f.write("class AgentSystem:\n")
        f.write("    " + forward_function.replace("\n", "\n    "))  

    # Import the AgentSystem class from the temp file
    spec = importlib.util.spec_from_file_location("agent_system_temp", temp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    AgentSystem = module.AgentSystem

    results_list = []
    for question in multiple_choice_questions:
        agentSystem = AgentSystem()

        task = format_question(question)
        agent_framework_answer = agentSystem.forward(task)

        if agent_framework_answer == question.correct_answer_letter:
            results_list.append(1)
        else:
            results_list.append(0)

    
    print(f"acc: {bootstrap_confidence_interval(results_list)}")
    return results_list




class MultipleChoiceQuestion(BaseModel):
    question_id: uuid.UUID
    question: str
    A: str
    B: str
    C: str
    D: str
    correct_answer_letter: str
    subject: str


def load_eval_dataset(args) -> list[MultipleChoiceQuestion]:

    df = pandas.read_csv(args.data_filename)
    examples = [row.to_dict() for _, row in df.iterrows()]    
    random.shuffle(examples)

    multiple_choice_questions = []
    for example in examples:
        multiple_choice_questions.append(MultipleChoiceQuestion(
            question_id=uuid.uuid4(),
            question=example['Question'],
            A=example['A'],
            B=example['B'],
            C=example['C'],
            D=example['D'],
            correct_answer_letter=example['Answer'],
            subject=example['Subject']
        ))

    return multiple_choice_questions


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
    parser.add_argument('--data_filename', type=str, default="/home/j/Documents/AgentBreeder/backend/ADAS_mmlu/architectures/mmlu_sample_3.csv")
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