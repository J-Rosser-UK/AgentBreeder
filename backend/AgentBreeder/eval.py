import os
import random
import pandas

import importlib.util
from pydantic import BaseModel
import uuid
import logging
from utils import bootstrap_confidence_interval



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
    
    # Create a new session for this thread
    logging.info(f"Evaluating framework: {framework.framework_name}")

    # Create the agent framework in temporary code
    current_directory = os.path.dirname(os.path.abspath(__file__))
    temp_file = f"{current_directory}/temp/agent_system_temp_{framework.framework_name}_{framework.framework_id}.py"
    forward_function = framework.framework_code

    # Write the complete AgentSystem class to the file, including the forward function
    with open(temp_file, "w") as f:
        f.write("import random\n")
        f.write("import pandas\n\n")
        f.write(f"from base import Agent, Meeting, Chat\n\n")
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

    
    confidence_level = 0.95
    confidence_interval_string, ci_lower, ci_upper, median = bootstrap_confidence_interval(results_list, confidence_level=confidence_level)

    framework.update(
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_median=median,
        ci_sample_size=len(results_list),
        ci_confidence_level=confidence_level
    )

    logging.info(f"Framework: {framework.framework_name}, Confidence Interval: {confidence_interval_string}")

    # delete file at the end
    os.remove(temp_file)
    

    return median


