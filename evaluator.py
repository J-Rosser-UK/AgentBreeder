import os
import random
import pandas

import importlib.util
from pydantic import BaseModel
import uuid
import logging
from utils import bootstrap_confidence_interval
from base import Framework, initialize_session, Population
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from chat import get_structured_json_response_from_gpt

class MultipleChoiceQuestion(BaseModel):
    question_id: uuid.UUID
    question: str
    A: str
    B: str
    C: str
    D: str
    correct_answer_letter: str
    subject: str

class AgentSystemException(Exception):
    """Custom exception for errors in the agent system."""
    pass



class Evaluator:

    def __init__(self, args):
        self.args = args
        self.multiple_choice_questions = self.load_eval_dataset()
       

    def load_eval_dataset(self) -> list[MultipleChoiceQuestion]:

        df = pandas.read_csv(self.args.data_filename)
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



    def format_question(self, multiple_choice_question):
        
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
    

    def evaluate_mocked_forward_function(self, forward_function, temp_file) -> None:
        
        try:
        
            # Write the complete AgentSystem class to the file, including the forward function
            with open(temp_file, "w") as f:
                f.write("import random\n")
                f.write("import numpy as np\n")
                f.write("import pandas\n\n")
                f.write(f"from mocked_base import MockAgent as Agent\n\n")
                f.write(f"from mocked_base import MockMeeting as Meeting\n\n")
                f.write(f"from mocked_base import MockChat as Chat\n\n")
                f.write(f"from sqlalchemy.orm import Session\n\n")
                f.write("class AgentSystem:\n")
                f.write("    def __init__(self, session: Session = None):\n")
                f.write("        self.Agent = Agent\n")
                f.write("        self.Meeting = Meeting\n")
                f.write("        self.Chat = Chat\n")
                f.write("        self.session = session\n\n")
                f.write("    " + forward_function.replace("\n", "\n    "))
                f.write("\n\n")
                f.write("if __name__ == '__main__':\n")
                f.write("    " + "from base import initialize_session\n")
                f.write("    " + "session, Base = initialize_session\n")
                f.write("    " + "agent_system = AgentSystem()\n")
                f.write("    " + "task = \"What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others.\"\n")
                f.write("    " + "output = agent_system.forward(task)\n")
                f.write("    " + "print(output)\n")

            # Import the AgentSystem class from the temp file
            spec = importlib.util.spec_from_file_location("agent_system_temp", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AgentSystem = module.AgentSystem

            agentSystem = AgentSystem()

            task = "Fake task"
            agent_framework_answer = agentSystem.forward(task)

            # delete file at the end
            os.remove(temp_file)

        except Exception as e:
            raise AgentSystemException(f"Error evaluating framework: {e}")
        

    
    def illuminate(self, population:Population, frameworks_for_evaluation:Framework):


        generation = population.generations[-1] 

        
        print(f"Number of clusters in generation {generation.generation_id}: {len(generation.clusters)}")
       
        for framework in frameworks_for_evaluation:

            new_framework = {"framework_name": framework.framework_name, "framework_thought_process": framework.framework_thought_process, "framework_code": framework.framework_code}


            framework_cluster_id = framework.cluster_id

            framework_cluster = framework.cluster

            print(framework_cluster_id, framework_cluster)


            cluster_frameworks:list[Framework] = [
                {"framework_name": fw.framework_name, "framework_thought_process": fw.framework_thought_process, "framework_code": fw.framework_code}
                 
                 for fw in framework.cluster.frameworks]

            messages = [
                {"role": "system", "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""},
                {"role": "user", "content": f"""
                    Given the following multi-agent frameworks defined in code, do you think there is a 50% or greater chance that the
                    new framework will outperform them in a simulated environment?
                
                    Here are the frameworks in the cluster:
                    {cluster_frameworks}

                    Here is the new framework:
                    {new_framework}
                    
                    """.strip() 
                
                },
            ]

            print(messages)

            response_format = {
                "thinking": "Your step by step thinking.",
                "Will it outperform the other frameworks?": "A single letter, Y or N."
            }

            
            # Generate new solution and do reflection
            illuminated_frameworks_for_evaluation = []
            try:
                Y_or_N = get_structured_json_response_from_gpt(messages, response_format, model=self.args.model, temperature=0.5, retry=0)
                if Y_or_N["Will it outperform the other frameworks"] == "Y":
                    illuminated_frameworks_for_evaluation.append(Framework(
                        framework_name=new_framework["framework_name"],
                        framework_thought_process=new_framework["framework_thought_process"],
                        framework_code=new_framework["framework_code"],
                        population_id=population.population_id
                    ))
            except Exception as e:
                logging.error(f"Error in illumination: {e}")
                continue

        return illuminated_frameworks_for_evaluation


            

    
    def async_evaluate(self, frameworks_for_evaluation: list[Framework]):

        framework_id_map = {}
        framework_question_pairs:list[dict] = []
        for framework in frameworks_for_evaluation:
            framework.update(framework_fitness=-1)
            for n in range(5):
                framework_question_pairs.append({
                    "framework_id": framework.framework_id,
                    "framework_name": framework.framework_name,
                    "forward_function": framework.framework_code,
                    "multiple_choice_question":random.choice(self.multiple_choice_questions),
                    "result": None
                })
                framework_id_map[framework.framework_id] = framework


        # use concurrent futures to multithread this and return updated framework_question_pairs
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._thread_eval, pair) for pair in framework_question_pairs]
            for future in tqdm(futures, desc="Evaluating Async Frameworks"):
                pair = future.result()

        

        framework_results_pairs = {}

        for pair in framework_question_pairs:
            framework_id = pair["framework_id"]
            if framework_id not in framework_results_pairs:
                framework_results_pairs[framework_id] = []
            framework_results_pairs[framework_id].append(pair["result"])

        # Update the frameworks with the results
        for framework_id, results_list in framework_results_pairs.items():
            framework = framework_id_map[framework_id]

            confidence_level = 0.95
            confidence_interval_string, ci_lower, ci_upper, median = bootstrap_confidence_interval(results_list, confidence_level=confidence_level)

            framework.update(
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_median=median,
                ci_sample_size=len(results_list),
                ci_confidence_level=confidence_level,
                framework_fitness=median
            )


            logging.info(f"Framework: {framework.framework_name}, Confidence Interval: {confidence_interval_string}")
    
    def _thread_eval(self, pair):

        session, Base = initialize_session()

        # Create the agent framework in temporary code
        current_directory = os.path.dirname(os.path.abspath(__file__))
        temp_file = f"{current_directory}/temp/agent_system_temp_{pair['framework_name']}_{pair['framework_id']}_{uuid.uuid4()}.py"
        

        result = self.evaluate_forward_function_on_one_question(
            session,
            pair["multiple_choice_question"],
            pair["forward_function"],
            temp_file
        )
        
        pair["result"] = result

        # delete file at the end
        os.remove(temp_file)

        session.close()

    def evaluate_forward_function_on_one_question(self, session, multiple_choice_question, forward_function, temp_file) -> int:
        
        if "return self.forward" in forward_function:
            return 0

        try:
        
            # Write the complete AgentSystem class to the file, including the forward function
            with open(temp_file, "w") as f:
                f.write("import random\n")
                f.write("import pandas\n\n")
                f.write(f"from base import Agent, Meeting, Chat, Wrapper\n\n")
                f.write(f"from sqlalchemy.orm import Session\n\n")
                f.write("class AgentSystem:\n")
                f.write("    def __init__(self, session: Session):\n")
                f.write("        self.Agent = Wrapper(Agent, session)\n")
                f.write("        self.Meeting = Wrapper(Meeting, session)\n")
                f.write("        self.Chat = Wrapper(Chat, session)\n")
                f.write("        self.session = session\n\n")
                f.write("    " + forward_function.replace("\n", "\n    "))
                f.write("\n\n")
                f.write("if __name__ == '__main__':\n")
                f.write("    " + "from base import initialize_session\n")
                f.write("    " + "session, Base = initialize_session\n")
                f.write("    " + "agent_system = AgentSystem()\n")
                f.write("    " + "task = \"What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others.\"\n")
                f.write("    " + "output = agent_system.forward(task)\n")
                f.write("    " + "print(output)\n")

            # Import the AgentSystem class from the temp file
            spec = importlib.util.spec_from_file_location("agent_system_temp", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            AgentSystem = module.AgentSystem
            
            agentSystem = AgentSystem(session)

            task = self.format_question(multiple_choice_question)
            agent_framework_answer = agentSystem.forward(task)

            if agent_framework_answer == multiple_choice_question.correct_answer_letter:
                return 1
            else:
                return 0

            

        except Exception as e:
            return 0
   
    

    
