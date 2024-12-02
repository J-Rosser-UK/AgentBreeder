import random
import pandas as pd
from typing import List
from base import Framework, Population, initialize_session
from chat import get_structured_json_response_from_gpt
from prompts.mutation_base import get_base_prompt, get_init_archive
from prompts.mutation_reflexion import get_reflexion_prompt
import os
import uuid
from evaluator import AgentSystemException


class Mutator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning.

    Attributes:
        mutation_data: A dataframe containing mutation SMARTS patterns and their associated probabilities.

    Methods:
        __init__(mutation_data): Initializes the Mutator with the given mutation data.
        __call__(molecule): Applies a mutation to a given molecule and returns the resulting molecules.
    """

    def __init__(
        self, args, session, population, mutation_operators, evaluator
    ) -> None:
        """
        Initializes the Mutator with the given mutation data.

        Args:
            mutation_data (list[str]): A list of mutation operations to be applied to molecules.
        """
        self.mutation_operators = mutation_operators
        self.args = args
        self.evaluator = evaluator
        self.session = session
        self.population = population

    def __call__(self, framework: Framework) -> List[Framework]:
        """
        Applies a mutation to a given framework and returns the resulting frameworks.

        Args:
            framework: The framework to be mutated.

        Returns:
            List[Framework]: A list of new frameworks resulting from the mutation as applied
            by positional analogue scanning.
        """

        sampled_mutation = random.choice(self.mutation_operators)
        base_prompt, base_prompt_response_format = get_base_prompt()
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {base_prompt}
             
                Here is the framework I would like you to mutate:
                {framework.framework_code}

                The mutation I would like to apply is:
                {sampled_mutation}
                
                """.strip(),
            },
        ]

        # Generate new solution and do reflection
        try:
            next_response: dict[str, str] = get_structured_json_response_from_gpt(
                messages,
                base_prompt_response_format,
                model=self.args.model,
                temperature=0.5,
                retry=0,
            )

            Reflexion_prompt_1, Reflexion_prompt_2, reflexion_response_format = (
                get_reflexion_prompt(next_response)
            )
            # Reflexion 1
            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": Reflexion_prompt_1})
            next_response = get_structured_json_response_from_gpt(
                messages,
                reflexion_response_format,
                model=self.args.model,
                temperature=0.5,
                retry=0,
            )
            # Reflexion 2
            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": Reflexion_prompt_2})
            next_response = get_structured_json_response_from_gpt(
                messages,
                reflexion_response_format,
                model=self.args.model,
                temperature=0.5,
                retry=0,
            )
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            return None

        print("---New Framework!---")
        print(next_response)
        # Fix code if broken loop
        mutated_framework = None
        current_directory = os.path.dirname(os.path.abspath(__file__))
        temp_file = f"{current_directory}/temp/agent_system_temp_{next_response['name']}_{uuid.uuid4()}.py"
        for _ in range(self.args.debug_max):

            try:
                if "return self.forward" in next_response["code"]:
                    raise AgentSystemException(
                        "The output of the forward function must not be the forward function itself, as it will recurse infinitely."
                    )
                assert self.session is not None
                acc_list = self.evaluator.evaluate_mocked_forward_function(
                    next_response["code"], temp_file
                )

            except AgentSystemException as e:
                print("During evaluation:")
                print(e)
                messages.append({"role": "assistant", "content": str(next_response)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'",
                    }
                )
                try:
                    next_response = get_structured_json_response_from_gpt(
                        messages,
                        reflexion_response_format,
                        model=self.args.model,
                        temperature=0.5,
                        retry=0,
                    )
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)

        # print("Mutated Framework:")
        # print({
        #     "session":self.session,
        #     "framework_name":next_response["name"],
        #     "framework_code":next_response["code"],
        #     "framework_thought_process":next_response["thought"],
        #     "population":self.population
        # })
        mutated_framework = Framework(
            session=self.session,
            framework_name=next_response["name"],
            framework_code=next_response["code"],
            framework_thought_process=next_response["thought"],
            population=self.population,
        )

        # print("Successfully mutated:", mutated_framework)

        return mutated_framework
