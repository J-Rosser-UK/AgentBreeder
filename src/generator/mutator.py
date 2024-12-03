import random
import pandas as pd
from typing import List
from base import Framework
from chat import get_structured_json_response_from_gpt
from prompts.mutation_base import get_base_prompt
from prompts.mutation_reflexion import Reflexion_prompt_1
import os
import uuid
from evaluator import AgentSystemException
import logging
import json


class Mutator:

    def __init__(
        self, args, session, population, mutation_operators, evaluator
    ) -> None:
        """
        Initializes the Mutator class.

        Args:
            args: Arguments object containing configurations for the mutator, such
            as debugging limits and model settings.
            session: The session object for managing framework interactions.
            population: The population of frameworks for mutation.
            mutation_operators: A list of mutation operator strings to apply.
            evaluator: An evaluator object for validating and testing mutated frameworks.
        """

        self.mutation_operators = mutation_operators
        self.args = args
        self.evaluator = evaluator
        self.session = session
        self.population = population

    def __call__(self, framework: Framework) -> List[Framework]:
        """
        Applies a mutation to a given framework.

        Args:
            framework (Framework): The framework object to mutate.

        Returns:
            List[Framework]: A list containing the mutated framework objects.
        """

        sampled_mutation = random.choice(self.mutation_operators)

        next_response, messages, reflexion_response_format = self._mutate(
            framework, sampled_mutation
        )

        next_response = self._debug(messages, next_response, reflexion_response_format)

        mutated_framework = Framework(
            session=self.session,
            framework_name=next_response["name"],
            framework_code=next_response["code"],
            framework_thought_process=next_response["thought"],
            population=self.population,
        )

        return mutated_framework

    def _mutate(self, framework: Framework, sampled_mutation: str):
        """
        Applies a sampled mutation to a framework and refines it using reflexion-based prompts.

        Args:
            framework (Framework): The framework object to mutate.
            sampled_mutation (str): The mutation operator selected for this mutation.

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """

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

            # Reflexion 1
            Reflexion_prompt_1, reflexion_response_format = (
                self._get_reflexion_prompt_1(next_response)
            )

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
            Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section,
            revise the code further. Put your new reflection thinking in "reflection". Repeat the
            previous "thought" and "name", and update the corrected version of the code in "code".
            """
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

        return next_response, messages, reflexion_response_format

    def _get_reflexion_prompt_1(self, prev_example):

        prev_example_str = (
            "Here is the previous agent you tried:\n"
            + json.dumps(prev_example)
            + "\n\n"
        )
        r1 = (
            Reflexion_prompt_1.replace("<<EXAMPLE>>", prev_example_str)
            if prev_example
            else Reflexion_prompt_1.replace("<<EXAMPLE>>", "")
        )
        reflexion_response_format = {
            "reflection": """
                Provide your thoughts on the interestingness of the architecture,
                identify any mistakes in the implementation, and suggest improvements.
            """,
            "thought": """
                Revise your previous proposal or propose a new architecture if necessary,
                using the same format as the example response.
            """,
            "name": """
                Provide a name for the revised or new architecture. (Don't put words like
                'new' or 'improved' in the name.)
            """,
            "code": """
                Provide the corrected code or an improved implementation. Make sure you
                actually implement your fix and improvement in this code.
            """,
        }

        return r1, reflexion_response_format

    def _debug(self, messages, next_response, reflexion_response_format):
        """
        Handles debugging for a given response during mutation.

        Args:
            messages: List of messages exchanged during the mutation process.
            next_response: The generated response containing the framework code and metadata.
            reflexion_response_format: The response format for reflection.

        Returns:
            dict: The updated next_response after debugging attempts.
        """

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        temp_file = f"""
            {parent_directory}/temp/agent_system_temp_{next_response['name']}_{uuid.uuid4()}.py
        """.strip()

        for _ in range(self.args.debug_max):

            try:
                if "return self.forward" in next_response["code"]:
                    raise AgentSystemException(
                        """The output of the forward function must not be the forward function
                        itself, as it will recurse infinitely."""
                    )
                self.evaluator.run_forward_pass(
                    next_response["code"], temp_file, self.session
                )

            except AgentSystemException as e:
                logging.error(f"Error during debugging: {e}")
                messages.append({"role": "assistant", "content": str(next_response)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"""Error during evaluation:\n{e}\nCarefully consider where
                        you went wrong in your latest implementation. Using insights from
                        previous attempts, try to debug the current code to implement the
                        same thought. Repeat your previous thought in 'thought', and put
                        your thinking for debugging in 'debug_thought'""",
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

        return next_response
