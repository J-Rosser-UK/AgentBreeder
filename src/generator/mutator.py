import random
import pandas as pd
from typing import List
from base import System
from chat import get_structured_json_response_from_gpt
from prompts.mutation_base import get_base_prompt
from prompts.mutation_reflexion import Reflexion_prompt_1
import os
import uuid
from evals import AgentSystemException
import logging
import json
import re


class Mutator:

    def __init__(
        self,
        args,
        session,
        population,
        mutation_operators,
        evaluator,
        generation_timestamp,
    ) -> None:
        """
        Initializes the Mutator class.

        Args:
            args: Arguments object containing configurations for the mutator, such
            as debugging limits and model settings.
            session: The session object for managing system interactions.
            population: The population of systems for mutation.
            mutation_operators: A list of mutation operator strings to apply.
            evaluator: An evaluator object for validating and testing mutated systems.
        """

        self.mutation_operators = mutation_operators
        self.args = args
        self.evaluator = evaluator
        self.session = session
        self.population = population
        self.generation_timestamp = generation_timestamp

    async def mutate(self) -> System:
        """
        Applies a mutation to a given system.

        Args:
            system (System): The system object to mutate.

        Returns:
            System: The mutated system object. Returns None if the mutation fails.
        """

        try:

            system_response, messages, reflexion_response_format, parent_system_ids = (
                await random.choice([self._mutate, self._crossover])()
            )

            system_response = await self._debug(
                messages, system_response, reflexion_response_format
            )

            mutated_system = System(
                session=self.session,
                system_name=system_response["name"],
                system_code=system_response["code"],
                system_first_parent_id=str(parent_system_ids[0]),
                system_second_parent_id=str(parent_system_ids[1]),
                system_thought_process=system_response["thought"],
                population=self.population,
                generation_timestamp=self.generation_timestamp,
            )
        except Exception as e:

            print(f"Error evolving system: {e}")
            mutated_system = None

        return mutated_system

    async def _mutate(self):
        """
        Applies a sampled mutation to a system and refines it using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        system = random.choice(self.population.elites)
        logging.info(f"Mutating {system.system_name} system...")
        print(f"Mutating {system.system_name} system...")

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
             
                Here is the multi-agent system I would like you to mutate:

                ---------------
                System: {system.system_name}
                {system.system_thought_process}
                ---------------
                {system.system_code}

                The mutation I would like to apply is:
                {sampled_mutation}

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the task. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.
                
                """.strip(),
            },
        ]

        return await self._evolve(
            messages, base_prompt_response_format, [system.system_id, None]
        )

    async def _crossover(self):
        """
        Applies crossover to two systems and refines the result using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        system_1 = random.choice(self.population.elites)
        system_2 = random.choice(self.population.elites)
        logging.info(
            f"Crossing over {system_1.system_name} and {system_2.system_name} systems..."
        )
        print(
            f"Crossing over {system_1.system_name} and {system_2.system_name} systems..."
        )

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
             
                Here are the two systems I'd like you to crossover/combine into a novel new system:

                ---------------
                System 1: {system_1.system_name}
                {system_1.system_thought_process}
                ---------------
                {system_1.system_code}

                ---------------
                System 2: {system_2.system_name}
                {system_2.system_thought_process}
                ---------------
                {system_2.system_code}   

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the task. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.             
                """.strip(),
            },
        ]

        return await self._evolve(
            messages,
            base_prompt_response_format,
            [system_1.system_id, system_2.system_id],
        )

    async def _evolve(self, messages, base_prompt_response_format, parent_system_ids):

        # Generate new solution and do reflection
        try:
            next_response: dict[str, str] = await get_structured_json_response_from_gpt(
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
            next_response = await get_structured_json_response_from_gpt(
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
            next_response = await get_structured_json_response_from_gpt(
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

        # Clean up next_response["system_name"] to only allow numbers, letters, hyphens and underscores
        next_response["name"] = re.sub(r"[^A-Za-z0-9 ]+", "", next_response["name"])

        return next_response, messages, reflexion_response_format, parent_system_ids

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

    async def _debug(self, messages, next_response, reflexion_response_format):
        """
        Handles debugging for a given response during mutation.

        Args:
            messages: List of messages exchanged during the mutation process.
            next_response: The generated response containing the multi-agent system code and metadata.
            reflexion_response_format: The response format for reflection.

        Returns:
            dict: The updated next_response after debugging attempts.
        """

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)

        temp_file = f"""
            {parent_directory}/temp/agent_system_temp_{next_response["name"]}_{uuid.uuid4()}.py
        """.strip()

        for d in range(self.args.debug_max):

            try:

                if "return self.forward" in next_response["code"]:
                    raise AgentSystemException(
                        """The output of the forward function must not be the forward function
                        itself, as it will recurse infinitely."""
                    )
                await self.evaluator.benchmark.forward_pass(
                    next_response["code"], temp_file, self.session
                )
                break

            except AgentSystemException as e:
                logging.info(f"Debugging meta agent's code: {e}")
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
                    next_response = await get_structured_json_response_from_gpt(
                        messages,
                        reflexion_response_format,
                        model=self.args.model,
                        temperature=0.5,
                        retry=0,
                    )
                except Exception as e:
                    print(f"Error during debugging: {e}")

        return next_response
