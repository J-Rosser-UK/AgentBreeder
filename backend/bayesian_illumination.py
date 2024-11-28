import random
import pandas as pd
from typing import List
from base import Framework, Population
from chat import get_structured_json_response_from_gpt
from prompts.mutation_base import get_base_prompt, get_init_archive
from prompts.mutation_reflexion import get_reflexion_prompt
from eval import evaluate_forward_function
import os
import uuid
from eval import MultipleChoiceQuestion, AgentSystemException
from rich import print
from descriptor import Descriptor
from eval import Evaluator
from prompts.ma_mutation_prompts import multi_agent_system_mutation_prompts


class Generator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning.

    Attributes:
        population: The population of elite molecules used for generating new molecules.
        crossover: An instance of the Crossover class for generating molecule pairs.
        mutator: An instance of the Mutator class for mutating molecules.
        batch_size: The number of molecules to sample and mutate/crossover per batch.
        initial_data: The path to the initial data file containing molecule information.
        initial_size: The initial number of molecules to load from the database.

    Methods:
        __init__(config): Initializes the Generator with the given configuration.
        set_population(population): Sets the population for the Generator.
        __call__(): Generates a batch of new molecules by mutating and crossing over sampled molecules.
        load_from_database(): Loads a set of molecules from the initial data database.
    """

    def __init__(self, args, mutation_operators, multiple_choice_questions:list[MultipleChoiceQuestion]) -> None:
        """
        Initializes the Generator with the given configuration.

        Args:
            config: Configuration object containing settings for the Generator.
        """
        self.population = self.initialize_population()
        self.mutation_operators = multi_agent_system_mutation_prompts
        self.multiple_choice_questions = multiple_choice_questions
        # self.crossover = Crossover(args)
        self.mutator = Mutator(args, self.population, mutation_operators, multiple_choice_questions)
        self.batch_size = 1
        self.descriptor = Descriptor()
        self.evaluator = Evaluator(args)

        
    def initialize_population(self) -> Population:
        """Initialize the first generation of frameworks for the population."""

        archive = get_init_archive()

        population = Population()

        for framework in archive:
            population.frameworks.append(Framework(
                framework_name=framework['name'],
                framework_code=framework['code'],
                framework_thought_process=framework['thought'],
                framework_generation=0,
                population=population
            ))

        for framework in population.frameworks:
            framework.update(framework_fitness = 0.5)

        return population

    def __call__(self) -> Framework:
        """
        Generates a batch of new populations by mutating and crossing over sampled populations.

        Returns:
            List[Molecule]: A list of newly generated populations.
        """
        population_samples = [random.choice(self.population.frameworks) for _ in range(self.batch_size)] #TODO: sample from elites
        population_sample_pairs = [(random.choice(self.population.frameworks), random.choice(self.population.frameworks)) for _ in range(self.batch_size)]
        for framework in population_samples:
            mutant_framework = self.mutator(framework)
            if mutant_framework is not None:
                
                mutant_framework.update(framework_fitness = self.evaluator.evaluate(mutant_framework))
                mutant_framework.update(framework_descriptor = self.descriptor.generate(mutant_framework))

                self.population.frameworks.append(mutant_framework)

        # for population_pair in population_sample_pairs:
        #     population.frameworks.extend(self.crossover(population_pair))
        return mutant_framework



class Mutator:
    """
    A catalog class containing and implementing mutations to small molecules according to the principles of positional analogue scanning.

    Attributes:
        mutation_data: A dataframe containing mutation SMARTS patterns and their associated probabilities.

    Methods:
        __init__(mutation_data): Initializes the Mutator with the given mutation data.
        __call__(molecule): Applies a mutation to a given molecule and returns the resulting molecules.
    """
    def __init__(self, args, population, mutation_operators, multiple_choice_questions) -> None:
        """
        Initializes the Mutator with the given mutation data.

        Args:
            mutation_data (list[str]): A list of mutation operations to be applied to molecules.
        """
        self.mutation_operators = mutation_operators
        self.args = args
        self.multiple_choice_questions = multiple_choice_questions

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
            {"role": "system", "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""},
            {"role": "user", "content": f"""
                {base_prompt}
             
                Here is the framework I would like you to mutate:
                {framework.framework_code}

                The mutation I would like to apply is:
                {sampled_mutation}
                
                """.strip() 
             
            },
        ]

        
        # Generate new solution and do reflection
        try:
            next_response:dict[str,str] = get_structured_json_response_from_gpt(messages, base_prompt_response_format, model=self.args.model, temperature=0.5, retry=0)

            Reflexion_prompt_1, Reflexion_prompt_2, reflexion_response_format = get_reflexion_prompt(next_response)  
            # Reflexion 1
            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": Reflexion_prompt_1})
            next_response = get_structured_json_response_from_gpt(messages, reflexion_response_format, model=self.args.model, temperature=0.5, retry=0)
            # Reflexion 2
            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": Reflexion_prompt_2})
            next_response = get_structured_json_response_from_gpt(messages, reflexion_response_format, model=self.args.model, temperature=0.5, retry=0)
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
                acc_list = evaluate_forward_function(next_response["code"], temp_file, [self.multiple_choice_questions[0]])

                mutated_framework = Framework(
                    framework_name=next_response["name"],
                    framework_code=next_response["code"],
                    framework_thought_process=next_response["thought"],
                    population=framework.population
                )
                
            except AgentSystemException as e:
                print("During evaluation:")
                print(e)
                messages.append({"role": "assistant", "content": str(next_response)})
                messages.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_response = get_structured_json_response_from_gpt(messages, reflexion_response_format, model=self.args.model, temperature=0.5, retry=0)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)


        return mutated_framework
    
    


# class Crossover:
#     """
#     A strategy class implementing a parent-centric crossover of small molecules.

#     Methods:
#         __init__(): Initializes the Crossover object.
#         __call__(molecule_pair): Performs a crossover on a pair of molecules.
#         merge(molecule_pair): Merges the fragments of a molecule pair.
#         fragment(molecule_pair): Fragments a molecule pair into cores and sidechains.
#     """

#     def __init__(self, args):
#         """
#         Initializes the Crossover object.
#         """
#         pass

#     def __call__(self, molecule_pair):
#         """
#         Performs a crossover on a pair of molecules.

#         Args:
#             molecule_pair: A pair of molecules to be crossed over.

#         Returns:
#             List[Molecule]: A list of new molecules resulting from the crossover.
#         """
#         pedigree = ("crossover", molecule_pair[0].smiles, molecule_pair[1].smiles)
#         smiles_list = self.merge(molecule_pair)
#         molecules = [Molecule(Chem.CanonSmiles(smiles), pedigree) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
#         return molecules

#     def merge(self, molecule_pair):
#         """
#         Merges the fragments of a molecule pair.

#         Args:
#             molecule_pair: A pair of molecules to be merged.

#         Returns:
#             List[str]: A list of SMILES strings representing the merged molecules.
#         """
#         molecular_graphs = []
#         graph_cores, graph_sidechains = self.fragment(molecule_pair)
#         random.shuffle(graph_sidechains)
#         reaction = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
#         for core, sidechain in zip(graph_cores, graph_sidechains):
#             molecular_graphs.append(reaction.RunReactants((core, sidechain))[0][0])
#         smiles_list = [Chem.MolToSmiles(molecular_graph) for molecular_graph in molecular_graphs if molecular_graph is not None]
#         return smiles_list

#     def fragment(self, molecule_pair):
#         """
#         Fragments a molecule pair into cores and sidechains.

#         Args:
#             molecule_pair: A pair of molecules to be fragmented.

#         Returns:
#             Tuple[List[Chem.Mol], List[Chem.Mol]]: Two lists containing the cores and sidechains of the fragmented molecules.
#         """
#         graph_cores = []
#         graph_sidechains = []
#         for molecule in molecule_pair:
#             graph_frags = rdMMPA.FragmentMol(Chem.MolFromSmiles(molecule.smiles), maxCuts=1, resultsAsMols=False)
#             if len(graph_frags) > 0:
#                 _, graph_frags = map(list, zip(*graph_frags))
#                 for frag_pair in graph_frags:
#                     core, sidechain = frag_pair.split(".")
#                     graph_cores.append(Chem.MolFromSmiles(core.replace("[*:1]", "[1*]")))
#                     graph_sidechains.append(Chem.MolFromSmiles(sidechain.replace("[*:1]", "[1*]")))
#         return graph_cores, graph_sidechains