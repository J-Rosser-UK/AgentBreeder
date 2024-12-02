import random
from base import Framework, Population, initialize_session
from prompts.mutation_base import get_init_archive
from rich import print
from descriptor import Descriptor
from evaluator import Evaluator
from prompts.ma_mutation_prompts import multi_agent_system_mutation_prompts
from icecream import ic
from .mutator import Mutator

def generate_mutant(args, population_id):
    session, Base = initialize_session()
    generator = Generator(args, session, population_id)
    mutant_framework = generator()
    session.close()
    return mutant_framework


class Generator:
    

    def __init__(self, args, session, population_id) -> None:
        
        self.session = session
        self.mutation_operators = multi_agent_system_mutation_prompts

        
        # self.crossover = Crossover(args)
        self.population = self.session.query(Population).filter_by(
            population_id=population_id
        ).one()

        self.frameworks = self.population.frameworks  # error on this line

        self.batch_size = 1
        self.descriptor = Descriptor()
        self.evaluator = Evaluator(args)
        self.mutator = Mutator(args, session, self.population, self.mutation_operators, self.evaluator)

        
    

    def __call__(self) -> Framework:
        """
        Generates a batch of new populations by mutating and crossing over sampled populations.
        """
        sample_framework = random.choice(self.population.elites)
   
        
        try:
            mutant_framework = self.mutator(sample_framework)
            
            if mutant_framework:
                mutant_framework.update(framework_descriptor = self.descriptor.generate(mutant_framework))

                self.population.frameworks.append(mutant_framework)
            
        except Exception as e:
            print(f"Error mutating {sample_framework.framework_name} framework: {e}")
            mutant_framework = None
        
        return mutant_framework


    

def initialize_population_id(args) -> str:
    """Initialize the first generation of frameworks for the population."""
    session, Base = initialize_session()

    archive = get_init_archive()

    population = Population(session=session)
    descriptor = Descriptor()

    for framework in archive:
        framework = Framework(
            session=session,
            framework_name=framework['name'],
            framework_code=framework['code'],
            framework_thought_process=framework['thought'],
            population=population
        )
        population.frameworks.append(framework)
   
    for framework in population.frameworks:
        framework.update(framework_descriptor = descriptor.generate(framework))


    population_id = str(population.population_id)    
    session.close()

    return population_id
