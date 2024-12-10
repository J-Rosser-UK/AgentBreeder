import random
from base import Framework, Population, initialize_session
from prompts.mutation_base import get_init_archive
from rich import print
from descriptor import Descriptor
from evals import Evaluator
from prompts.ma_mutation_prompts import multi_agent_system_mutation_prompts
from icecream import ic
from .mutator import Mutator
from evals import Evaluator


def generate_mutant(args, population_id):
    session, Base = initialize_session(args.db_name)
    generator = Generator(args, session, population_id)
    mutant_framework = generator()
    session.close()
    return mutant_framework


class Generator:

    def __init__(self, args, session, population_id) -> None:
        """
        Initializes the Generator class.

        Args:
            args: Arguments object containing configurations for the generator.
            session: The session object for managing database interactions.
            population_id: The ID of the population to operate on.
        """

        self.session = session
        self.mutation_operators = multi_agent_system_mutation_prompts

        # self.crossover = Crossover(args)
        self.population = (
            self.session.query(Population).filter_by(population_id=population_id).one()
        )

        self.frameworks = self.population.frameworks  # error on this line

        self.batch_size = 1
        self.descriptor = Descriptor()
        self.evaluator = Evaluator(args)
        self.mutator = Mutator(
            args, session, self.population, self.mutation_operators, self.evaluator
        )

    def __call__(self) -> Framework:
        """
        Generates a mutated framework from the population by sampling and applying mutations.

        Returns:
            Framework: The mutated framework object. Returns None if the mutation fails.
        """

        sample_framework = random.choice(self.population.elites)
        mutant_framework = None
        if sample_framework:

            try:
                mutant_framework = self.mutator(sample_framework)

                if mutant_framework:
                    mutant_framework.update(
                        framework_descriptor=self.descriptor.generate(mutant_framework)
                    )

                    self.population.frameworks.append(mutant_framework)

            except Exception as e:
                if sample_framework:
                    print(
                        f"Error mutating {sample_framework.framework_name} framework: {e}"
                    )
                    mutant_framework = None

        return mutant_framework


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of frameworks for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    session, Base = initialize_session(args.db_name)

    archive = get_init_archive()

    population = Population(session=session)
    descriptor = Descriptor()
    evaluator = Evaluator(args)

    for framework in archive:
        framework = Framework(
            session=session,
            framework_name=framework["name"],
            framework_code=framework["code"],
            framework_thought_process=framework["thought"],
            population=population,
        )
        population.frameworks.append(framework)

    for framework in population.frameworks:
        framework.update(framework_descriptor=descriptor.generate(framework))

    population_id = str(population.population_id)

    frameworks_for_evaluation = (
        session.query(Framework)
        .filter_by(population_id=population_id, framework_fitness=None)
        .all()
    )

    # evaluator.async_evaluate(illuminated_frameworks_for_evaluation)
    evaluator.inspect_evaluate(frameworks_for_evaluation[0:1])

    session.close()

    return population_id
