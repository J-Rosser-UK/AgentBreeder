import pydantic
from typing import List
from prompts.mutation_prompts import mutation_prompts
from prompts.thinking_styles import thinking_styles

class EvolutionUnit(pydantic.BaseModel):
    """ A individual unit of the overall population.

    Attributes:
        'T': the thinking_style.

        'M': the mutation_prompt.

        'P': the task_prompt.
        
        'fitness': the estimated performance of the unit.
        
        'history': historical prompts for analysis. 
    """
    T: str
    M: str
    P: str
    fitness: float
    history: List[str]

class Population(pydantic.BaseModel):
    """ Population model that holds the age of the population, its size, and a list of individuals.
    
    Attributes:
        'size' (int): the size of the population.

        'age' (int): the age of the population.
        
        'units' (List[EvolutionUnit]): the individuals of a population.
    """
    size: int
    age: int
    elites: List[EvolutionUnit]
    units: List[EvolutionUnit]


def create_population(mutation_prompts_set: List, thinking_styles_set: List, problem) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
    """
    data = {
        'size': len(mutation_prompts_set)*len(thinking_styles_set),
        'age': 0,
        'elites' : [],
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in mutation_prompts_set for m in thinking_styles_set]
    }

    return Population(**data)


def initialize_mutation_thinking_population(args) -> Population:
    mutation_prompts_set:list[str] = mutation_prompts[:int(args['num_mutation_prompts'])]       # mutation prompts set
    thinking_styles_set:list[str] = thinking_styles[:int(args['num_thinking_styles'])]      # thinking style set

    p = create_population(mutation_prompts_set=mutation_prompts_set, thinking_styles_set=thinking_styles_set)

    init_run(p, co, int(args['num_evals']))
