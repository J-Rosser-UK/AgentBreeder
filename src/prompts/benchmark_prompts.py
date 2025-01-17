from textwrap import dedent

benchmark_prompts = {
    "arc": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the ARC (Abstraction and Reasoning Corpus)
        challenge.
        In this challenge, each task consists of three demonstration examples, and one test example. Each
        Example consists of an “input grid” and an “output grid”. Test-takers need to use the transformation rule
        learned from the examples to predict the output grid for the test example.
        # #An example task from ARC challenge:
        ## Task Overview:
        You will be given some number of paired example inputs and outputs grids. The outputs were produced
        by applying a transformation rule to the input grids. In addition to the paired example inputs and
        outputs, there is also one test input without a known output.
        The inputs and outputs are each “grids”. A grid is a rectangular matrix of integers between 0 and 9
        (inclusive). Each number corresponds to a color. 0 is black.
        Your task is to determine the transformation rule from examples and find out the answer, involving
        determining the size of the output grid for the test and correctly filling each cell of the grid with the
        appropriate color or number.
        The transformation only needs to be unambiguous and applicable to the example inputs and the test
        input. It doesn’t need to work for all possible inputs. Observe the examples carefully, imagine the grid
        visually, and try to find the pattern.
        ## Examples:
        ### Example 0:
        input = [[0,0,0,0,5,0,0,0,0], [0,0,0,0,5,0,0,0,0], [0,0,0,4,5,0,0,0,0], [0,0,0,4,5,4,4,0,0],
        [0,0,3,3,5,0,0,0,0], [0,0,0,3,5,0,0,0,0], [0,0,0,3,5,3,3,3,0], [0,0,0,3,5,0,0,0,0], [0,0,0,0,5,0,0,0,0],
        [0,0,0,0,5,0,0,0,0]]
        output = [[0,0,0,0], [0,0,0,0], [0,0,0,4], [0,0,4,4], [0,0,3,3], [0,0,0,3], [0,3,3,3], [0,0,0,3], [0,0,0,0],
        [0,0,0,0]]
        ### Example 1:
        input = [[0,0,0,0,5,0,0,0,0], [0,0,0,2,5,0,0,0,0], [0,0,0,2,5,2,6,0,0], [0,0,0,2,5,0,0,0,0],
        [0,0,0,2,5,2,2,2,0], [0,0,6,6,5,6,0,0,0], [0,0,0,2,5,0,0,0,0], [0,2,2,0,5,2,0,0,0], [0,0,0,2,5,0,0,0,0],
        [0,0,0,0,5,0,0,0,0]]
        output = [[0,0,0,0], [0,0,0,2], [0,0,6,2], [0,0,0,2], [0,2,2,2], [0,0,6,6], [0,0,0,2], [0,2,2,2], [0,0,0,2],
        [0,0,0,0]]
        ### Example 2:
        input = [[0,0,0,0,5,0,0,0,0], [0,0,0,0,5,7,0,0,0], [0,0,0,8,5,0,0,0,0], [0,0,0,8,5,0,0,0,0],
        [0,7,8,8,5,0,0,0,0], [0,0,0,0,5,8,8,0,0], [0,0,0,8,5,0,0,0,0], [0,0,0,8,5,0,0,0,0], [0,0,0,0,5,8,7,0,0],
        [0,0,0,0,5,0,0,0,0]]
        output= [[0,0,0,0], [0,0,0,7], [0,0,0,8], [0,0,0,8], [0,7,8,8], [0,0,8,8], [0,0,0,8], [0,0,0,8], [0,0,7,8],
        [0,0,0,0]]
        ### Test Problem:
        input = [[0,0,0,0,5,0,0,0,0], [0,0,0,1,5,0,0,0,0], [0,0,0,1,5,1,0,0,0], [0,1,1,1,5,1,1,1,6],
        [0,0,0,6,5,6,6,0,0], [0,0,0,0,5,1,1,1,0], [0,0,0,1,5,0,0,0,0], [0,0,0,1,5,1,6,0,0], [0,0,0,0,5,6,0,0,0],
        [0,0,0,0,5,0,0,0,0]]
        Analyze the transformation rules and Test Input and return the transformation function in the format:
```
def transform(grid: list[list[int]]) -> list[list[int]]:
    # Your code here
    return transformed_grid
```
        """
    ),
    "gpqa": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the GPQA (Graduate-Level Google-Proof Q&A
        Benchmark). This benchmark consists of challenging multiple-choice questions across the domains of
        biology, physics, and chemistry, designed by domain experts to ensure high quality and difficulty.
        ## An example question from GPQA:
        Two quantum states with energies E1 and E2 have a lifetime of 10-9 sec and 10-8 sec, respectively. We
        want to clearly distinguish these two energy levels. Which one of the following options could be their
        energy difference so that they be clearly resolved?
        Answer choices:
        10-9 eV
        10-8 eV
        10-7 eV
        10-6 eV
        Correct answer [Not provided]:
        10-7 eV
        Explanation [Not provided]:
        According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the
        width of the energy level. With Delta t=10-9 s==> Delta E1= 3.3 10-7 ev. And Delta t=10-11 s gives
        Delta E2=3.310-8 eV. Therefore, the energy difference between the two states must be significantly
        greater than 10-7 ev. So the answer is 10-4 ev
        """
    ),
    "mmlu": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the MMLU (Massive Multitask Language
        Understanding) benchmark, a challenging evaluation that assesses a model’s ability to answer questions
        across a wide range of subjects and difficulty levels. It includes subjects from STEM, social sciences,
        humanities, and more.
        ## An example question from MMLU:
        Answer the following multiple-choice question.
        The constellation ... is a bright W-shaped constellation in the northern sky.
        (A) Centaurus
        (B) Cygnus
        (C) Cassiopeia
        (D) Cepheus
        """
    ),
    "drop": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the Reading Comprehension Benchmark
        Requiring Discrete Reasoning Over Paragraphs (DROP), which assesses the ability to perform discrete
        reasoning and comprehend detailed information across multiple paragraphs.
        ## An example question from DROP:
        You will be asked to read a passage and answer a question.
        Passage:
        Non-nationals make up more than half of the population of Bahrain, with immigrants making up
        about 55% of the overall population. Of those, the vast majority come from South and Southeast Asia:
        according to various media reports and government statistics dated between 2005-2009 roughly 290,000
        Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians.
        Question: What two nationalities had the same number of people living in Bahrain between
        2005-2009?
        Answer [Not Given]: Pakistanis and Filipinos
        """
    ),
    "mgsm": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the Multilingual Grade School Math Benchmark
        (MGSM) which evaluates mathematical problem-solving abilities across various languages to ensure
        broad and effective multilingual performance.
        ## An example question from MGSM:
        **Question**: この数学の問題を解いてください。
        近所では、ペットのウサギの数がペットの犬と猫を合わせた数よりも12匹少ない。犬1匹あたり2匹
        の猫がおり、犬の数は60匹だとすると、全部で近所には何匹のペットがいますか?
        **Answer (Not Given)**: 348
        """
    ),
    "clrs_text": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the CLRS Text Benchmark,
        which evaluates the ability to understand and answer questions based on the content of the
        Introduction to Algorithms textbook by Cormen, Leiserson, Rivest, and Stein.
        """
    ),
    "salad_data": {},
    "simple_qa": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the SimpleQA benchmark, which evaluates
        question-answering abilities across a wide range of topics and difficulty levels.
        ## An example question from SimpleQA:
        **Question**: What is the capital of France?
        **Answer (Not Given)**: Paris
        """
    ),
    "math_500": dedent(
        """
        Your aim is to find an optimal multi-agent system performing well on the Math_500 benchmark, which evaluates
        mathematical problem-solving abilities across a wide range of topics and difficulty levels.
        This benchmark includes problems from algebra, geometry, calculus, number theory, and more,
        designed to test both computational skills and conceptual understanding.
        ## An example question from Math_500:
        **Question**: Solve the following mathematical problem.
        A car travels 100 miles at a speed of 50 mph. How long does it take to travel 100 miles?
        **Answer (Not Given)**: 2 hours
        """
    ),
}
