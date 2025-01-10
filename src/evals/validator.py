import os
import importlib.util

from base import System
from tqdm import tqdm
from sqlalchemy.orm import Session

from textwrap import dedent
import asyncio
import logging

from evals.arc import ARC
from evals.mmlu import MMLU
from evals.drop import DROP
from evals.gpqa import GPQA
from evals.mgsm import MGSM
from evals.clrs_text import CLRSText
from .benchmark import AgentSystemException


class Validator:

    def __init__(self, args, split="validation"):
        """
        Initializes the Validator class.

        Args:
            args: Arguments object containing configurations for the evaluator, including
            dataset file paths and model settings.
        """
        self.args = args
        self.benchmarks = {
            "arc": ARC,  #          # 20 questions in validation set, 60 in test set
            # "gpqa": GPQA, #       # 32 questions in validation set, 166 in test set
            "mmlu": MMLU,  #        # 128 questions in validation set, 800 in test set
            # "drop": DROP, #       # 128 questions in validation set, 800 in test set
            # "mgsm": MGSM, #       # 128 questions in validation set, 800 in test set
            "clrs_text": CLRSText,
        }

        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args, split=split, shuffle=True, limit=self.args.n_evals
        )

    def validate(
        self,
        systems_for_validation: list[System],
    ):

        for i, system in tqdm(enumerate(systems_for_validation)):

            accuracy, ci_lower, ci_upper, median = self.benchmark.evaluate(
                system,
                i + 1,
                len(systems_for_validation),
                limit=self.args.n_evals,
            )

            system.update(
                system_fitness=accuracy,
            )
            system.update(
                ci_sample_size=self.args.n_evals,
            )

            system.update(
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_median=median,
                ci_confidence_level=0.95,
            )
