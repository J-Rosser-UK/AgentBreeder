import os
import importlib.util

from base import System
from tqdm import tqdm
from sqlalchemy.orm import Session

from textwrap import dedent
import asyncio
import logging

from evals.arc import EvaluateARC
from evals.mmlu import EvaluateMMLU
from evals.drop import EvaluateDROP
from evals.gpqa import EvaluateGPQA
from evals.mgsm import EvaluateMGSM
from evals.clrs_text import EvaluateCLRSText
from .inspect_base import AgentSystemException


class Evaluator:

    def __init__(self, args):
        """
        Initializes the Evaluator class.

        Args:
            args: Arguments object containing configurations for the evaluator, including
            dataset file paths and model settings.
        """
        self.args = args
        self.benchmarks = {
            "arc": EvaluateARC(
                args=self.args, split="validation", shuffle=True, limit=20
            ),  # 20 questions in validation set, 60 in test set
            # "gpqa": EvaluateGPQA(args=self.args, split="validation", shuffle=True, limit=32), # 32 questions in validation set, 166 in test set
            "mmlu": EvaluateMMLU(
                args=self.args, split="validation", shuffle=True, limit=20
            ),  # 128 questions in validation set, 800 in test set
            # "drop": EvaluateDROP(args=self.args, split="validation", shuffle=True, limit=20), # 128 questions in validation set, 800 in test set
            # "mgsm": EvaluateMGSM(args=self.args, split="validation", shuffle=True, limit=20), # 128 questions in validation set, 800 in test set
            "clrs_text": EvaluateCLRSText(
                args=self.args, split="validation", shuffle=True, limit=20
            ),
        }

        self.benchmark = self.benchmarks[args.benchmark]

    def inspect_evaluate(
        self,
        systems_for_evaluation: list[System],
    ):

        for i, system in tqdm(enumerate(systems_for_evaluation)):

            accuracy, ci_lower, ci_upper, median = self.benchmark.evaluate(
                system,
                i + 1,
                len(systems_for_evaluation),
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
