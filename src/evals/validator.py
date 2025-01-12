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
from evals.salad_data import SaladData
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
            "salad_data": SaladData,
        }

        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args, split=split, shuffle=True, limit=self.args.n_evals
        )

    def validate(
        self,
        systems_for_validation: list[System],
    ):

        model_metrics = self.benchmark.evaluate(
            systems_for_validation,
            limit=self.args.n_evals,
        )

        # You can now access each model’s metrics via model_metrics
        for model, metrics in model_metrics.items():
            print(f"Model: {model}")
            print(f"  accuracy: {metrics['accuracy']}")
            print(f"  ci_lower: {metrics['ci_lower']}")
            print(f"  ci_upper: {metrics['ci_upper']}")
            print(f"  median:   {metrics['median']}")

            for system in systems_for_validation:
                if str(system.system_id) == model.split("||")[1]:
                    system.update(
                        system_fitness=metrics["median"],
                        ci_sample_size=self.args.n_evals,
                        ci_lower=metrics["ci_lower"],
                        ci_upper=metrics["ci_upper"],
                        ci_median=metrics["median"],
                        ci_confidence_level=0.95,
                    )
