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
from evals.simple_qa import SimpleQA
from evals.math_500 import Math500
from evals.math_ import Math
from evals.mmlu_cf import MMLUCF
from evals.anti_salad_data import AntiSaladData
from evals.truthful_qa import TruthfulQA
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
            "arc": ARC,
            "gpqa": GPQA,
            "mmlu": MMLU,
            "drop": DROP,
            "mgsm": MGSM,
            "clrs_text": CLRSText,
            "salad_data": SaladData,
            "anti_salad_data": AntiSaladData,
            "truthful_qa": TruthfulQA,
            "simple_qa": SimpleQA,
            "math_500": Math500,
            "math": Math,
            "mmlu_cf": MMLUCF,
        }

        self.benchmark = self.benchmarks[args.benchmark](
            args=self.args, split=split, shuffle=False, limit=self.args.n_evals
        )

    def validate(self, systems_for_validation: list[System], log_d="logs"):
        if len(systems_for_validation) == 0:
            return
        model_metrics = self.benchmark.evaluate(
            systems_for_validation, limit=self.args.n_evals, log_d=log_d
        )

        # You can now access each model’s metrics via model_metrics
        for model, task_metrics in model_metrics.items():
            for task, metrics in task_metrics.items():

                print(f"Model: {model}")
                print(f"Task: {task}")
                print(f"  accuracy: {metrics['accuracy']}")
                print(f"  ci_lower: {metrics['ci_lower']}")
                print(f"  ci_upper: {metrics['ci_upper']}")
                print(f"  median:   {metrics['median']}")

                for system in systems_for_validation:
                    if str(system.system_id) == model.split("||")[1]:
                        if task == self.benchmarks[self.args.benchmark].__name__:
                            system.update(
                                system_fitness=metrics["median"],
                                system_capability_ci_sample_size=self.args.n_evals,
                                system_capability_ci_lower=metrics["ci_lower"],
                                system_capability_ci_upper=metrics["ci_upper"],
                                system_capability_ci_median=metrics["median"],
                                system_capability_ci_confidence_level=0.95,
                            )
                        elif (
                            task == SaladData.__name__ or task == AntiSaladData.__name__
                        ):
                            system.update(
                                system_safety_ci_sample_size=self.args.n_evals,
                                system_safety_ci_lower=metrics["ci_lower"],
                                system_safety_ci_upper=metrics["ci_upper"],
                                system_safety_ci_median=metrics["median"],
                                system_safety_ci_confidence_level=0.95,
                            )
