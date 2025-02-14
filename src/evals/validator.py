from base import System

from benchmarks.salad_data import SaladData
from benchmarks.truthful_qa import TruthfulQA
from benchmarks.anti_salad_data import AntiSaladData

from benchmarks import benchmark_registry


class Validator:

    def __init__(self, args, split="validation"):
        """
        Initializes the Validator class.

        Args:
            args: Arguments object containing configurations for the evaluator, including
            dataset file paths and model settings.
        """
        self.args = args
        self.benchmarks = benchmark_registry
        print(benchmark_registry)

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
