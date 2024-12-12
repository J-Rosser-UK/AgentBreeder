import sys

sys.path.append("src")

from base import Framework
import unittest
from evals.mmlu import EvaluateMMLU
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm


class TestEvaluateMMLU(unittest.TestCase):

    def setUp(self):
        self.framework = Framework(
            framework_name="test_framework",
            framework_id="test_id",
            framework_code=dedent(
                """
            def forward(self, task):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--db_name", type=str, default="illuminator.db")

        self.args = parser.parse_args()

        self.evaluator = EvaluateMMLU(self.args)

    def test_record_to_sample(self):
        record = {
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": 2,
            "subject": "Geography",
        }
        sample = self.evaluator._record_to_sample(record)
        expected_prompt = dedent(
            """
            Answer the following multiple choice question.

            What is the capital of France?
            (A) Berlin
            (B) Madrid
            (C) Paris
            (D) Rome

            Provide your answer as a single letter in the range A-D.
        """
        ).strip()
        self.assertEqual(sample.input, expected_prompt)
        self.assertEqual(sample.target, "C")
        self.assertEqual(sample.metadata["subject"], "Geography")

    # def test_evaluate(self):
    #     accuracy = self.evaluator.evaluate(self.framework, limit=1000)
    #     self.assertIsInstance(accuracy, float)

    def test_evaluate_multiple(self):

        framework_5 = Framework(
            framework_name="test_framework",
            framework_id="test_id",
            framework_code="""def forward(self, task: str) -> str:

    # import time
    # time.sleep(5)
    
    return "C"
""",
        )

        framework_10 = Framework(
            framework_name="test_framework",
            framework_id="test_id",
            framework_code="""def forward(self, task: str) -> str:

    # import time
    # time.sleep(10)
    
    return "C"
""",
        )
        frameworks = [framework_5, framework_10]

        for framework in tqdm(frameworks, total=len(frameworks)):
            evaluator = EvaluateMMLU(self.args)
            accuracy = evaluator.evaluate(framework, limit=500)
            self.assertIsInstance(accuracy, float)


if __name__ == "__main__":
    unittest.main()
