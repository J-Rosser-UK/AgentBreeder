import sys

sys.path.append("src")

from base import System
import unittest
from benchmarks.arc import ARC
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from prompts.initial_population import COT_SC
import re
import asyncio


class TestARC(unittest.TestCase):

    def setUp(self):
        self.system = System(
            system_name="arc_test_system",
            system_id="test_id",
            system_code=dedent(
                """
            async def forward(self, task, required_answer_format):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--db_name", type=str, default="illuminator.db")
        parser.add_argument("--random_seed", type=int, default=42)

        self.args = parser.parse_args()

        self.evaluator = ARC(args=self.args, split="validation", limit=1)

    # def test_record_to_sample(self):
    #     record = {
    #         "train": [
    #             {
    #                 "input": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #                 "output": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #             },
    #             {
    #                 "input": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #                 "output": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 1, 1, 1, 1, 1, 1, 8, 1, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 8, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #             },
    #         ],
    #         "test": [
    #             {
    #                 "input": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #                 "output": [
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                     [0, 0, 0, 1, 1, 1, 1, 1, 8, 1, 1, 1, 0],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 ],
    #             }
    #         ],
    #         "id": "e7639916",
    #     }
    #     sample = self.evaluator._record_to_sample(record)
    #     print(sample)

    # def test_evaluate_system(self):
    #     system = System(
    #         system_name="arc_test_system",
    #         system_id=str(uuid.uuid4()),
    #         system_code=COT_SC["code"],
    #     )

    #     accuracy = self.evaluator.evaluate(system, limit=1)
    #     print("accuracy", accuracy)
    #     self.assertIsInstance(accuracy, float)

    # def test_forward_pass(self):
    #     system = System(
    #         system_name="arc_test_system",
    #         system_id=str(uuid.uuid4()),
    #         system_code=COT_SC["code"],
    #     )
    #     output = asyncio.run(self.evaluator.forward_pass(system))

    #     print(output)

    def test_record_to_sample_size(self):
        self.evaluator = ARC(args=self.args, split="test", shuffle=False, limit=100)
        print([sample.input[-500] for sample in self.evaluator.dataset])


if __name__ == "__main__":
    unittest.main()
