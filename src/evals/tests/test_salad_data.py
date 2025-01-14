import sys

sys.path.append("src")

from base import System
import unittest
from evals.salad_data import SaladData
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from prompts.initial_population import COT_SC
import re
import asyncio


class TestSaladData(unittest.TestCase):

    def setUp(self):
        self.system = System(
            system_name="salad_data_test_system",
            system_id="test_id",
            system_code=dedent(
                """
            async def forward(self, task):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--random_seed", type=int, default=42)

        self.args = parser.parse_args()

    def test_record_to_sample(self):
        for session in initialize_session():
            systems_for_validation = session.query(System).limit(1).all()
            print(systems_for_validation[0].system_id)
            self.evaluator = SaladData(
                args=self.args, split="test", shuffle=False, limit=100
            )
            print([sample.input[30] for sample in self.evaluator.dataset])


if __name__ == "__main__":
    unittest.main()
