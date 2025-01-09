import sys

sys.path.append("src")

from base import System
import unittest
from evals.clrs_text import EvaluateCLRS
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from prompts.initial_population import COT_SC
import re
import asyncio


class TestEvaluateCLRS(unittest.TestCase):

    def setUp(self):
        self.system = System(
            system_name="clrs_text_test_system",
            system_id="test_id",
            system_code=dedent(
                """
            async def forward(self, task):
                return "A"
            """
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--db_name", type=str, default="illuminator.db")
        parser.add_argument("--random_seed", type=int, default=42)

        self.args = parser.parse_args()

        self.session, _ = initialize_session()

    def test_record_to_sample(self):
        self.evaluator = EvaluateCLRS(args=self.args, split="validation", limit=1)


if __name__ == "__main__":
    unittest.main()
