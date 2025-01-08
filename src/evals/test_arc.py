import sys

sys.path.append("src")

from base import System
import unittest
from evals.arc import EvaluateARC
from inspect_ai.dataset import Sample
from textwrap import dedent
import argparse
from tqdm import tqdm
import uuid
from base import initialize_session
from prompts.initial_population import COT_SC
import re


class TestEvaluateARC(unittest.TestCase):

    def setUp(self):
        self.system = System(
            system_name="arc_test_system",
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

        self.args = parser.parse_args()

        self.evaluator = EvaluateARC(self.args)
        self.session, _ = initialize_session(self.args.db_name)

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

    def test_evaluate_system(self):
        system = System(
            system_name="arc_test_system",
            system_id=str(uuid.uuid4()),
            system_code=COT_SC["code"],
        )
        # system = self.session.query(System).first()
        accuracy = self.evaluator.evaluate(system, limit=1)
        print("accuracy", accuracy)
        self.assertIsInstance(accuracy, float)


#     def test_extract_function(self):
#         function = """ alsdfakjdslk
# asdkfljsfjlj
# def transform(grid: list[list[int]]) -> list[list[int]]:
#     transformed_grid = [[0] * 9 for _ in range(9)]
#     for i in range(3):
#         for j in range(3):
#             transformed_grid[i*3:(i+1)*3][j*3:(j+1)*3] = [[grid[i][j]] * 3 for _ in range(3)]
#     return transformed_grid
#       asdlfjasdlj
# """
#         output = EvaluateARC.extract_function_code(function)

#         expected = """def transform(grid: list[list[int]]) -> list[list[int]]:
#     transformed_grid = [[0] * 9 for _ in range(9)]
#     for i in range(3):
#         for j in range(3):
#             transformed_grid[i*3:(i+1)*3][j*3:(j+1)*3] = [[grid[i][j]] * 3 for _ in range(3)]
#     return transformed_grid
# """
#         self.assertEqual(output, expected)
#         print(output)


if __name__ == "__main__":
    unittest.main()
