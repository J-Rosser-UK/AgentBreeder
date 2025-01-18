import unittest
import sys

sys.path.append("src")
from unittest.mock import MagicMock
from dataclasses import dataclass
from evals.drop import (
    drop_metric,
)  # Replace 'your_module' with the actual module name where DROP is defined
import argparse


class TestDropMetric(unittest.TestCase):
    def test_exact_match_single_reference(self):
        sample = "Pakistanis, Filipinos"
        gold_reference = ["Pakistanis", "Filipinos"]
        em, f1 = drop_metric(sample, gold_reference)
        self.assertEqual(em, 1.0)
        self.assertEqual(f1, 100.0)

    def test2(self):
        sample = "duke of york", "king of england"

    # def test_exact_match_multiple_references(self):
    #     sample = "Pakistanis and Filipinos"
    #     reference = [
    #         "Pakistanis and Filipinos",
    #         "Filipinos and Pakistanis",
    #         "Bangladeshis and Indians",
    #     ]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 1.0)
    #     self.assertEqual(f1, 100.0)

    # def test_partial_match(self):
    #     sample = "Pakistanis"
    #     reference = ["Pakistanis and Filipinos"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     # F1 calculation:
    #     # Predicted tokens: {"pakistanis"}
    #     # Gold tokens: {"pakistanis", "filipinos"}
    #     # Intersection: {"pakistanis"} => 1
    #     # Precision: 1 / 1 = 1.0
    #     # Recall: 1 / 2 = 0.5
    #     # F1 = (2 * 1.0 * 0.5) / (1.0 + 0.5) = 0.666... * 100 = 66.67
    #     self.assertAlmostEqual(f1, 66.67, places=2)

    # def test_no_match(self):
    #     sample = "Indians"
    #     reference = ["Pakistanis and Filipinos"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     self.assertEqual(f1, 0.0)

    # def test_multiple_references_best_match(self):
    #     sample = "Pakistanis and Filipinos"
    #     reference = ["Pakistanis", "Filipinos", "Pakistanis and Filipinos"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 1.0)
    #     self.assertEqual(f1, 100.0)

    # def test_empty_sample(self):
    #     sample = ""
    #     reference = ["Pakistanis and Filipinos"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     self.assertEqual(f1, 0.0)

    # def test_empty_references(self):
    #     sample = "Pakistanis and Filipinos"
    #     reference = []
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     self.assertEqual(f1, 0.0)

    # def test_numeric_matching(self):
    #     sample = "290000 Indians and 125000 Bangladeshis"
    #     reference = ["290,000 Indians and 125,000 Bangladeshis"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 1.0)
    #     self.assertEqual(f1, 100.0)

    # def test_numeric_partial_matching(self):
    #     sample = "290000 Indians"
    #     reference = ["290,000 Indians and 125,000 Bangladeshis"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     # F1 calculation:
    #     # Predicted tokens: {"290000", "indians"}
    #     # Gold tokens: {"290000", "indians", "125000", "bangladeshis"}
    #     # Intersection: {"290000", "indians"} => 2
    #     # Precision: 2 / 2 = 1.0
    #     # Recall: 2 / 4 = 0.5
    #     # F1 = (2 * 1.0 * 0.5) / (1.0 + 0.5) = 0.666... * 100 = 66.67
    #     self.assertAlmostEqual(f1, 66.67, places=2)

    # def test_case_insensitivity_and_punctuation(self):
    #     sample = "pakistanis and filipinos!"
    #     reference = ["Pakistanis and Filipinos"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 1.0)
    #     self.assertEqual(f1, 100.0)

    # def test_multiple_correct_references(self):
    #     sample = "Pakistanis and Filipinos"
    #     reference = [
    #         "Filipinos and Pakistanis",
    #         "Pakistanis and Filipinos",
    #         "Filipinos, Pakistanis",
    #     ]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 1.0)
    #     self.assertEqual(f1, 100.0)

    # def test_partial_overlap_multiple_references(self):
    #     sample = "Pakistanis and Bangladeshis"
    #     reference = ["Pakistanis and Filipinos", "Bangladeshis and Indians"]
    #     em, f1 = drop_metric(sample, reference)
    #     self.assertEqual(em, 0.0)
    #     # F1 for first reference:
    #     # Predicted tokens: {"pakistanis", "bangladeshis"}
    #     # Gold tokens: {"pakistanis", "filipinos"}
    #     # Intersection: {"pakistanis"} => 1
    #     # Precision: 1 / 2 = 0.5
    #     # Recall: 1 / 2 = 0.5
    #     # F1 = (2 * 0.5 * 0.5) / (0.5 + 0.5) = 0.5 * 100 = 50.0
    #     #
    #     # F1 for second reference:
    #     # Predicted tokens: {"pakistanis", "bangladeshis"}
    #     # Gold tokens: {"bangladeshis", "indians"}
    #     # Intersection: {"bangladeshis"} => 1
    #     # Precision: 1 / 2 = 0.5
    #     # Recall: 1 / 2 = 0.5
    #     # F1 = 50.0
    #     #
    #     # Max F1 = 50.0
    #     self.assertAlmostEqual(f1, 50.0, places=2)


if __name__ == "__main__":
    unittest.main()
