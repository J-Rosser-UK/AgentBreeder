from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from .benchmark import Benchmark
import json
from .metrics import ci_lower, ci_upper, median
from inspect_ai.scorer import (
    accuracy,
    Score,
    scorer,
)
import re
import string
import numpy as np
from scipy.optimize import linear_sum_assignment

#
# Below are helper functions and DROP logic. The only significant change
# is how we combine the list of gold references in `drop_metric`.
#


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> list[str]:
    return re.split(r" |-", text)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _normalize_answer(text: str) -> str:
    """
    Lower text, remove punctuation, articles, and extra whitespace,
    and normalize numbers to float form.
    """
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _compute_f1(predicted_bag: set[str], gold_bag: set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    # Avoid division by zero
    precision = intersection / float(len(predicted_bag)) if predicted_bag else 1.0
    recall = intersection / float(len(gold_bag)) if gold_bag else 1.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall / (precision + recall)) * 100


def _match_numbers_if_present(gold_bag: set[str], predicted_bag: set[str]) -> bool:
    """
    If there are any numbers in the gold, at least one must match in the predicted.
    Otherwise, we allow them to match just by text.
    """
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    if not gold_numbers or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _answer_to_bags(
    answer: Union[str, list[str], tuple[str, ...]]
) -> tuple[list[str], list[set[str]]]:
    """
    Turn an answer (or list of answers) into a list of normalized strings and
    a parallel list of token sets (bags).
    """
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]

    normalized_spans: list[str] = []
    token_bags: list[set[str]] = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> list[float]:
    """
    Takes gold and predicted answer sets, finds the optimal 1-1 alignment
    for maximum F1 across all answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # If numbers are present, ensure they match
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def get_drop_metrics(
    predicted: Union[str, list[str], tuple[str, ...]],
    gold: Union[str, list[str], tuple[str, ...]],
) -> tuple[float, float]:
    """
    Compute official DROP exact match (EM) and F1 for a single predicted answer
    vs. a single gold answer (each of which can be a string or list of strings).
    """
    predicted_spans, predicted_bags = _answer_to_bags(predicted)
    gold_spans, gold_bags = _answer_to_bags(gold)

    # Exact match by string sets (ignoring order)
    if set(predicted_spans) == set(gold_spans) and len(predicted_spans) == len(
        gold_spans
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    # F1 across aligned sets
    f1_per_bag = _align_bags(predicted_bags, gold_bags)
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def drop_metric(sample: str, reference: list[str]) -> tuple[float, float]:
    """
    Previously, we iterated over the reference list. Now, to treat the entire
    gold reference as one multi-token set (so that "Pakistanis, Filipinos" will
    get EM=1.0 vs. gold=["Pakistanis", "Filipinos"]), we unify the gold reference
    into a single comma-separated string and compare once.

    This ensures that if the predicted answer is "Pakistanis, Filipinos" and the
    gold reference is ["Pakistanis", "Filipinos"], we get full credit.
    """
    # If no valid gold references, return 0
    if not reference:
        return 0.0, 0.0

    # Combine all items in reference into a single comma-separated string.
    # e.g. reference = ["Pakistanis", "Filipinos"] => "Pakistanis, Filipinos"
    gold_answer = ", ".join(ref.strip() for ref in reference if ref.strip())
    if not gold_answer:
        return 0.0, 0.0

    # Compute official DROP metrics for the single combined gold
    em, f1 = get_drop_metrics(sample, gold_answer)
    return em, f1


class DROP(Benchmark):
    """
    A benchmark for the DROP dataset. We load passages/questions, then ask the model
    to produce an answer, and we score it using the official DROP F1 metric.
    """

    def __init__(
        self,
        args=None,
        split: Union[Literal["validation"], Literal["test"]] = "validation",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:
        self.split = split
        self.args = args

        split_mapping = {
            "validation": "train",
            "test": "validation",
        }

        self.dataset = self.filtered_hf_dataset(
            path="ucinlp/drop",
            name="default",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=self.args.random_seed,
            limit=limit,
        )

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a record containing a passage/question into a `Sample` object.
        Embed a short example prompt style as in the openai evals snippet.
        """

        passage = record["passage"]
        question = record["question"]
        # gold answers (list of strings)
        target = json.dumps(record["answers_spans"]["spans"])

        prompt = dedent(
            f"""
        You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.

        # Example
        Passage:
        Non-nationals make up more than half of the population of Bahrain, with immigrants making up
        about 55% of the overall population. Of those, the vast majority come from South and Southeast Asia:
        according to various media reports and government statistics dated between 2005-2009 roughly 290,000
        Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians.
        Question: What two nationalities had the same number of people living in Bahrain between 2005-2009?
        Answer: Pakistanis and Filipinos

        # Your Task
        Passage:
        {passage}

        Question:
        {question}

        OUTPUT ANSWER FORMAT: Provide your final answer as a single value e.g. a string, a single number (converted to a string), equation, or a comma separated list of those (e.g. "Pakistanis, Filipinos"). Don't include units if they aren't required.
        """
        ).strip()

        return Sample(
            input=prompt,
            target=target,
            metadata={
                "section_id": record["section_id"],
                "query_id": record["query_id"],
            },
        )

    @task
    def match_task(self):
        return Task(
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.match_solver(),
            scorer=self.span_match(),
            config=GenerateConfig(temperature=0.5),
        )

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def span_match():
        """
        This scorer uses the updated DROP F1 logic. Notably, 'drop_metric' now combines
        a list of gold reference strings into one comma-separated gold string, so that
        a predicted comma-separated list will get perfect credit if it contains exactly
        those tokens.
        """

        async def score(state, target):
            # If the model output starts with "error", mark as zero
            if state.output.completion.lower().startswith("error"):
                return Score(
                    name="span_match",
                    value=0.0,
                    answer=state.output.completion,
                    explanation="Error in model response.",
                )

            # Gold answers:
            gold_answers = json.loads(target.text)  # List of strings
            # Model's predicted answer (raw text):
            predicted_answer = state.output.completion

            # Compute DROP-based EM/F1
            em, f1 = drop_metric(predicted_answer, gold_answers)
            # The F1 returned is in [0..100], convert to [0..1] to match typical 0-1 range
            f1_scaled = f1 / 100.0

            return Score(
                name="span_match",
                value=f1_scaled,
                answer=predicted_answer,
                explanation=f"EM: {em}, F1: {f1} (scaled to {f1_scaled})",
            )

        return score
