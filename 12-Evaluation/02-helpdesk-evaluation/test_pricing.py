# Unit tests and a code-evaluator test for the same price boundary.
# Why: exercise the real pricing tool without an LLM, API key or LangSmith call.
# Prerequisite: pip install -r requirements-dev.txt (repository root).
# Run from the repository root: python -m pytest -q 12-Evaluation/02-helpdesk-evaluation/test_pricing.py
# Expected: 4 passed. The v3 example deliberately keeps its teaching bug.

import pytest

from desk import rental_quote, rental_quote_refactored
from evaluators import facts_present


@pytest.mark.parametrize("hours,expected", [(4, "EGP 100"), (5, "EGP 100"), (6, "EGP 120")])
def test_price_at_discount_boundary(hours, expected):
    quote = rental_quote.invoke({"bike_type": "kids", "hours": hours})
    assert quote.startswith(f"kids bike, {hours} h: {expected}")


def test_evaluator_catches_the_same_boundary_bug():
    args = {"bike_type": "kids", "hours": 5}
    reference = {"must_include": ["EGP 100"]}
    good = rental_quote.invoke(args)
    broken = rental_quote_refactored.invoke(args)
    assert facts_present({"answer": good}, reference)["score"] == 1
    assert facts_present({"answer": broken}, reference)["score"] == 0
