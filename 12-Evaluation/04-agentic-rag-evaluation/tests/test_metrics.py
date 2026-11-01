# Unit tests for the metrics, the evaluator adapters and the CI gate.
#
# None of these call a model, Chroma or LangSmith, so they run in a second with no keys.
# Run them with (from the repository root):
#    python -m pytest -q 12-Evaluation/04-agentic-rag-evaluation/tests
# Expected result: all tests pass.

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.evaluators import context_recall_eval, keyword_coverage_eval, tool_call_accuracy_eval
from evals.gate import case_passed, find_regressions, summarize
from evals.metrics.heuristic_metrics import length_penalty, rouge
from evals.metrics.rag_metrics import context_precision, context_recall
from evals.metrics.rule_based_metrics import keyword_coverage, tool_call_accuracy
from evals.metrics.safety_metrics import bias_fairness_score, toxicity_score

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "data" / "golden_dataset.json"


def test_tool_call_accuracy():
    assert tool_call_accuracy("search_knowledge_base", "search_knowledge_base").score == 1.0
    assert tool_call_accuracy(None, "get_weather").score == 0.0
    assert tool_call_accuracy(None, None).score == 0.0  # "no tool" is never the expected first call here


def test_keyword_coverage_is_case_insensitive_and_names_what_is_missing():
    result = keyword_coverage("Healthcare generated 900,000 in 2025-Q2", ["Healthcare", "900000", "2025-Q2"])
    assert result.score == 0.667          # "900,000" does not contain "900000": formats matter
    assert result.note == "missing ['900000']"


def test_rouge_counts_reference_words_covered():
    assert rouge("Europe is handled by Sarah Lee.", "Sarah Lee covers Europe.").score == 0.75  # 3 of 4


def test_length_penalty():
    assert length_penalty("too short").score == 0.0
    assert length_penalty(" ".join(["word"] * 20)).score == 1.0


def test_safety_matches_whole_words_only():
    assert toxicity_score("You are an idiot").score == 0.0
    assert toxicity_score("Alex Martin has the right skill set").score == 1.0  # "skill" is not "kill"
    assert bias_fairness_score("This always happens").score == 0.0
    assert bias_fairness_score("I can call the sales rep for all small accounts").score == 1.0


def test_context_precision_and_recall():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc2", "doc4", "doc5"]
    assert context_precision(retrieved, relevant).score == 0.5
    assert context_recall(retrieved, relevant).score == 0.667


def test_evaluators_return_langsmith_feedback():
    outputs = {"answer": "Sarah Lee covers Europe.", "tool": "search_knowledge_base",
               "sources": ["crm_exports/sales_reps.csv"]}
    reference = {"expected_keywords": ["Sarah Lee", "Europe"], "expected_tool": "search_knowledge_base",
                 "expected_sources": ["crm_exports/sales_reps.csv"]}
    assert keyword_coverage_eval(outputs, reference) == {"key": "keyword_coverage", "score": 1.0,
                                                         "comment": "all keywords present"}
    assert tool_call_accuracy_eval(outputs, reference)["score"] == 1.0
    assert context_recall_eval(outputs, reference)["score"] == 1.0


def test_pass_rule_needs_both_correctness_and_the_right_tool():
    assert case_passed({"correctness": 0.8, "tool_call_accuracy": 1.0})
    assert not case_passed({"correctness": 0.7, "tool_call_accuracy": 1.0})
    assert not case_passed({"correctness": 1.0, "tool_call_accuracy": 0.0})


def test_summary_and_regressions():
    baseline = [{"id": "a", "passed": True, "scores": {"x": 1.0}}, {"id": "b", "passed": False, "scores": {"x": 0.0}}]
    current = [{"id": "a", "passed": False, "scores": {"x": 0.0}}, {"id": "b", "passed": True, "scores": {"x": 1.0}}]
    assert summarize(baseline)["pass_rate"] == summarize(current)["pass_rate"] == 0.5  # same rate...
    assert find_regressions(baseline, current) == ["a"]                                # ...but "a" broke


def test_golden_set_shape():
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 18
    for case in cases:
        assert case["expected_tool"] in {"search_knowledge_base", "get_weather"}, case["id"]
        assert all(not s.startswith("datasets/") for s in case["expected_sources"]), case["id"]


def test_review_queue_takes_flagged_runs_and_a_sample_of_the_rest():
    from review_queue import select_for_review
    runs = [{"question": "q1", "scores": {"answer_relevancy": 0.6}},
            {"question": "q2", "scores": {"answer_relevancy": 1.0, "toxicity": 1.0}},
            {"question": "q3", "scores": {"bias_fairness": 0.0}},
            {"question": "q4", "scores": {}},
            {"question": "q5", "scores": {"answer_relevancy": 0.9}}]
    chosen = select_for_review(runs, flag_below=0.8, sample_size=2, seed=14)
    assert [(r["question"], why) for r, why in chosen[:2]] == [("q1", "flagged: answer_relevancy"),
                                                              ("q3", "flagged: bias_fairness")]
    assert len(chosen) == 4 and all(why == "sampled" for _, why in chosen[2:])
    assert chosen == select_for_review(runs, flag_below=0.8, sample_size=2, seed=14)  # same seed, same sample


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_get_weather_formats_the_live_reading(monkeypatch):
    import src.tools as tools
    replies = {
        tools.GEOCODE_URL: {"results": [{"name": "London", "country": "United Kingdom",
                                         "latitude": 51.5, "longitude": -0.13}]},
        tools.FORECAST_URL: {"current": {"time": "2026-11-01T16:30", "temperature_2m": 11.0,
                                         "weather_code": 3, "wind_speed_10m": 10.1}},
    }
    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: _FakeResponse(replies[url]))
    assert tools.get_weather.invoke({"location": "London, UK"}) == \
        "London, United Kingdom: 11°C, overcast, wind 10 km/h (Open-Meteo, local time 2026-11-01T16:30)."


def test_get_weather_reports_unknown_places_and_outages(monkeypatch):
    import src.tools as tools
    monkeypatch.setattr(tools.requests, "get", lambda url, **kwargs: _FakeResponse({}))
    assert tools.get_weather.invoke({"location": "Atlantisville"}) == "No place called 'Atlantisville' was found."

    def offline(url, **kwargs):
        raise tools.requests.ConnectionError("no network")
    monkeypatch.setattr(tools.requests, "get", offline)
    assert tools.get_weather.invoke({"location": "London"}) == \
        "The weather service is unavailable right now (ConnectionError)."


def test_agent_reply_reads_back_the_retrieved_sources():
    from src.agents import _retrieved
    from src.tools import SOURCE_TAG
    tool_text = f"{SOURCE_TAG}crm_exports/sales_reps.csv]\nname: Sarah Lee, region: Europe" \
                f"\n\n---\n\n{SOURCE_TAG}crm_exports/accounts.csv]\nname: FinTrust"
    messages = [HumanMessage("Who covers Europe?"),
                AIMessage("", tool_calls=[{"name": "search_knowledge_base", "args": {"query": "Europe"}, "id": "1"}]),
                ToolMessage(tool_text, name="search_knowledge_base", tool_call_id="1")]
    sources, contexts = _retrieved(messages)
    assert sources == ["crm_exports/sales_reps.csv", "crm_exports/accounts.csv"]
    assert contexts[0] == "name: Sarah Lee, region: Europe"
