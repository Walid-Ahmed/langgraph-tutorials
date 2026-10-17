# Local tests for fail-closed coverage and historical-dataset comparisons.
# Why: missing scores, failed targets or changed references must never pass CI.
# Prerequisite: pip install -r requirements-dev.txt (repository root).
# Run: python -m pytest -q 12-Evaluation/02-helpdesk-evaluation/test_regression_gate.py
# LangSmith calls are replaced with fakes; no API key or paid call is required.

from datetime import datetime, timezone
from types import SimpleNamespace as Obj
from unittest.mock import patch

import pytest
from langsmith import Client
from langsmith.evaluation import EvaluationResult

# Import the harness without starting background tracing workers.
with patch("langsmith.Client", lambda: Client(auto_batch_tracing=False)):
    import regression_gate as gate
    import run_experiment as harness
from gate_checks import DEFAULT_METRICS, validate_scores

SNAPSHOT = "2026-10-17T10:00:00+00:00"


def scores(value=1):
    return {"boundary": dict.fromkeys(DEFAULT_METRICS, value)}


@pytest.mark.parametrize("bad", [{}, {"boundary": {}}, {"other": {}},
                                    {"boundary": {"correct_tools": None}},
                                    scores(float("nan")), scores(float("inf")), scores("1")])
def test_missing_or_invalid_scores_fail(bad):
    with pytest.raises(ValueError):
        validate_scores(bad, {"boundary"}, DEFAULT_METRICS)


@pytest.mark.parametrize("ids,metrics", [(set(), DEFAULT_METRICS), ({"boundary"}, ()),
                                        ({"boundary"}, ("correct_tools", "correct_tools"))])
def test_empty_or_duplicate_configuration_fails(ids, metrics):
    with pytest.raises(ValueError):
        validate_scores(scores(), ids, metrics)


@pytest.mark.parametrize("candidate,expected", [(scores(), 0), (scores(0), 1), ({}, 1),
                                                ({"boundary": {"correct_tools": 1}}, 1)])
def test_gate_pins_snapshot_and_requires_complete_candidate(monkeypatch, candidate, expected):
    monkeypatch.setattr(gate, "baseline_scores", lambda *a: ("reviewed", SNAPSHOT, scores()))
    called = {}

    def run(version, **kwargs):
        called.update(kwargs)
        return candidate

    monkeypatch.setattr(gate, "run", run)
    assert gate.gate("v2", "v3") == expected
    assert called["dataset_version"] == SNAPSHOT


def test_route_only_requires_explicit_metric_selection(monkeypatch):
    monkeypatch.setattr(gate, "baseline_scores", lambda *a: ("reviewed", SNAPSHOT, scores()))
    monkeypatch.setattr(gate, "run", lambda *a, **kw: {"boundary": {"correct_tools": 1}})
    assert gate.gate("v2", "v3", metrics=("correct_tools",)) == 0


@pytest.fixture
def baseline(monkeypatch):
    example = Obj(id="example-uuid", metadata={"id": "boundary"})
    run = Obj(reference_example_id=example.id, error=None,
              feedback_stats={m: {"avg": 1} for m in DEFAULT_METRICS})
    project = Obj(name="desk-v2-reviewed", metadata={"version": "v2", "dataset_version": SNAPSHOT},
                  reference_dataset_id="dataset-uuid", start_time=datetime.now(timezone.utc))
    calls = []
    client = Obj(list_projects=lambda **kw: [project], read_project=lambda **kw: project,
                 read_dataset=lambda **kw: Obj(id="dataset-uuid"),
                 list_examples=lambda **kw: calls.append(kw) or [example],
                 list_runs=lambda **kw: [run])
    monkeypatch.setattr(gate, "client", client)
    return Obj(client=client, project=project, run=run, example=example, calls=calls)


def test_baseline_reads_historical_examples(baseline):
    assert gate.baseline_scores("v2") == ("desk-v2-reviewed", SNAPSHOT, scores())
    assert baseline.calls == [{"dataset_name": gate.DATASET, "as_of": SNAPSHOT}]


@pytest.mark.parametrize("problem", ["no project", "no snapshot", "no runs", "failed run",
                                     "no metric", "unknown example", "repeated run", "bad id"])
def test_incomplete_baseline_fails(baseline, problem):
    if problem == "no project":
        baseline.client.list_projects = lambda **kw: []
    elif problem == "no snapshot":
        baseline.project.metadata.pop("dataset_version")
    elif problem == "no runs":
        baseline.client.list_runs = lambda **kw: []
    elif problem == "failed run":
        baseline.run.error = "target failed"
    elif problem == "no metric":
        baseline.run.feedback_stats.pop("judge_reply")
    elif problem == "unknown example":
        baseline.run.reference_example_id = "unknown"
    elif problem == "repeated run":
        baseline.client.list_runs = lambda **kw: [baseline.run, baseline.run]
    else:
        baseline.example.metadata = {}
    with pytest.raises(ValueError):
        gate.baseline_scores("v2")


def test_can_pin_exact_baseline(baseline):
    assert gate.baseline_scores("v2", "desk-v2-reviewed")[0] == "desk-v2-reviewed"
    baseline.project.reference_dataset_id = "another-dataset"
    with pytest.raises(ValueError):
        gate.baseline_scores("v2", "desk-v2-reviewed")


@pytest.fixture
def target_rows(monkeypatch):
    example = Obj(metadata={"id": "boundary"})
    row = {"example": example, "run": Obj(error=None),
           "evaluation_results": {"results": [EvaluationResult(key="correct_tools", score=1)]}}
    recorded = {}
    monkeypatch.setattr(harness, "build_desk", lambda version: object())
    client = Obj(list_examples=lambda **kw: recorded.update(kw) or [example],
                 evaluate=lambda *a, **kw: recorded.update(kw) or [row])
    monkeypatch.setattr(harness, "client", client)
    return Obj(row=row, example=example, recorded=recorded)


def test_candidate_uses_frozen_examples(target_rows):
    assert harness.run("v3", dataset_version=SNAPSHOT) == {"boundary": {"correct_tools": 1}}
    assert target_rows.recorded["as_of"] == SNAPSHOT
    assert target_rows.recorded["data"] == [target_rows.example]


@pytest.mark.parametrize("failure", ["target", "evaluator", "duplicate"])
def test_failed_or_duplicate_candidate_rows_rejected(target_rows, monkeypatch, failure):
    if failure == "target":
        target_rows.row["run"].error = "failed"
    elif failure == "evaluator":
        target_rows.row["evaluation_results"]["results"] = [
            EvaluationResult(key="correct_tools", extra={"error": True})]
    else:
        monkeypatch.setattr(harness.client, "evaluate", lambda *a, **kw: [target_rows.row] * 2)
    with pytest.raises(ValueError):
        harness.run("v3", dataset_version=SNAPSHOT)


def test_explicit_baseline_rejects_wrong_version(baseline):
    baseline.project.metadata["version"] = "v1"
    with pytest.raises(ValueError):
        gate.baseline_scores("v2", "desk-v2-reviewed")


def test_baseline_rejects_duplicate_metadata_ids(baseline):
    baseline.client.list_examples = lambda **kw: [baseline.example, Obj(id="other", metadata={"id": "boundary"})]
    with pytest.raises(ValueError):
        gate.baseline_scores("v2")


def test_failed_baseline_stops_before_candidate(monkeypatch):
    def incomplete(*args):
        raise ValueError("missing baseline score")

    def unexpected(*args, **kwargs):
        pytest.fail("candidate must not run with an incomplete baseline")

    monkeypatch.setattr(gate, "baseline_scores", incomplete)
    monkeypatch.setattr(gate, "run", unexpected)
    assert gate.gate("v2", "v3") == 1
