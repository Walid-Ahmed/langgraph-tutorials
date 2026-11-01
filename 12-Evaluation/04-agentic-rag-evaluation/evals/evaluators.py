# The nine metrics, wrapped as LangSmith evaluators.
#
# LangSmith calls each evaluator once per golden example, passing by name the
# target's `outputs` (what the agent did) and the example's `inputs` and
# `reference_outputs` (what it should have done). Each returns one piece of
# feedback: a key, a 0-1 score and a comment, stored on that example's trace.
# Imported by run_eval.py, not run.

from evals.metrics.heuristic_metrics import length_penalty, rouge
from evals.metrics.llm_as_judge_metrics import correctness_judge, relevance_judge
from evals.metrics.rag_metrics import context_precision, context_recall
from evals.metrics.result import MetricResult
from evals.metrics.rule_based_metrics import keyword_coverage, tool_call_accuracy
from evals.metrics.safety_metrics import toxicity_score


# Turn a MetricResult into the feedback LangSmith stores: a key, a score and a comment.
def _feedback(result: MetricResult) -> dict:
    return {"key": result.name, "score": result.score, "comment": result.note}


# Rule-based: are the expected keywords in the answer?
def keyword_coverage_eval(outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(keyword_coverage(outputs["answer"], reference_outputs["expected_keywords"]))


# Was the first tool the agent called the expected one?
def tool_call_accuracy_eval(outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(tool_call_accuracy(outputs["tool"], reference_outputs["expected_tool"]))


# Heuristic: how much of the expected answer's wording does the answer cover?
def rouge_eval(outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(rouge(outputs["answer"], reference_outputs["expected_answer"]))


# Is the answer at least 20 words long?
def length_penalty_eval(outputs: dict) -> dict:
    return _feedback(length_penalty(outputs["answer"]))


# RAG: how much of what the searches returned came from the expected files?
def context_precision_eval(outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(context_precision(outputs["sources"], reference_outputs["expected_sources"]))


# Did the searches return every expected file?
def context_recall_eval(outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(context_recall(outputs["sources"], reference_outputs["expected_sources"]))


# Safety: does the answer contain a toxic word?
def toxicity_eval(outputs: dict) -> dict:
    return _feedback(toxicity_score(outputs["answer"]))


# LLM-as-judge: does the answer match the golden answer, according to the judge?
def correctness_eval(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return _feedback(correctness_judge(inputs["query"], outputs["answer"], reference_outputs["expected_answer"]))


# Does the answer address the question, according to the judge?
def relevance_eval(inputs: dict, outputs: dict) -> dict:
    return _feedback(relevance_judge(inputs["query"], outputs["answer"]))


ALL_EVALUATORS = [
    keyword_coverage_eval, tool_call_accuracy_eval,
    rouge_eval, length_penalty_eval,
    context_precision_eval, context_recall_eval,
    toxicity_eval,
    correctness_eval, relevance_eval,
]
