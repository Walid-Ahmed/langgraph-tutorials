# Component-level evaluation: test the `llm` node on its own, without running the graph.
#
# This script runs a LangSmith experiment whose target is not the compiled desk
# but ONE NODE of it: the `llm` node from make_llm_node(). For each golden
# question it calls the node once, takes the tool calls in its reply, and grades
# them with first_step_calls: the right tools, with the right arguments.
#
# Why this matters:
# - End-to-end ("black box") evaluation tells you THAT a run failed. Component
#   ("glass box") evaluation tells you WHERE. If first_step_calls fails, the
#   model asked for the wrong thing; if it passes and the answer is still wrong,
#   look at the tools or at how the model used their results.
# - A LangGraph node is a plain function from state to a state update, so testing
#   one in isolation needs no special machinery: call it with a state.
# - It's cheap: one model call per example and no judge. It only works this
#   simply because every golden question is decided in the first step; in a
#   multi-lap agent you'd feed the node the state from a recorded run.
#
# Before you run this:
# 1. Run run_experiment.py once first, so the dataset exists in LangSmith.
# 2. The same .env as run_experiment.py.
#
# Run it with (from the 12-Evaluation folder):
#    python component_eval.py v1
#    python component_eval.py v2
#
# Expected result: a link to the experiment, then the failing examples and the
# average. On our runs v1 failed stock-unknown and price-and-type (the node asked
# no tool at all) and scored 0.83; v2 scored 1.00. (After promote_example.py adds
# a 13th example, v1 scores 0.85: the new example passes.)

import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langchain_core.messages import HumanMessage
from langsmith import Client

from desk import make_llm_node
from evaluators import first_step_calls

DATASET = "corniche-cycles-desk"


def main(version: str):
    llm_node = make_llm_node(version)

    def first_step(inputs: dict) -> dict:
        update = llm_node({"messages": [HumanMessage(inputs["question"])]})
        reply = update["messages"][-1]
        return {"tool_calls": [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]}

    results = Client().evaluate(
        first_step,
        data=DATASET,
        evaluators=[first_step_calls],
        experiment_prefix=f"llm-node-{version}",
        metadata={"version": version, "component": "llm"},
        max_concurrency=4,
    )
    scores = []
    for row in results:
        result = row["evaluation_results"]["results"][0]
        scores.append(result.score)
        if result.score < 1:
            print(f"{row['example'].metadata['id']:<16}{result.comment}")
    print(f"{version}: first_step_calls {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
