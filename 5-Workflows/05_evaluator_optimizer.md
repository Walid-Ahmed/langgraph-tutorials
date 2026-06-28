# 05. Evaluator-Optimizer

## Part 1 — Core Tutorial

An evaluator-optimizer workflow generates an output, checks it, then improves it if needed.

![Evaluator optimizer workflow](figures/workflow-06-evaluator-optimizer.png)

```mermaid
flowchart TD
    INPUT["input"] --> GENERATE["generate answer"]
    GENERATE --> EVAL["evaluate answer"]
    EVAL --> GOOD{"good enough?"}
    GOOD -->|yes| FINAL["final answer"]
    GOOD -->|no| OPTIMIZE["improve answer"]
    OPTIMIZE --> EVAL
```

## When To Use

Use this pattern when quality matters and the system should review or improve its own output.

Examples:

- writing assistant
- code review assistant
- answer quality checker

## Part 2 — Code Example That Reinforces The Concept

Placeholder for future LangGraph implementation.

## Code Explanation

TODO: Explain generation node, evaluator node, conditional retry loop, and stopping condition.
