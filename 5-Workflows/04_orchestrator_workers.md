# 04. Orchestrator-Workers

## Part 1 — Core Tutorial

An orchestrator-worker workflow uses one central node to plan or delegate work to specialized worker nodes. The orchestrator decides the subtasks; the workers focus on execution.

![Orchestrator worker workflow](figures/workflow-05-orchestrator-workers.png)

```mermaid
flowchart TD
    TASK["task"] --> ORCH["orchestrator"]
    ORCH --> W1["worker 1"]
    ORCH --> W2["worker 2"]
    ORCH --> W3["worker 3"]
    W1 --> ORCH
    W2 --> ORCH
    W3 --> ORCH
    ORCH --> FINAL["final result"]
```

## When To Use

Use this pattern when the task has multiple parts and one controller should decide who does what. It is especially useful when the number or type of subtasks is not known upfront.

Examples:

- research assistant
- multi-step report generation
- task planner with specialist workers

## Part 2 — Code Example That Reinforces The Concept

No runnable code yet. This page is the concept guide for a future orchestrator-worker example.

## Code Explanation

Future code should show an orchestrator creating a task plan, workers returning partial results, and a final synthesis step. State should make the plan and collected worker outputs visible.
