# Exercise Solutions

Solutions to the exercises from tutorials 1–6. Each file is self-contained and runnable from the repo root.

## Structure

| Folder | Tutorial |
|--------|----------|
| `1-basics/` | [1. LangGraph Basics](../1-Langgraph%20basics/README.md) |
| `2-reducers/` | [2. Reducers](../2-Reducer/README.md) |
| `3-llm-messages/` | [3. LLM Messages](../3_LLM_Messages/README.md) |
| `4-conditional-edges/` | [4. Conditional Edges](../4-Conditional%20Edges/README.md) |
| `5-workflows/` | [5. Workflows](../5-Workflows/README.md) |
| `6-agents/` | [6. Agents](../6-Agents/README.md) |

## Running a solution

```bash
python "Exercise-Solutions/1-basics/ex1_second_node.py"
```

Solutions that call the OpenAI API require a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

## Solutions index

### 1. Basics
| File | Exercise |
|------|----------|
| `1-basics/ex1_second_node.py` | Add a `reverse_node` after `process` |
| `1-basics/ex2_visited_nodes.py` | Track visited nodes with a list reducer |
| `1-basics/ex3_input_validation.py` | Validate input before processing |

### 2. Reducers
| File | Exercise |
|------|----------|
| `2-reducers/ex1_max_reducer.py` | Keep the highest value seen |
| `2-reducers/ex2_deduplicate_list.py` | Append without duplicates |
| `2-reducers/ex3_two_nodes_accumulate.py` | Accumulate count across two nodes |

### 3. LLM Messages
| File | Exercise |
|------|----------|
| `3-llm-messages/ex1_multi_turn.py` | Two-message history, verify recall |
| `3-llm-messages/ex2_system_prompt.py` | Add a system prompt |
| `3-llm-messages/ex3_messages_state.py` | Extend MessagesState with turn_count |

### 4. Conditional Edges
| File | Exercise |
|------|----------|
| `4-conditional-edges/ex1_three_branches.py` | Add a "needs review" branch |
| `4-conditional-edges/ex2_retry_loop.py` | Loop on retry with attempt limit |
| `4-conditional-edges/ex3_llm_grader.py` | Replace keyword scoring with an LLM call |

### 5. Workflows
| File | Exercise |
|------|----------|
| `5-workflows/ex1_tone_step.py` | Extend prompt chain with a casual tone step |
| `5-workflows/ex2_parallel_haiku.py` | Add a haiku branch to the parallelization example |
| `5-workflows/ex3_structured_router.py` | Classify input and route to a specialist node |

### 6. Agents
| File | Exercise |
|------|----------|
| `6-agents/ex1_word_count_tool.py` | Add a word_count tool |
| `6-agents/ex2_iteration_cap.py` | Cap the agent loop at 5 iterations |
| `6-agents/ex3_conversation_memory.py` | Multi-turn agent with carry-over history |


## Graph Gallery

Each solution script calls `plot_graph(...)` when it runs. The diagrams below show the graph shape for each exercise before you open the code.

### 1. Basics

`1-basics/ex1_second_node.py`

```mermaid
flowchart LR
    START([START]) --> PROCESS["process"] --> REVERSE["reverse"] --> END([END])
```

`1-basics/ex2_visited_nodes.py`

```mermaid
flowchart LR
    START([START]) --> PROCESS["process"] --> REVERSE["reverse"] --> END([END])
```

`1-basics/ex3_input_validation.py`

```mermaid
flowchart TD
    START([START]) --> VALIDATE["validate"]
    VALIDATE --> ROUTE{"valid input?"}
    ROUTE -->|yes| PROCESS["process"] --> END([END])
    ROUTE -->|no| END
```

### 2. Reducers

`2-reducers/ex1_max_reducer.py`

```mermaid
flowchart LR
    START([START]) --> UPDATE["update"] --> END([END])
```

`2-reducers/ex2_deduplicate_list.py`

```mermaid
flowchart LR
    START([START]) --> UPDATE["update"] --> END([END])
```

`2-reducers/ex3_two_nodes_accumulate.py`

```mermaid
flowchart LR
    START([START]) --> A["node_a"] --> B["node_b"] --> END([END])
```

### 3. LLM Messages

`3-llm-messages/ex1_multi_turn.py`

```mermaid
flowchart LR
    START([START]) --> CHATBOT["chatbot"] --> END([END])
```

`3-llm-messages/ex2_system_prompt.py`

```mermaid
flowchart LR
    START([START]) --> CHATBOT["chatbot"] --> END([END])
```

`3-llm-messages/ex3_messages_state.py`

```mermaid
flowchart LR
    START([START]) --> CHATBOT["chatbot"] --> END([END])
```

### 4. Conditional Edges

`4-conditional-edges/ex1_three_branches.py`

```mermaid
flowchart TD
    START([START]) --> GRADE["grade_node"]
    GRADE --> ROUTE{"score"}
    ROUTE -->|>=70| PASS["pass_node"] --> END([END])
    ROUTE -->|60-69| REVIEW["review_node"] --> END
    ROUTE -->|<60| RETRY["retry_node"] --> END
```

`4-conditional-edges/ex2_retry_loop.py`

```mermaid
flowchart TD
    START([START]) --> GRADE["grade_node"]
    GRADE --> ROUTE{"pass, retry, or max attempts?"}
    ROUTE -->|pass| PASS["pass_node"] --> END([END])
    ROUTE -->|retry| RETRY["retry_node"] --> GRADE
    ROUTE -->|max attempts| STOP["end_node"] --> END
```

`4-conditional-edges/ex3_llm_grader.py`

```mermaid
flowchart TD
    START([START]) --> GRADE["grade_node"]
    GRADE --> ROUTE{"score >= 70?"}
    ROUTE -->|yes| PASS["pass_node"] --> END([END])
    ROUTE -->|no| RETRY["retry_node"] --> END
```

### 5. Workflows

`5-workflows/ex1_tone_step.py`

```mermaid
flowchart LR
    START([START]) --> DRAFT["generate_draft"] --> CHECK["fact_check"] --> IMPROVE["improve_content"] --> TONE["casual_tone"] --> END([END])
```

`5-workflows/ex2_parallel_haiku.py`

```mermaid
flowchart TD
    START([START]) --> JOKE["generate_joke"]
    START --> STORY["generate_story"]
    START --> POEM["generate_poem"]
    START --> HAIKU["generate_haiku"]
    JOKE --> AGG["aggregator"]
    STORY --> AGG
    POEM --> AGG
    HAIKU --> AGG
    AGG --> END([END])
```

`5-workflows/ex3_structured_router.py`

```mermaid
flowchart TD
    START([START]) --> CLASSIFY["classify"]
    CLASSIFY --> ROUTE{"category"}
    ROUTE -->|factual| FACT["factual"] --> END([END])
    ROUTE -->|creative| CREATIVE["creative"] --> END
    ROUTE -->|technical| TECH["technical"] --> END
```

`5-workflows/ex4_evaluator_loop_guard.py`

```mermaid
flowchart TD
    START([START]) --> GENERATE["generate"] --> EVALUATE["evaluate"]
    EVALUATE --> ROUTE{"accepted or max iterations?"}
    ROUTE -->|revise| IMPROVE["improve"] --> EVALUATE
    ROUTE -->|done| END([END])
```

`5-workflows/ex5_evaluator_headline_writer.py`

```mermaid
flowchart TD
    START([START]) --> DRAFT["draft_headline"] --> SCORE["score_headline"]
    SCORE --> ROUTE{"good enough?"}
    ROUTE -->|revise| REVISE["revise_headline"] --> SCORE
    ROUTE -->|accept| END([END])
```

`5-workflows/ex6_evaluator_strict_score.py`

```mermaid
flowchart TD
    START([START]) --> GENERATE["generate"] --> STRICT["strict_evaluate"]
    STRICT --> ROUTE{"score threshold or cap?"}
    ROUTE -->|revise| REVISE["revise"] --> STRICT
    ROUTE -->|done| END([END])
```

### 6. Agents

`6-agents/ex1_word_count_tool.py`

```mermaid
flowchart TD
    START([START]) --> LLM["llm"]
    LLM --> ROUTER{"tool calls?"}
    ROUTER -->|yes| TOOLS["tools"] --> LLM
    ROUTER -->|no| END([END])
```

`6-agents/ex2_iteration_cap.py`

```mermaid
flowchart TD
    START([START]) --> LLM["llm"]
    LLM --> ROUTER{"tool calls and under cap?"}
    ROUTER -->|yes| TOOLS["tools"] --> LLM
    ROUTER -->|no| END([END])
```

`6-agents/ex3_conversation_memory.py`

```mermaid
flowchart TD
    START([START]) --> LLM["llm"]
    LLM --> ROUTER{"tool calls?"}
    ROUTER -->|yes| TOOLS["tools"] --> LLM
    ROUTER -->|no| END([END])
```
