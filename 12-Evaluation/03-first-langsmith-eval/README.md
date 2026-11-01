# A first LangSmith evaluation

The whole evaluation loop at its smallest, before the [agentic RAG evaluation](../04-agentic-rag-evaluation/README.md) (book Section 14.7). A travel assistant answers flight questions from a five-row timetable, and one LangSmith experiment grades it on five golden cases with one scorer.

It needs `OPENAI_API_KEY` and `LANGSMITH_API_KEY` in a `.env` file; without them, [`01-minimal-evaluation`](../01-minimal-evaluation/README.md) runs the same loop locally, with no model and no upload. From the repository root:

```bash
pip install -r requirements.txt
python 12-Evaluation/03-first-langsmith-eval/first_eval.py
```

**Results we got** (`gpt-4o-mini`; the experiment link is shortened):

```text
View the evaluation results for experiment: 'travel-assistant-99f985c7' at:
https://smith.langchain.com/o/.../compare?selectedSessions=...

1.00  When does flight EK924 leave Cairo?            all keywords present
1.00  Which flights go from Cairo to Dubai?          all keywords present
1.00  Which days does MS777 fly?                     all keywords present
1.00  What time is the London flight on a Tuesday?   all keywords present
1.00  When does the Doha flight leave?               all keywords present
```

SDK progress bars may appear too, and the cases may print in a different order.

## The four pieces

| Piece | In `first_eval.py` | Role |
|---|---|---|
| Golden dataset | `GOLDEN_CASES`, uploaded once by `ensure_dataset()` as `travel-assistant-golden` | Each question with the facts a right answer must contain |
| Target | `target()` | The system under test: one golden question in, the assistant's answer out |
| Scorer | `keyword_coverage()` | Plain Python: the share of the expected keywords in the answer, and which are missing |
| Wrapper (evaluator) | `keyword_coverage_eval()` | Picks the fields from LangSmith's `outputs` and `reference_outputs`, calls the scorer, returns a key, a score and a comment |

`client.evaluate` runs every example through `target`, passes the target's `outputs` and the example's `reference_outputs` to the wrapper by argument name, and stores the run as an experiment. Find it in LangSmith under **Datasets & Experiments**, then `travel-assistant-golden`, on the **Experiments** tab: one row per case, with the answer, its trace and the `keyword_coverage` score and comment.

The first run creates the dataset; later runs reuse it, so editing `GOLDEN_CASES` after that changes nothing until you delete the dataset or edit its examples in LangSmith.

To see a failure, add a case the assistant can't satisfy word for word, for example `("When does the Doha flight leave, in 12-hour time?", ["QR1302", "2:10 AM"])`, delete the dataset, and run again: the comment names the missing keyword.
