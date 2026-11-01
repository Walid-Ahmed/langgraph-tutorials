# Retrieval metrics in a few lines: precision, recall and reciprocal rank (Chapter 14, Section 14.5).
#
# Three searches. The first is a small worked example: four documents retrieved,
# three relevant, one of them never found. The other two are searches for "Which
# sales rep covers Europe?", which needs one file: the chatbot's real search, and
# a worse one that ranks the right file third. Precision@k and recall@k over
# everything retrieved are what RAGAS and DeepEval call context precision and
# context recall; the reciprocal rank also rewards putting the right file first.
#
# Prerequisites: none (no model, no API key, standard library only).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/rag.py
# Expected result: precision@1, precision@4, recall@4 and reciprocal rank per search.

SEARCHES = {  # name: (retrieved in rank order, relevant)
    "worked example": (["doc1", "doc2", "doc3", "doc4"], ["doc2", "doc4", "doc5"]),
    "chatbot's search": (["crm_exports/sales_reps.csv", "crm_exports/accounts.csv",
                          "crm_exports/accounts.csv", "crm_exports/opportunities.csv"],
                         ["crm_exports/sales_reps.csv"]),
    "a worse search": (["crm_exports/accounts.csv", "crm_exports/opportunities.csv",
                        "crm_exports/sales_reps.csv", "crm_exports/accounts.csv"],
                       ["crm_exports/sales_reps.csv"]),
}


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of the top k results that are relevant."""
    return sum(doc in relevant for doc in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    return sum(doc in retrieved[:k] for doc in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / the position of the first relevant result (MRR averages this over questions)."""
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


if __name__ == "__main__":
    for name, (retrieved, relevant) in SEARCHES.items():
        print(f"{name:<17} P@1 {precision_at_k(retrieved, relevant, 1):.2f}  "
              f"P@4 {precision_at_k(retrieved, relevant, 4):.2f}  "
              f"R@4 {recall_at_k(retrieved, relevant, 4):.3f}  "
              f"RR {reciprocal_rank(retrieved, relevant):.2f}")
