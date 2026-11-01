# Retrieval metrics in a few lines: precision, recall and reciprocal rank (Chapter 14, Section 14.6).
#
# Three searches. The first is Kerolos Yacoub's worked example: four documents
# retrieved, three relevant, one of them never found. The other two answer the
# travel assistant's "Can I take a 10 kg bag into the cabin?", which needs one
# document: a good search that ranks it first, and a worse one that ranks it
# third. Precision and recall over everything retrieved are what RAGAS and DeepEval
# call context precision and context recall; the reciprocal rank also rewards
# putting the right document first.
#
# Prerequisites: none (no model, no API key, standard library only).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/rag.py
# Expected result: precision@1, precision@4, recall@4 and reciprocal rank per search.

SEARCHES = {  # name: (retrieved in rank order, relevant)
    "worked example": (["doc1", "doc2", "doc3", "doc4"], ["doc2", "doc4", "doc5"]),
    "good search": (["baggage_policy.pdf", "fares.csv", "fares.csv", "routes.csv"], ["baggage_policy.pdf"]),
    "worse search": (["fares.csv", "routes.csv", "baggage_policy.pdf", "fares.csv"], ["baggage_policy.pdf"]),
}


# Precision@k: the share of the top k results that are relevant.
def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return sum(doc in relevant for doc in retrieved[:k]) / k


# Recall@k: the share of the relevant documents that appear in the top k.
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return sum(doc in retrieved[:k] for doc in relevant) / len(relevant)


# Reciprocal rank: 1 / the position of the first relevant result (MRR averages it over questions).
def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


if __name__ == "__main__":
    for name, (retrieved, relevant) in SEARCHES.items():
        print(f"{name:<15} P@1 {precision_at_k(retrieved, relevant, 1):.2f}  "
              f"P@4 {precision_at_k(retrieved, relevant, 4):.2f}  "
              f"R@4 {recall_at_k(retrieved, relevant, 4):.3f}  "
              f"RR {reciprocal_rank(retrieved, relevant):.2f}")
