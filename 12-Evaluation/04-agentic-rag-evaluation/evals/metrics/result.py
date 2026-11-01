# The one return type every metric shares: a name, a 0-1 score and a short note.
# A LangSmith evaluator turns it into feedback: key=name, score=score, comment=note.

from dataclasses import dataclass


# What every metric returns: a name, a score from 0 to 1, and a note saying why.
@dataclass
class MetricResult:
    name: str
    score: float
    note: str
