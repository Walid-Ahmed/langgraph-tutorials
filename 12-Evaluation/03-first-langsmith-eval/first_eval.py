# A first LangSmith evaluation: five golden cases, one scorer, one wrapper (Chapter 14).
#
# The whole evaluation loop at its smallest, before the chatbot. A travel assistant
# answers flight questions from a five-row timetable. A golden dataset of five cases
# says which facts each answer must contain; one rule-based scorer counts them; and a
# wrapper turns that scorer into a LangSmith evaluator. evaluate() runs every case
# through the assistant and the evaluator, and stores the run as an experiment.
#
# Prerequisites: OPENAI_API_KEY and LANGSMITH_API_KEY in a .env file, plus the
# repository requirements. The first run creates the LangSmith dataset
# travel-assistant-golden; later runs reuse it.
# Run it with (from the repository root):
#    python 12-Evaluation/03-first-langsmith-eval/first_eval.py
# Expected result: a link to the experiment, then one keyword_coverage score per case;
# on our run with gpt-4o-mini, all five scored 1.00.

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client

load_dotenv(find_dotenv(usecwd=True))

DATASET = "travel-assistant-golden"
TIMETABLE = """Flight  Route            Departs  Days
EK924   Cairo -> Dubai   10:15    daily
MS910   Cairo -> Dubai   14:40    daily
MS777   Cairo -> London  09:30    Mon, Thu
BA154   Cairo -> London  15:05    daily
QR1302  Cairo -> Doha    02:10    daily"""

# The golden dataset: each question with the facts a right answer must contain.
GOLDEN_CASES = [
    ("When does flight EK924 leave Cairo?", ["EK924", "10:15"]),
    ("Which flights go from Cairo to Dubai?", ["EK924", "MS910"]),
    ("Which days does MS777 fly?", ["Mon", "Thu"]),
    ("What time is the London flight on a Tuesday?", ["BA154", "15:05"]),
    ("When does the Doha flight leave?", ["QR1302", "02:10"]),
]

assistant = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# The system under test: answer one golden question from the timetable.
def target(inputs: dict) -> dict:
    reply = assistant.invoke([
        ("system", f"You answer questions about flights from Cairo, using only this timetable:\n{TIMETABLE}"),
        ("human", inputs["question"]),
    ])
    return {"answer": reply.content}


# The scorer: the share of the expected keywords found in the answer, and which are missing.
def keyword_coverage(answer: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    missing = [k for k in expected_keywords if k.lower() not in answer.lower()]
    return 1 - len(missing) / len(expected_keywords), missing


# The wrapper: pick the fields from LangSmith's dicts, call the scorer, package the feedback.
def keyword_coverage_eval(outputs: dict, reference_outputs: dict) -> dict:
    score, missing = keyword_coverage(outputs["answer"], reference_outputs["expected_keywords"])
    return {"key": "keyword_coverage", "score": score,
            "comment": f"missing {missing}" if missing else "all keywords present"}


# Upload the five golden cases once; after that the LangSmith dataset is the one that counts.
def ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET):
        return
    dataset = client.create_dataset(DATASET, description="Five golden cases for the travel assistant")
    client.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": {"question": question}, "outputs": {"expected_keywords": keywords}}
        for question, keywords in GOLDEN_CASES
    ])


# Run every case through the assistant and the evaluator, and print each score.
def main() -> None:
    client = Client()
    ensure_dataset(client)
    results = client.evaluate(target, data=DATASET, evaluators=[keyword_coverage_eval],
                              experiment_prefix="travel-assistant")
    for row in results:
        result = row["evaluation_results"]["results"][0]
        print(f"{result.score:.2f}  {row['example'].inputs['question']:<46} {result.comment}")


if __name__ == "__main__":
    main()
