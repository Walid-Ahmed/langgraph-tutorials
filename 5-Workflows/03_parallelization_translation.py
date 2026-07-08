# Parallelization variant: START fans out into three independent translation
# nodes (Arabic, French, Italian) for the same English paragraph, then an
# aggregator node joins all three translations once they finish.

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


# ---------------------------------------------------------
# 1. State Definition
#
# One English paragraph enters the graph.
# Three independent branches translate it into different languages.
#
# No reducer is needed because each branch writes to a different key:
# - arabic_translation
# - french_translation
# - italian_translation
# ---------------------------------------------------------
class TranslationState(TypedDict):
    english_paragraph: str
    arabic_translation: str
    french_translation: str
    italian_translation: str
    combined_output: str


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 2. Parallel Translation Nodes
#
# These nodes are independent: Arabic does not need French,
# French does not need Italian, and so on.
# ---------------------------------------------------------
def translate_to_arabic(state: TranslationState) -> dict:
    """Translate the English paragraph into Arabic."""
    msg = llm.invoke(
        "Translate the following English paragraph into clear Modern Standard Arabic. "
        "Return only the Arabic translation.\n\n"
        f"{state['english_paragraph']}"
    )
    return {"arabic_translation": msg.content}


def translate_to_french(state: TranslationState) -> dict:
    """Translate the English paragraph into French."""
    msg = llm.invoke(
        "Translate the following English paragraph into natural French. "
        "Return only the French translation.\n\n"
        f"{state['english_paragraph']}"
    )
    return {"french_translation": msg.content}


def translate_to_italian(state: TranslationState) -> dict:
    """Translate the English paragraph into Italian."""
    msg = llm.invoke(
        "Translate the following English paragraph into natural Italian. "
        "Return only the Italian translation.\n\n"
        f"{state['english_paragraph']}"
    )
    return {"italian_translation": msg.content}


# ---------------------------------------------------------
# 3. Aggregator Node
#
# The aggregator waits for the three translations, then combines
# them into one readable translation package.
# ---------------------------------------------------------
def aggregate_translations(state: TranslationState) -> dict:
    """Combine all translations into one formatted output."""
    combined_output = f"""
ORIGINAL ENGLISH
{state['english_paragraph']}

ARABIC
{state['arabic_translation']}

FRENCH
{state['french_translation']}

ITALIAN
{state['italian_translation']}
"""
    return {"combined_output": combined_output}


# ---------------------------------------------------------
# 4. Build The Graph
#
# START fans out to the three translation nodes.
# The translation nodes fan in to aggregate_translations.
# ---------------------------------------------------------
builder = StateGraph(TranslationState)

builder.add_node("translate_to_arabic", translate_to_arabic)
builder.add_node("translate_to_french", translate_to_french)
builder.add_node("translate_to_italian", translate_to_italian)
builder.add_node("aggregate_translations", aggregate_translations)

builder.add_edge(START, "translate_to_arabic")
builder.add_edge(START, "translate_to_french")
builder.add_edge(START, "translate_to_italian")

builder.add_edge("translate_to_arabic", "aggregate_translations")
builder.add_edge("translate_to_french", "aggregate_translations")
builder.add_edge("translate_to_italian", "aggregate_translations")

builder.add_edge("aggregate_translations", END)

translation_workflow = builder.compile()


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "03_parallelization_translation_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(translation_workflow, graph_image_path)

    english_paragraph = (
        "Artificial intelligence is changing how teams work. "
        "It can summarize long documents, draft ideas, translate content, "
        "and help people focus on higher-value decisions."
    )

    result = translation_workflow.invoke(
        {
            "english_paragraph": english_paragraph,
            "arabic_translation": "",
            "french_translation": "",
            "italian_translation": "",
            "combined_output": "",
        }
    )

    print(result["combined_output"])


if __name__ == "__main__":
    main()
