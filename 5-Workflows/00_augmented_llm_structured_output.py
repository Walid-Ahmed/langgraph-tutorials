# Augmented LLM workflow: a Pydantic schema (ProductReview) is bound to the
# model via with_structured_output(), so a single node turns free-text
# review input into a validated, structured object instead of raw text.

import json
import sys
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


# ---------------------------------------------------------
# 1. Structured Output Schema
#
# This Pydantic model defines the exact shape we want back
# from the LLM.
# ---------------------------------------------------------
class ProductReview(BaseModel):
    """Structured product review analysis."""

    product_name: str = Field(description="Name of the product")
    sentiment: str = Field(
        description="Overall sentiment: positive, negative, or neutral"
    )
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")


# ---------------------------------------------------------
# 2. Graph State
#
# review_text is the input.
# analysis will hold the validated ProductReview object.
# ---------------------------------------------------------
class ReviewState(TypedDict):
    review_text: str
    analysis: ProductReview | None


# ---------------------------------------------------------
# 3. LLM With Structured Output
#
# with_structured_output(ProductReview) tells the model to
# return data that matches the Pydantic schema.
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(ProductReview)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a product review analyzer. Extract structured information from reviews.",
        ),
        ("user", "{review_text}"),
    ]
)

chain = prompt | structured_llm


# ---------------------------------------------------------
# 4. Node
#
# The node sends review_text to the chain and stores the
# validated ProductReview object in state.
# ---------------------------------------------------------
def analyze_review(state: ReviewState) -> dict:
    result = chain.invoke({"review_text": state["review_text"]})
    return {"analysis": result}


# ---------------------------------------------------------
# 5. Build Graph
#
# Flow:
# START -> analyze_review -> END
# ---------------------------------------------------------
graph_builder = StateGraph(ReviewState)

graph_builder.add_node("analyze_review", analyze_review)
graph_builder.add_edge(START, "analyze_review")
graph_builder.add_edge("analyze_review", END)

graph = graph_builder.compile()


# ---------------------------------------------------------
# 6. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "00_augmented_llm_structured_output_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

    review_text = """
    I bought this wireless mouse last month and it's been mostly great.
    The battery life is incredible - I've only charged it once in 4 weeks.
    The ergonomic design fits my hand perfectly and the buttons are responsive.
    However, the scroll wheel is a bit stiff and makes clicking sounds.
    Also, it's quite expensive compared to similar models.
    Overall, I'd give it 4 out of 5 stars.
    """

    final_state = graph.invoke(
        {
            "review_text": review_text,
            "analysis": None,
        }
    )

    analysis = final_state["analysis"]
    print(json.dumps(analysis.model_dump(), indent=2))


if __name__ == "__main__":
    main()
