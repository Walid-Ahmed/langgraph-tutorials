import html
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
# Prompt chaining works by passing outputs forward.
# Each field below is created by one step and used by the next.
# ---------------------------------------------------------
class ContentState(TypedDict):
    topic: str
    requirements: str
    draft: str
    fact_check_results: str
    improved_content: str
    final_draft: str


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 2. Nodes
#
# Each node performs one focused LLM call.
# The output of one node becomes input to the next node.
# ---------------------------------------------------------
def generate_draft(state: ContentState) -> dict:
    """Generate the first blog post draft."""

    prompt = f"""
    Write a 200-word blog post about: {state['topic']}

    Requirements: {state['requirements']}

    Focus on creating engaging, informative content.
    """

    draft = llm.invoke(prompt).content

    print("=== STEP 1: Draft Generated ===")
    print(draft[:150] + "...\n")

    return {"draft": draft}


def fact_check(state: ContentState) -> dict:
    """Review the draft for accuracy, consistency, and missing citations."""

    prompt = f"""
    Review the following blog post draft for factual accuracy and consistency:

    {state['draft']}

    Identify:
    1. Any factual claims that seem questionable
    2. Internal inconsistencies
    3. Statements that need citations

    Provide a brief report.
    """

    fact_check_results = llm.invoke(prompt).content

    print("=== STEP 2: Fact Check Complete ===")
    print(fact_check_results[:150] + "...\n")

    return {"fact_check_results": fact_check_results}


def improve_content(state: ContentState) -> dict:
    """Revise the draft using the fact-check feedback."""

    prompt = f"""
    Here is a blog post draft:

    {state['draft']}

    Here is feedback from fact-checking:

    {state['fact_check_results']}

    Revise the blog post to address the feedback while maintaining engaging writing.
    Keep it around 200 words.
    """

    improved_content = llm.invoke(prompt).content

    print("=== STEP 3: Content Improved ===")
    print(improved_content[:150] + "...\n")

    return {"improved_content": improved_content}


def format_output(state: ContentState) -> dict:
    """Format the improved content for web publication."""

    prompt = f"""
    Format the following blog post for web publication:

    {state['improved_content']}

    Add:
    - An engaging title wrapped in <h1> tags
    - Subheadings where appropriate with <h2> tags
    - Paragraph tags <p>
    - A meta description in 1-2 sentences

    Output only the formatted HTML. Do not wrap it in Markdown code fences.
    """

    final_draft = llm.invoke(prompt).content.strip()
    final_draft = final_draft.removeprefix("```html").removeprefix("```").removesuffix("```").strip()

    print("=== STEP 4: Formatted for Publication ===")
    print(final_draft[:200] + "...\n")

    return {"final_draft": final_draft}


# ---------------------------------------------------------
# 3. Build The Graph
#
# Flow:
# START -> generate_draft -> fact_check -> improve_content -> format_output -> END
# ---------------------------------------------------------
graph_builder = StateGraph(ContentState)

graph_builder.add_node("generate_draft", generate_draft)
graph_builder.add_node("fact_check", fact_check)
graph_builder.add_node("improve_content", improve_content)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "generate_draft")
graph_builder.add_edge("generate_draft", "fact_check")
graph_builder.add_edge("fact_check", "improve_content")
graph_builder.add_edge("improve_content", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()


# ---------------------------------------------------------
# 4. HTML Report Helper
#
# This saves every pipeline stage so you can inspect the chain.
# ---------------------------------------------------------
def save_html_report(result: ContentState) -> Path:
    output_path = Path(__file__).with_name("prompt_chaining_output.html")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Prompt Chaining Content Pipeline</title>
    <style>
        body {{
            font-family: Georgia, serif;
            max-width: 850px;
            margin: 60px auto;
            padding: 0 20px;
            line-height: 1.7;
            color: #333;
            background: #fafafa;
        }}
        .section {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px 28px;
            margin-bottom: 32px;
            background: white;
        }}
        .section h2 {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin: 0 0 14px;
        }}
        .section pre {{
            white-space: pre-wrap;
            font-family: Georgia, serif;
            margin: 0;
            font-size: 15px;
        }}
    </style>
</head>
<body>
    <div class="section">
        <h2>Topic</h2>
        <pre>{html.escape(result['topic'])}</pre>
    </div>
    <div class="section">
        <h2>Requirements</h2>
        <pre>{html.escape(result['requirements'])}</pre>
    </div>
    <div class="section">
        <h2>Draft</h2>
        <pre>{html.escape(result['draft'])}</pre>
    </div>
    <div class="section">
        <h2>Fact Check Results</h2>
        <pre>{html.escape(result['fact_check_results'])}</pre>
    </div>
    <div class="section">
        <h2>Improved Content</h2>
        <pre>{html.escape(result['improved_content'])}</pre>
    </div>
    <div class="section">
        <h2>Final Draft</h2>
        {result['final_draft']}
    </div>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    plot_graph(graph)

    result = graph.invoke(
        {
            "topic": "The benefits of morning exercise",
            "requirements": "Target audience: AI engineers",
        }
    )

    report_path = save_html_report(result)

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(result["final_draft"])
    print(f"\nHTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
