# Parallelization (fan-out/fan-in): START branches into three independent
# LLM nodes that each write a different social-media post for the same
# topic, then all three branches converge on one aggregator node.

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
# Parallelization works best when several nodes can read the
# same input and write independent outputs.
#
# No reducer is needed in this example because each parallel
# node writes to a different state field:
# - generate_instagram writes instagram_post
# - generate_twitter writes twitter_post
# - generate_linkedin writes linkedin_post
#
# If multiple parallel nodes wrote to the same field, then a
# reducer would be needed to merge those updates safely.
# ---------------------------------------------------------
class OverallState(TypedDict):
    topic: str
    instagram_post: str
    twitter_post: str
    linkedin_post: str
    final_output: str


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 2. Parallel Worker Nodes
#
# These three nodes do not depend on each other.
# They all read the same topic, then create content for a
# different social media platform.
# ---------------------------------------------------------
def generate_instagram(state: OverallState) -> dict:
    """Generate an engaging Instagram post with emojis and hashtags."""
    print("Instagram Generator: creating post...")

    prompt = f"""
    Create an Instagram post about: {state['topic']}

    Requirements:
    - Engaging and visual language
    - 2-3 short paragraphs, 150-200 words maximum
    - Include relevant emojis
    - End with 5-8 relevant hashtags
    - Casual, friendly tone
    - Call-to-action to engage with the post

    Make it perfect for Instagram's audience.
    """

    instagram_post = llm.invoke(prompt).content

    print("Instagram Generator: complete\n")

    return {"instagram_post": instagram_post}


def generate_twitter(state: OverallState) -> dict:
    """Generate a concise Twitter/X post."""
    print("Twitter Generator: creating post...")

    prompt = f"""
    Create a Twitter/X post about: {state['topic']}

    Requirements:
    - Maximum 280 characters
    - Punchy and attention-grabbing
    - Include 2-3 relevant hashtags
    - Conversational tone
    - Can use emojis sparingly
    - Should spark engagement or replies

    Make it perfect for a fast-paced social feed.
    """

    twitter_post = llm.invoke(prompt).content

    print("Twitter Generator: complete\n")

    return {"twitter_post": twitter_post}


def generate_linkedin(state: OverallState) -> dict:
    """Generate a professional LinkedIn post."""
    print("LinkedIn Generator: creating post...")

    prompt = f"""
    Create a LinkedIn post about: {state['topic']}

    Requirements:
    - Professional yet engaging tone
    - 3-4 paragraphs, 200-300 words
    - Include insights or lessons learned
    - Use line breaks for readability
    - Add 3-5 professional hashtags
    - Include a thought-provoking question at the end
    - Focus on value and professional development

    Make it perfect for LinkedIn's professional audience.
    """

    linkedin_post = llm.invoke(prompt).content

    print("LinkedIn Generator: complete\n")

    return {"linkedin_post": linkedin_post}


# ---------------------------------------------------------
# 3. Aggregator Node
#
# This node runs after all three platform-specific nodes finish.
# It reads their outputs and combines them into one final package.
# ---------------------------------------------------------
def aggregate_posts(state: OverallState) -> dict:
    """Combine all platform posts into a formatted final output."""
    print("Aggregator: combining all posts...\n")

    final_output = f"""
{'=' * 70}
SOCIAL MEDIA CONTENT PACKAGE
{'=' * 70}
Topic: {state['topic']}

{'=' * 70}
INSTAGRAM POST
{'=' * 70}

{state['instagram_post']}

{'=' * 70}
TWITTER/X POST
{'=' * 70}

{state['twitter_post']}

{'=' * 70}
LINKEDIN POST
{'=' * 70}

{state['linkedin_post']}
"""

    return {"final_output": final_output}


# ---------------------------------------------------------
# 4. Build The Graph
#
# START fans out into three independent branches.
# All three branches fan in to aggregate_posts.
# ---------------------------------------------------------
builder = StateGraph(OverallState)

builder.add_node("generate_instagram", generate_instagram)
builder.add_node("generate_twitter", generate_twitter)
builder.add_node("generate_linkedin", generate_linkedin)
builder.add_node("aggregate_posts", aggregate_posts)

builder.add_edge(START, "generate_instagram")
builder.add_edge(START, "generate_twitter")
builder.add_edge(START, "generate_linkedin")

builder.add_edge("generate_instagram", "aggregate_posts")
builder.add_edge("generate_twitter", "aggregate_posts")
builder.add_edge("generate_linkedin", "aggregate_posts")

builder.add_edge("aggregate_posts", END)

graph = builder.compile()


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "03_parallelization_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

    topic = "The impact of AI on workplace productivity"
    print(f"\nTopic: {topic}\n")

    result = graph.invoke(
        {
            "topic": topic,
            "instagram_post": "",
            "twitter_post": "",
            "linkedin_post": "",
            "final_output": "",
        }
    )

    print(result["final_output"])


if __name__ == "__main__":
    main()
