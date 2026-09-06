import sys
import os

from dotenv import load_dotenv

# Add the project root to sys.path so 'agent' is importable as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from agent.graph import graph

if __name__ == "__main__":
    out_path = "graph.png"
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved graph diagram to {out_path}")
