import sys
import os

# Add this folder to sys.path so document_workflow is importable as a package.
# graph.py loads document_workflow/.env (the only env file this graph uses).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_workflow.graph import graph

if __name__ == "__main__":
    out_path = "graph.png"
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved graph diagram to {out_path}")
