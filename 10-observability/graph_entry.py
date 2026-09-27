import sys
import os

# Add this folder to sys.path so document_workflow is importable as a package.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_workflow.graph import build_graph

graph = build_graph()
