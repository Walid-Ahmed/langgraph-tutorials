import sys
import os

# Add the project root to sys.path so 'agent' is importable as a package.
# This makes relative imports inside agent/ resolve correctly without pip install -e .
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import build_graph

graph = build_graph()
