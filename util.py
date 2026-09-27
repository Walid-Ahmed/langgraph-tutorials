# Run from the repository root:
#   python "util.py"
#
# Shared helper imported by tutorial scripts to print a compiled graph's
# Mermaid diagram to the console and save it as a PNG for the README/docs.

def plot_graph(app, output_path="graph.png", *, print_mermaid=True):
    if print_mermaid:
        print("\n--- Mermaid Graph ---")
        print(app.get_graph().draw_mermaid())

    png_bytes = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"Graph saved to {output_path}")
