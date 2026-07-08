# Shared helper imported by tutorial scripts to print a compiled graph's
# Mermaid diagram to the console and save it as a PNG for the README/docs.

def plot_graph(app, output_path="graph.png"):
    # Visualize the graph
    print("\n--- Mermaid Graph ---")
    print(app.get_graph().draw_mermaid())

    # Save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"\nGraph saved to {output_path}")
