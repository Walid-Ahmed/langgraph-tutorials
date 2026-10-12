# Run from the repository root:
#   python "5-Workflows/figures/make_workflow_figures.py"
#
# What it does:
#   Draws the five workflow concept figures (workflow-07 to workflow-11) in the
#   same house style as the book's figures: white background, one teal accent,
#   DejaVu Sans / DejaVu Sans Mono, legible in black and white. They replace
#   earlier third-party images.
#
# What it demonstrates:
#   nothing about LangGraph; it's the source of the figures, so they can be
#   regenerated or edited instead of being fixed reference art.
#
# Prerequisites:
#   matplotlib (pip install matplotlib).
#
# Writes (into 5-Workflows/figures/):
#   workflow-07-augmented-llm-flow.png
#   workflow-08-augmented-llm-capabilities.png
#   workflow-09-prompt-chaining-quality-gate.png
#   workflow-10-parallelization-fan-out-fan-in.png
#   workflow-11-orchestrator-workers-flow.png

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK, STROKE, ACCENT, FILL, WHITE, MUTED = "#1E293B", "#334155", "#0D9488", "#F1F5F9", "#FFFFFF", "#64748B"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
OUT_DIR = Path(__file__).resolve().parent
DPI = 200


def canvas(width, height, title, subtitle):
    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.text(width / 2, height - 0.35, title, ha="center", fontsize=13, family=SANS, color=INK, weight="bold")
    ax.text(width / 2, height - 0.68, subtitle, ha="center", fontsize=10, family=SANS, color=INK, style="italic")
    return fig, ax


def box(ax, x, y, w, h, label, sub=None, *, fill=FILL, edge=STROKE, lw=1.5, mono=False, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    cy = y + h / 2 + (0.1 if sub else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=size, color=INK,
            family=MONO if mono else SANS, weight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center", fontsize=8.4,
                color=STROKE, family=SANS, style="italic", zorder=3)


def terminal(ax, x, y, label):
    """IN / OUT markers: a small teal pill."""
    ax.add_patch(FancyBboxPatch((x - 0.32, y - 0.2), 0.64, 0.4, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=ACCENT, edgecolor=ACCENT, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=9, color=WHITE, family=SANS, weight="bold", zorder=3)


def arrow(ax, p0, p1, *, color=STROKE, lw=1.5, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def note(ax, x, y, text, *, color=STROKE, size=8.6, mono=False, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            family=MONO if mono else SANS, style="normal" if mono else "italic", weight=weight)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {OUT_DIR / name}.png")


# ------------------------------------------------------------- workflow-07 --
def draw_07():
    """Three upgrades to one model call, each shown as input -> LLM + upgrade -> output."""
    fig, ax = canvas(12.3, 5.0, "One model call, three upgrades",
                     "Each row is still a single LLM call; the upgrade changes what goes in or what comes out.")
    rows = [
        ("Tools", '"Weather in Cairo now?"', "get_weather(city)", "the model asks; your code runs it",
         "an answer with live data"),
        ("Structured output", '"Great mouse, but the\nscroll wheel squeaks"', "ProductReview schema",
         "the reply must fit the schema", "ProductReview(rating=4, ...)"),
        ("Memory", '"What did I say my\nname was?"', "earlier messages", "sent with every call",
         '"You said it\'s Sam."'),
    ]
    for i, (name, inp, upgrade, how, out) in enumerate(rows):
        y = 3.55 - i * 1.25
        ax.text(0.3, y + 0.55, name.upper(), fontsize=8.6, family=SANS, color=ACCENT, weight="bold", va="center")
        box(ax, 0.3, y - 0.35, 2.7, 0.7, inp, fill=WHITE, mono=True, size=8.2)
        box(ax, 3.7, y - 0.38, 1.5, 0.76, "LLM", fill=FILL, size=11)
        box(ax, 5.85, y - 0.38, 3.0, 0.76, upgrade, how, fill=WHITE, edge=ACCENT, lw=2.0, mono=True, size=9)
        box(ax, 9.45, y - 0.35, 2.6, 0.7, out, fill=WHITE, mono=True, size=8.0)
        arrow(ax, (3.0, y), (3.7, y))
        arrow(ax, (5.2, y + 0.12), (5.85, y + 0.12), color=ACCENT)
        arrow(ax, (5.85, y - 0.12), (5.2, y - 0.12), color=ACCENT)
        arrow(ax, (8.85, y), (9.45, y))
    save(fig, "workflow-07-augmented-llm-flow")


# ------------------------------------------------------------- workflow-08 --
def draw_08():
    """The augmented LLM: one model in the middle, connected to retrieval, memory and tools."""
    fig, ax = canvas(10, 4.6, "The augmented LLM",
                     "The model is still in the middle; retrieval, memory and tools feed it what it can't know alone.")
    y = 3.0
    terminal(ax, 0.9, y, "IN")
    box(ax, 3.9, y - 0.42, 2.2, 0.84, "LLM", "one model call", size=12)
    terminal(ax, 9.1, y, "OUT")
    arrow(ax, (1.22, y), (3.9, y))
    arrow(ax, (6.1, y), (8.78, y))
    parts = [("Retrieval", "your documents,\nsearched by meaning", 1.0),
             ("Memory", "earlier turns,\nsaved facts", 3.95),
             ("Tools", "functions and APIs\nthe model can ask for", 6.9)]
    for name, sub, x in parts:
        box(ax, x, 0.35, 2.1, 1.05, "", fill=WHITE, edge=ACCENT, lw=1.8)
        ax.text(x + 1.05, 1.1, name, ha="center", va="center", fontsize=10.5, family=SANS, color=INK, weight="bold")
        ax.text(x + 1.05, 0.7, sub, ha="center", va="center", fontsize=8.2, family=SANS, color=STROKE, style="italic")
        arrow(ax, (x + 1.05, 1.4), (5.0, y - 0.42), color=ACCENT, style="<|-|>", rad=0.0)
    save(fig, "workflow-08-augmented-llm-capabilities")


# ------------------------------------------------------------- workflow-09 --
def draw_09():
    """Prompt chaining with a quality gate after the first call."""
    fig, ax = canvas(12, 4.0, "Prompt chaining with a quality gate",
                     "A check after the first call ends the chain early, or sends the draft on for more work.")
    y = 2.35
    terminal(ax, 0.55, y, "IN")
    box(ax, 1.3, y - 0.4, 1.9, 0.8, "LLM call 1", "first draft", size=10)
    # the gate: a diamond
    gx, gy, r = 4.35, y, 0.55
    ax.add_patch(Polygon([(gx - r, gy), (gx, gy + r), (gx + r, gy), (gx, gy - r)], closed=True,
                         facecolor=WHITE, edgecolor=ACCENT, linewidth=2.0, zorder=2))
    ax.text(gx, gy, "gate", ha="center", va="center", fontsize=10, family=SANS, color=INK, weight="bold", zorder=3)
    note(ax, gx, gy + 0.78, "a router; often plain code", size=8.2)
    box(ax, 6.2, y - 0.4, 1.9, 0.8, "LLM call 2", "improve", size=10)
    box(ax, 8.7, y - 0.4, 1.9, 0.8, "LLM call 3", "polish", size=10)
    terminal(ax, 11.35, y, "OUT")
    arrow(ax, (0.87, y), (1.3, y))
    arrow(ax, (3.2, y), (gx - r, y))
    arrow(ax, (gx + r, y), (6.2, y))
    note(ax, 5.35, y + 0.18, "fail", color=INK, size=9, weight="bold")
    arrow(ax, (8.1, y), (8.7, y))
    arrow(ax, (10.6, y), (11.03, y))
    # pass: straight to OUT
    arrow(ax, (gx, gy - r), (11.35, y - 0.2), rad=0.25, color=ACCENT)
    note(ax, 7.9, 0.82, "pass: good enough, stop here", color=ACCENT, size=9, weight="bold")
    note(ax, 6, 0.2, "Example in this folder: 01_prompt_chaining_quality_gate.py (the gate checks length and price)",
         size=8.4)
    save(fig, "workflow-09-prompt-chaining-quality-gate")


# ------------------------------------------------------------- workflow-10 --
def draw_10():
    """Parallelization: one input fans out to independent calls, then fans back in."""
    fig, ax = canvas(11, 5.0, "Parallelization: fan out, fan in",
                     "Independent calls run in the same step; the aggregator waits for all of them.")
    terminal(ax, 0.6, 2.35, "IN")
    xs, w = 3.0, 2.3
    fields = ["summary", "quiz", "analogy"]
    for i, field in enumerate(fields):
        y = 3.35 - i * 1.0
        box(ax, xs, y - 0.33, w, 0.66, f"LLM call {i + 1}", f"writes {field}", size=10)
        arrow(ax, (0.92, 2.35), (xs, y))
        arrow(ax, (xs + w, y), (7.0, 2.35))
    box(ax, 7.0, 1.95, 2.0, 0.8, "aggregator", "runs once, after all", size=10, edge=ACCENT, lw=2.0)
    terminal(ax, 10.4, 2.35, "OUT")
    arrow(ax, (9.0, 2.35), (10.08, 2.35))
    note(ax, 1.9, 3.75, "fan-out", color=ACCENT, size=9, weight="bold")
    note(ax, 6.2, 3.75, "fan-in", color=ACCENT, size=9, weight="bold")
    note(ax, 5.5, 0.55, "One writer per field, so nothing to merge. Two writers on one field need a reducer.", size=8.6)
    save(fig, "workflow-10-parallelization-fan-out-fan-in")


# ------------------------------------------------------------- workflow-11 --
def draw_11():
    """Orchestrator-workers: a planner decides N at run time; Send starts one worker per item."""
    fig, ax = canvas(12, 5.2, "Orchestrator–workers",
                     "The orchestrator decides how many workers at run time; Send starts one per planned item.")
    terminal(ax, 0.55, 2.45, "IN")
    box(ax, 1.3, 2.0, 2.2, 0.9, "orchestrator", "plans N subtasks", size=10.5, edge=ACCENT, lw=2.0)
    arrow(ax, (0.87, 2.45), (1.3, 2.45))
    workers = [("worker", "week 1", 3.7), ("worker", "week 2", 2.75), ("worker", "week 3", 1.8)]
    for label, sub, y in workers:
        box(ax, 5.0, y - 0.33, 2.0, 0.66, label, sub, size=10)
        arrow(ax, (3.5, 2.45), (5.0, y), color=ACCENT, ls="--")
        arrow(ax, (7.0, y), (8.5, 2.45))
    ax.text(6.0, 1.05, "…  up to N", ha="center", fontsize=10, family=SANS, color=STROKE)
    note(ax, 4.15, 3.62, "Send × N", color=ACCENT, size=9.5, mono=True, weight="bold")
    box(ax, 8.5, 2.0, 2.0, 0.9, "synthesizer", "joins the results", size=10.5)
    terminal(ax, 11.4, 2.45, "OUT")
    arrow(ax, (10.5, 2.45), (11.08, 2.45))
    note(ax, 6, 0.45, "Workers write to one shared list, so that field needs a reducer (operator.add). "
         "Example: 04_orchestrator_workers_study_plan.py", size=8.4)
    save(fig, "workflow-11-orchestrator-workers-flow")


if __name__ == "__main__":
    for draw in (draw_07, draw_08, draw_09, draw_10, draw_11):
        draw()
