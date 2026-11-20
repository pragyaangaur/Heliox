"""Draw the chart of Rust speedups over the original Python.

The numbers were measured on an Apple M4. Each Rust result was checked
against the output of the original Python code. Run the script from anywhere:

    python profiling/plot_speedups.py

It writes figures/rust_speedups.png at the root of the repository.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e2e1dd"

# Each entry is (function, package and input, speedup). The rows are sorted
# from largest to smallest speedup below.
RESULTS = [
    ("calculate_gamma_values", "sunkit-image, real 512 x 1024 field", 389),
    ("MEM reconstruction", "xrayvision, 129 x 129, real STIX data", 14.7),
    ("differential_rotate", "sunpy, 1024 x 1024 full-disk AIA map", 7.9),
    ("IAU 2000A nutation", "Heliox, 100,000 dates", 6.1),
    ("time string parsing", "Heliox, 10,000 strings", 2.1),
]


def speedup_label(value):
    """Format a speedup such as 389x or 14.7x."""
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def draw(path):
    """Draw the chart and save it to path."""
    rows = sorted(RESULTS, key=lambda row: row[2], reverse=True)
    count = len(rows)
    # The first row is drawn at the top, so y decreases going down the list.
    positions = [count - 1 - index for index in range(count)]

    fig = plt.figure(figsize=(9, 4.5), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0.37, 0.2, 0.58, 0.62], facecolor=SURFACE)

    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(-0.6, count - 0.4)

    ax.barh(
        positions,
        [value - 1 for _, _, value in rows],
        left=1,
        height=0.4,
        color=BLUE,
        zorder=3,
    )

    # Label each bar at its end with its speedup.
    for y, (_, _, value) in zip(positions, rows):
        ax.text(
            value * 1.1,
            y,
            speedup_label(value),
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT,
        )

    # Row labels: the function on the first line, the package and input below.
    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, (function, source, _) in zip(positions, rows):
        ax.text(
            -0.03,
            y + 0.13,
            function,
            transform=label_transform,
            va="center",
            ha="right",
            fontsize=10,
            color=TEXT,
        )
        ax.text(
            -0.03,
            y - 0.17,
            source,
            transform=label_transform,
            va="center",
            ha="right",
            fontsize=8.5,
            color=MUTED,
        )

    # Only the x axis is gridded, and the bars start at 1x.
    ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}x"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.text(
        0.03,
        0.92,
        "Rust speedup over the original Python",
        ha="left",
        va="center",
        fontsize=13,
        color=TEXT,
    )
    fig.text(
        0.03,
        0.04,
        "Measured on an Apple M4. Each Rust result was checked against the original output.",
        ha="left",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )

    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    draw(ROOT / "figures" / "rust_speedups.png")
