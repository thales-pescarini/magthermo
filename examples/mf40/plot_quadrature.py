"""Plot the 40 °C MF quadrature comparison from its results table.
Usage: python examples/mf40/plot_quadrature.py
Writes the requested image and a vector PDF twin at final print size.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nature_style as ns  # noqa: E402


def draw_figure(rows, protocol, path):
    with ns.style():
        fig = Figure(figsize=(ns.ONE_HALF_WIDE, 60 * ns.MM), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        ink, band, accent, threshold = ns.BLUE, ns.tint(ns.BLUE, .28), ns.VERMILION, ns.BLACK
        for ax, model, letter in zip(axes, protocol["curve_models"], "ab"):
            data = [r for r in rows if r["model"] == model]
            x = np.array([r["true_log10_duration_s"] for r in data])
            y = np.array([r["full_ratio"] for r in data], dtype=float)
            block_ratios = np.array([[r[f"block{i}_ratio"] for r in data] for i in range(1, 5)], dtype=float)
            unresolved = np.array([r["classification"].startswith("inconclusive") for r in data])
            ax.fill_between(x, np.min(block_ratios, axis=0), np.max(block_ratios, axis=0), color=band, lw=0)
            ax.axhline(1, color=threshold, linewidth=.5, linestyle=(0, (4, 2)))
            ax.plot(x, y, color=ink, linewidth=1.)
            ax.scatter(x[unresolved], y[unresolved], facecolors="white", edgecolors=accent, s=10,
                       linewidths=.6, zorder=3)
            ax.set_title(f"MF / {model}")
            ax.set_xticks([0, 3, 6, 9, 12, 15])
            ax.set_xlabel(r"True duration, $\log_{10}(t/\mathrm{s})$")
            ax.set_xlim(-.25, 15.25)
            ns.panel_label(ax, letter)
        axes[0].set_ylabel(r"$D_\mathrm{quad}\,/\,U_\mathrm{ref}$")
        handles = [Line2D([], [], color=ink, lw=1., label="Full sample"),
                   Patch(color=band, lw=0, label="Four-block range"),
                   Line2D([], [], color=threshold, lw=.5, linestyle=(0, (4, 2)), label="Detection threshold"),
                   Line2D([], [], marker="o", color=accent, markerfacecolor="white", markersize=3.4,
                          markeredgewidth=.6, linestyle="none", label="Inconclusive")]
        fig.legend(handles=handles, ncol=4, loc="outside lower center")
        caption = ("Duration recovery at a known event temperature of 40 °C, using partial MF support "
                   "and C16 as the operational reference. Shading spans four blocks of 512 observations. "
                   "Open circles mark inconclusive cases under the full-sample and four-block rule.")
        ns.save(fig, path, caption=caption)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=here / "results.csv")
    parser.add_argument("--output", type=Path, default=here / "quadrature.png")
    args = parser.parse_args()
    protocol = json.loads((here / "protocol.json").read_text())
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = ["true_log10_duration_s", "full_ratio", *(f"block{i}_ratio" for i in range(1, 5))]
    for row in rows:
        for column in columns:
            row[column] = float(row[column]) if row[column] else np.nan
    draw_figure(rows, protocol, args.output)


if __name__ == "__main__":
    main()
