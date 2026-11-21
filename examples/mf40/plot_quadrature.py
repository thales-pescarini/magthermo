"""Plot the 40 °C MF quadrature comparison from its results table.
Usage: python examples/mf40/plot_quadrature.py
"""
import argparse
import csv
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def draw_figure(rows, protocol, path):
    fig = Figure(figsize=(8.8, 4.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.975, bottom=.23, top=.78, wspace=.09)
    ink, band, accent = "#16556c", "#b6d9e2", "#a04416"
    for ax, model in zip(axes, protocol["curve_models"]):
        data = [r for r in rows if r["model"] == model]
        x = np.array([r["true_log10_duration_s"] for r in data])
        y = np.array([r["full_ratio"] for r in data], dtype=float)
        block_ratios = np.array([[r[f"block{i}_ratio"] for r in data] for i in range(1, 5)], dtype=float)
        unresolved = np.array([r["classification"].startswith("inconclusive") for r in data])
        ax.fill_between(x, np.min(block_ratios, axis=0), np.max(block_ratios, axis=0), color=band)
        ax.axhline(1, color="#555555", linewidth=1, linestyle="--")
        ax.plot(x, y, color=ink, linewidth=1.7)
        ax.scatter(x[unresolved], y[unresolved], facecolors="white", edgecolors=accent, s=16, zorder=3)
        ax.set_title(f"MF / {model}", loc="left", fontsize=11)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
        ax.set_xlabel(r"True duration, $\log_{10}(t/\mathrm{s})$")
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlim(-.25, 15.25)
    axes[0].set_ylabel(r"$D_{\rm quad}\,/\,U_{\rm ref}$")
    fig.suptitle("Crystal orientation and duration recovery at 40 °C", y=.965, fontsize=15)
    fig.text(.5, .876, "Known event temperature   ·   Partial MF support   ·   C16 operational reference",
             ha="center", fontsize=9.5)
    handles = [Line2D([], [], color=ink, label="Full sample"),
               Patch(color=band, label="Four-block range"),
               Line2D([], [], color="#555555", linestyle="--", label="Detection threshold"),
               Line2D([], [], marker="o", color=accent, markerfacecolor="white",
                      linestyle="none", label="Inconclusive")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .015),
               frameon=False, fontsize=9.2, handlelength=1.5)
    caption = ("Duration recovery at a known event temperature of 40 °C, using partial MF support "
               "and C16 as the operational reference. Shading spans four blocks of 512 observations. "
               "Open circles mark inconclusive cases under the full-sample and four-block rule.")
    fig.savefig(path, dpi=180, metadata={"Software": None, "Description": caption})


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
