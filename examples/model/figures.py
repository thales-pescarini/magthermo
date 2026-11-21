"""Draw the five figures in the model overview.
Usage from the repository root: PYTHONPATH=src:. python examples/model/figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grain_landscape import grain_geometry, energy_landscape, read_grain
from population_response import population, forward_grid
from inverse_grid import joint_fit


def main():
    directory = Path(__file__).resolve().parent
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#263849", "text.color": "#182a3c",
        "axes.edgecolor": "#637381", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.titleweight": "semibold", "axes.titlesize": 12,
        "axes.titlepad": 11, "legend.frameon": False, "savefig.facecolor": "white",
    })
    model, states = read_grain(directory)
    figures = (
        ("grain", grain_geometry(model)),
        ("landscape", energy_landscape(model, states)),
        ("population", population(directory)),
        ("forward_grid", forward_grid(directory)),
        ("joint_fit", joint_fit()),
    )
    for name, fig in figures:
        fig.savefig(directory / f"{name}.png", dpi=165, metadata={"Software": None})
        plt.close(fig)
        print(f"{name}.png")


if __name__ == "__main__":
    main()
