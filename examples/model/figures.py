"""Draw the five figures in the model overview.
Usage from the repository root: PYTHONPATH=src:. python examples/model/figures.py
Each figure is written as a PNG preview and a vector PDF at final print size.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nature_style as ns  # noqa: E402
from grain_landscape import grain_geometry, energy_landscape, read_grain  # noqa: E402
from population_response import population, forward_grid  # noqa: E402
from inverse_grid import joint_fit  # noqa: E402


def main():
    directory = Path(__file__).resolve().parent
    ns.use()
    model, states = read_grain(directory)
    figures = (
        ("grain", grain_geometry(model)),
        ("landscape", energy_landscape(model, states)),
        ("population", population(directory)),
        ("forward_grid", forward_grid(directory)),
        ("joint_fit", joint_fit()),
    )
    for name, fig in figures:
        ns.save(fig, directory / f"{name}.png")
        plt.close(fig)
        print(f"{name}.png  {name}.pdf")


if __name__ == "__main__":
    main()
