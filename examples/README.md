# Model figures and numerical tests

| Directory | What it contains | Command from the repository root |
| --- | --- | --- |
| [Model figures](model/README.md) | Single-grain energies and transition barriers, population weights, forward curves and an analytic test of inverse ambiguity. The directory documents the inputs for each plot. | `PYTHONPATH=src:. python examples/model/figures.py` |
| [Crystal-orientation sensitivity](mf40/README.md) | A regression test comparing two crystal-orientation quadratures on the same noisy observations. Temperature is fixed so that the effect on duration can be measured separately. | `python examples/mf40/reproduce.py` |

Both commands use the [pinned dependencies](../requirements.txt) and the inputs in their respective directories.
