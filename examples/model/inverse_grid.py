"""Illustrate amplitude profiling with the two-component analytic estimator test."""
import matplotlib.pyplot as plt
import numpy as np

from portao_a.inference import GridEstimator
from portao_a.inference_modes import FIXED_SCALE, L1


def joint_fit():
    temperatures = np.arange(40., 301., 20.)
    ell = np.arange(0., 15.125, .125)
    curves = np.zeros((len(temperatures), len(ell), 60))
    curves[:, :, 0] = temperatures[:, None] / 300.
    curves[:, :, 1] = ell[None, :] / 15.
    estimator = GridEstimator(temperatures, ell, curves, sigma=.04)
    truth = curves[temperatures == 160., ell == 6.][0]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.6, 4.4), layout="constrained")
    for ax, mode, title in zip(axes, (L1, FIXED_SCALE), ("One fitted amplitude (L1)", "Fixed amplitude")):
        fit = estimator.fit(truth, amplitude_mode=mode, return_chi2=True)
        surface = (fit["chi2"][0] - fit["minimum_chi2"][0]).reshape(curves.shape[:2])
        # All proportional template vectors fit exactly when one gain is free.
        ti, di = np.unravel_index(fit["minimizers"][0], surface.shape)
        if mode == L1:
            np.testing.assert_array_equal(temperatures[ti], temperatures)
            np.testing.assert_array_equal(ell[di], temperatures*6./160.)
        else:
            np.testing.assert_array_equal(temperatures[ti], [160.])
            np.testing.assert_array_equal(ell[di], [6.])
        mesh = ax.pcolormesh(ell, temperatures, surface, shading="nearest", cmap="Blues_r", vmin=0., vmax=25.)
        ax.scatter(ell[di], temperatures[ti], s=46, marker="o", facecolor="none",
                   edgecolor="white", linewidth=1.3, label="Grid minimum")
        ax.plot(6., 160., "+", color="#f19143", markersize=12, markeredgewidth=2, label="Input pair")
        ax.set(title=title, xlabel=r"Log duration, $\log_{10}(t/\mathrm{s})$",
               xticks=(0, 3, 6, 9, 12, 15), ylim=(30, 310))
    axes[0].set_ylabel("Event temperature (°C)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9,
               facecolor="#334e65", labelcolor="white", frameon=True, framealpha=1., edgecolor="none")
    fig.colorbar(mesh, ax=axes, location="right", shrink=.75, aspect=23, extend="max",
                 label=r"Increase in weighted misfit, $\Delta\chi^2$")
    return fig
