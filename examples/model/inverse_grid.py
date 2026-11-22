"""Illustrate amplitude profiling with the two-component analytic estimator test."""
import matplotlib.pyplot as plt
import numpy as np

import nature_style as ns
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
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(ns.ONE_HALF_WIDE, 62 * ns.MM),
                             layout="constrained")
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
        mesh = ax.pcolormesh(ell, temperatures, surface, shading="nearest", cmap=ns.CMAP_MISFIT,
                             vmin=0., vmax=25., rasterized=True)
        ax.scatter(ell[di], temperatures[ti], s=11, marker="o", facecolor="white",
                   edgecolor=ns.BLACK, linewidth=.5, zorder=3, label="Grid minimum")
        ax.plot(6., 160., "+", color=ns.VERMILION, markersize=7, markeredgewidth=1.2, zorder=4,
                label="Input pair")
        ns.frame(ax)
        ax.set(title=title, xlabel=r"Duration, $\log_{10}(t/\mathrm{s})$",
               xticks=(0, 3, 6, 9, 12, 15), ylim=(30, 310))
    axes[0].set_ylabel("Event temperature (°C)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    ns.colorbar(fig, mesh, axes, r"$\Delta\chi^2$", shrink=.7, extend="max", label_on_top=True)
    for ax, letter in zip(axes, "ab"):
        ns.panel_label(ax, letter)
    return fig
