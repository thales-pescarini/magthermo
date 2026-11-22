"""Draw the population quadrature, two saved vector curves and the forward grid."""
import json

import matplotlib.pyplot as plt
import numpy as np

import nature_style as ns

DURATION = r"Duration, $\log_{10}(t/\mathrm{s})$"


def stereo(vectors):
    vectors = np.asarray(vectors)
    return vectors[..., 1:] / (1. + vectors[..., :1])


def population(directory):
    fig, (bins, triangle) = plt.subplots(1, 2, figsize=(ns.DOUBLE, 68 * ns.MM),
                                         gridspec_kw={"width_ratios": (1.25, 1)}, layout="constrained")
    data = np.genfromtxt(directory / "population_weights.csv", names=True, delimiter=",")
    aspects = np.union1d(data["aspect_low"], data["aspect_high"])
    diameters = np.union1d(data["diameter_low_nm"], data["diameter_high_nm"])
    weights = np.full((len(diameters)-1, len(aspects)-1), np.nan)
    i = np.searchsorted(diameters, data["diameter_low_nm"])
    j = np.searchsorted(aspects, data["aspect_low"])
    weights[i, j] = data["parent_capacity_fraction"]
    assert np.isfinite(weights).all() and (weights >= 0.).all()
    mesh = bins.pcolormesh(aspects, diameters, 100*weights, cmap=ns.CMAP_SEQ, rasterized=True)
    ns.frame(bins)
    ns.colorbar(fig, mesh, bins, "Parent magnetic capacity per bin (%)", shrink=.9)
    bins.set(xlim=(1., 1.4), ylim=(15, 85), xlabel="Length / width",
             ylabel="Volume-equivalent diameter (nm)")

    path = directory.parents[1] / "config/crystal_orientations.json"
    nodes = [r for r in json.loads(path.read_text())["nodes"] if r["level"] == 1]
    np.testing.assert_allclose(sum(r["weight"] for r in nodes), 1.)
    for r in nodes:
        vertices = np.array(r["vertices"])
        boundary = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            segment = np.linspace(start, end, 40)
            segment /= np.linalg.norm(segment, axis=1)[:, None]
            boundary.extend(stereo(segment))
        x, y = np.array(boundary).T
        triangle.fill(x, y, facecolor=ns.FILL, edgecolor=ns.BLACK, lw=.5)
    for r in nodes:
        x, y = stereo([r["c_x"], r["c_y"], r["c_z"]])
        triangle.plot(x, y, "o", ms=2.6, color=ns.BLACK)
        triangle.annotate(f"{100*r['weight']:.1f}%", (x, y), xytext=(0, -9),
                          textcoords="offset points", ha="center", fontsize=6)
    for v, label, shift in [([1, 0, 0], "[100]", (0, -11)),
                             ([1, 1, 0], "[110]", (5, -10)),
                             ([1, 1, 1], "[111]", (0, 5))]:
        x, y = stereo(v / np.linalg.norm(v))
        triangle.annotate(label, (x, y), xytext=shift, textcoords="offset points", ha="center")
    triangle.set(aspect="equal", xlim=(-.055, .49), ylim=(-.05, .41))
    triangle.axis("off")
    for ax, letter in zip((bins, triangle), "ab"):
        ns.panel_label(ax, letter)
    return fig


def forward_grid(directory):
    bank = np.genfromtxt(directory / "forward_grid.csv", names=True, delimiter=",")
    curves = np.genfromtxt(directory / "forward_curves.csv", names=True, delimiter=",")
    temperatures, ti = np.unique(bank["event_C"], return_inverse=True)
    durations, di = np.unique(bank["log10_seconds"], return_inverse=True)
    initial_z = np.full((len(temperatures), len(durations)), np.nan)
    initial_z[ti, di] = bank["initial_Mz"]
    assert initial_z.shape == (14, 121) and np.isfinite(initial_z).all()

    fig = plt.figure(figsize=(ns.DOUBLE, 80 * ns.MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.12))
    z_axis = fig.add_subplot(grid[0, 0])
    x_axis = fig.add_subplot(grid[1, 0], sharex=z_axis)
    template = fig.add_subplot(grid[:, 1])
    mesh = template.pcolormesh(durations, temperatures, initial_z, shading="nearest", cmap=ns.CMAP_SEQ,
                               rasterized=True)
    events = []
    for ell, color, marker in [(3, ns.BLUE, "o"), (12, ns.VERMILION, "s")]:
        selected = curves[curves["log10_seconds"] == ell]
        np.testing.assert_array_equal(selected["step"], np.arange(20))
        event = selected["event_C"][0]
        np.testing.assert_array_equal(selected["event_C"], event)
        np.testing.assert_equal(selected["Mz"][0], initial_z[temperatures == event, durations == ell][0])
        events.append(event)
        peaks = selected["lab_plateau_C"].copy()
        assert np.isnan(peaks[0]) and np.all(np.diff(peaks[1:]) > 0.)
        peaks[0] = 20.
        for ax, component in [(z_axis, "Mz"), (x_axis, "Mx")]:
            ax.plot(peaks, selected[component], lw=.9, color=color, marker=marker, ms=2.4, mew=0.,
                    label=rf"$10^{{{ell}}}$ s")
            ax.plot(peaks[0], selected[component][0], marker=marker, ms=4.2, mew=.7,
                    markerfacecolor="white", markeredgecolor=color)
        template.scatter(ell, event, s=20, marker=marker, facecolor=color, edgecolor="white", lw=.7, zorder=3)
    title = f"Event at {events[0]:.0f} °C" if len(set(events)) == 1 else "Event duration"
    z_axis.legend(loc="upper right", ncols=2, title=title, alignment="left", columnspacing=1.,
                  fontsize=7, title_fontsize=7)  # 7 pt keeps the superscripts above 5 pt
    z_axis.set(ylabel=r"$M_z$ / parent capacity")
    z_axis.tick_params(labelbottom=False)
    x_axis.set(xlabel="Laboratory plateau (°C)", ylabel=r"$M_x$ / parent capacity",
               xticks=(20, 100, 200, 300, 400, 500), xticklabels=("NRM", "100", "200", "300", "400", "500"))
    fig.align_ylabels([z_axis, x_axis])
    ns.frame(template)
    template.set(xlabel=DURATION, ylabel="Event temperature (°C)",
                 ylim=(30, 310), xticks=(0, 3, 6, 9, 12, 15))
    ns.colorbar(fig, mesh, template, r"Initial $M_z$ / parent capacity", shrink=.7)
    for ax, letter in zip((z_axis, x_axis, template), "abc"):
        ns.panel_label(ax, letter)
    return fig
