"""Draw the population quadrature, two saved vector curves and the forward grid."""
import json

import matplotlib.pyplot as plt
import numpy as np


def stereo(vectors):
    vectors = np.asarray(vectors)
    return vectors[..., 1:] / (1. + vectors[..., :1])


def population(directory):
    fig, (bins, triangle) = plt.subplots(1, 2, figsize=(9.6, 4.3),
                                         gridspec_kw={"width_ratios": (1.22, 1)}, layout="constrained")
    data = np.genfromtxt(directory / "population_weights.csv", names=True, delimiter=",")
    aspects = np.union1d(data["aspect_low"], data["aspect_high"])
    diameters = np.union1d(data["diameter_low_nm"], data["diameter_high_nm"])
    weights = np.full((len(diameters)-1, len(aspects)-1), np.nan)
    i = np.searchsorted(diameters, data["diameter_low_nm"])
    j = np.searchsorted(aspects, data["aspect_low"])
    weights[i, j] = data["parent_capacity_fraction"]
    assert np.isfinite(weights).all() and (weights >= 0.).all()
    mesh = bins.pcolormesh(aspects, diameters, 100*weights, cmap="YlGnBu")
    fig.colorbar(mesh, ax=bins, location="bottom", shrink=.85, pad=.04, aspect=30,
                 label="Parent magnetic capacity per bin (%)")
    bins.set(xlim=(1., 1.4), ylim=(15, 85), xlabel="Length / width",
             ylabel="Volume-equivalent diameter (nm)", title="Size and shape of the modeled population")

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
        triangle.fill(x, y, facecolor="#e4f0f1", edgecolor="#61818b", lw=1.05)
    for r in nodes:
        x, y = stereo([r["c_x"], r["c_y"], r["c_z"]])
        triangle.plot(x, y, "o", ms=6, color="#175974")
        triangle.annotate(f"{100*r['weight']:.1f}%", (x, y), xytext=(0, -13),
                          textcoords="offset points", ha="center", color="#213b4e", fontsize=9)
    for v, label, shift in [([1, 0, 0], "[100]", (0, -15)),
                             ([1, 1, 0], "[110]", (5, -12)),
                             ([1, 1, 1], "[111]", (0, 6))]:
        x, y = stereo(v / np.linalg.norm(v))
        triangle.annotate(label, (x, y), xytext=shift, textcoords="offset points", ha="center")
    triangle.set(aspect="equal", xlim=(-.055, .49), ylim=(-.05, .41),
                 title="Shape-axis orientations in the crystal")
    triangle.axis("off")
    return fig


def forward_grid(directory):
    bank = np.genfromtxt(directory / "forward_grid.csv", names=True, delimiter=",")
    curves = np.genfromtxt(directory / "forward_curves.csv", names=True, delimiter=",")
    temperatures, ti = np.unique(bank["event_C"], return_inverse=True)
    durations, di = np.unique(bank["log10_seconds"], return_inverse=True)
    initial_z = np.full((len(temperatures), len(durations)), np.nan)
    initial_z[ti, di] = bank["initial_Mz"]
    assert initial_z.shape == (14, 121) and np.isfinite(initial_z).all()

    fig = plt.figure(figsize=(9.6, 4.9), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.12))
    z_axis = fig.add_subplot(grid[0, 0])
    x_axis = fig.add_subplot(grid[1, 0], sharex=z_axis)
    template = fig.add_subplot(grid[:, 1])
    mesh = template.pcolormesh(durations, temperatures, initial_z, shading="nearest", cmap="cividis")
    for ell, color, marker in [(3, "#096985", "o"), (12, "#b84e2b", "s")]:
        selected = curves[curves["log10_seconds"] == ell]
        np.testing.assert_array_equal(selected["step"], np.arange(20))
        event = selected["event_C"][0]
        np.testing.assert_array_equal(selected["event_C"], event)
        np.testing.assert_equal(selected["Mz"][0], initial_z[temperatures == event, durations == ell][0])
        peaks = selected["lab_plateau_C"].copy()
        assert np.isnan(peaks[0]) and np.all(np.diff(peaks[1:]) > 0.)
        peaks[0] = 20.
        for ax, component in [(z_axis, "Mz"), (x_axis, "Mx")]:
            ax.plot(peaks, selected[component], lw=1.6, color=color, marker=marker, ms=3.4,
                    label=rf"$10^{{{ell}}}$ s")
            ax.plot(peaks[0], selected[component][0], marker=marker, ms=6,
                    markerfacecolor="white", markeredgecolor=color)
        template.scatter(ell, event, s=62, marker=marker, facecolor=color, edgecolor="white", lw=1.5)
    z_axis.set(ylabel=r"$M_z$ / parent capacity", title="Demagnetization after an event at 160 °C")
    z_axis.legend(loc="upper right", ncols=2, columnspacing=.9, fontsize=9)
    z_axis.tick_params(labelbottom=False)
    x_axis.set(xlabel="Laboratory plateau (°C)", ylabel=r"$M_x$ / parent capacity",
               xticks=(20, 100, 200, 300, 400, 500), xticklabels=("NRM", "100", "200", "300", "400", "500"))
    for ax in (z_axis, x_axis):
        ax.grid(axis="y", color="#dce2e7", lw=.65)
        ax.set_axisbelow(True)
    template.set(xlabel=r"Log duration, $\log_{10}(t/\mathrm{s})$", ylabel="Event temperature (°C)",
                 ylim=(30, 310), xticks=(0, 3, 6, 9, 12, 15), title="Initial remanence on the forward grid")
    fig.colorbar(mesh, ax=template, location="bottom", shrink=.86, pad=.04, aspect=30,
                 label=r"Initial $M_z$ / parent capacity")
    return fig
