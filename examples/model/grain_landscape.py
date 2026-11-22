"""Draw a grain section, its directional energy and a saddle joining two minima."""
import json
import warnings

import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from scipy.constants import Boltzmann

import nature_style as ns
from gate1_verified_methods import riem

ENERGY = r"$e - e_\mathrm{min}$ (kJ m$^{-3}$)"


def read_grain(directory):
    data = json.loads((directory / "grain_states.json").read_text())
    with warnings.catch_warnings():
        # SDCC returns the identity for [100] to [100] after normalizing a zero cross product.
        warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning,
                                module=r"sdcc\.energy$", lineno=417)
        from e1_core import PointModel

        model = PointModel(data["log_polar_to_equatorial_ratio"], data["shape_axis_crystal"],
                           data["temperature_C"], data["field_crystal_T"],
                           TM=data["titanomagnetite_percent"])
    for p in data["points"]:
        u = np.asarray(p["direction"])
        gradient, hessian, _ = riem(model, u)
        np.testing.assert_allclose(np.linalg.norm(u), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(model.energy_np(u), p["energy_J_m3"], atol=1e-7, rtol=0)
        assert np.linalg.norm(gradient) <= 1e-9 * model.escale  # Stationarity tolerance of the saved census.
        assert np.count_nonzero(np.linalg.eigvalsh(hessian) < 0.) == p["index"]
    return model, data


def grain_geometry(model):
    fig = plt.figure(figsize=(ns.DOUBLE, 74 * ns.MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.3), height_ratios=(1.9, 1))
    section = fig.add_subplot(grid[:, 0])
    terms = fig.add_subplot(grid[0, 1])
    field = fig.add_subplot(grid[1, 1], sharex=terms)

    field_axis = model.B / np.linalg.norm(model.B)
    transverse = model.c - np.dot(model.c, field_axis) * field_axis
    transverse /= np.linalg.norm(transverse)
    axis = np.array([model.c @ field_axis, model.c @ transverse])
    short_angle = np.degrees(np.arctan2(axis[1], axis[0]))
    ratio = np.exp(model.s)
    equatorial_diameter = 45. / np.cbrt(ratio)
    section.add_patch(Ellipse((0, 0), equatorial_diameter * ratio, equatorial_diameter,
                              angle=short_angle, facecolor=ns.FILL, edgecolor=ns.BLACK, lw=.6))
    section.axhline(0., color=ns.RULE, lw=.4, zorder=1)
    section.axvline(0., color=ns.RULE, lw=.4, zorder=1)
    section.plot([-26 * axis[0], 26 * axis[0]], [-26 * axis[1], 26 * axis[1]],
                 color=ns.INK_SOFT, ls=(0, (4, 2)), lw=.6, zorder=2)
    for tip, color in [((26, 0), ns.BLACK), ((-8, 17), ns.BLUE)]:
        section.annotate("", xy=tip, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=.9, color=color,
                                         shrinkA=0, shrinkB=0), zorder=3)
    section.text(26, -2.2, r"$\mathbf{B}$", ha="center", va="top")
    section.text(-9, 18.4, r"$\mathbf{u}$", ha="center")
    section.text(24 * axis[0], 24 * axis[1] + 2, "short axis", color=ns.INK_SOFT, ha="center", fontsize=6)
    section.plot(0., 0., "o", ms=2, color=ns.BLACK, zorder=4)
    section.set(xlim=(-29, 31), ylim=(-28, 28), aspect="equal",
                xlabel="Distance along [100] (nm)", ylabel="Distance along [031] (nm)")

    angles = np.linspace(0., 360., 361)
    theta = np.deg2rad(angles)
    u = np.cos(theta)[:, None] * field_axis + np.sin(theta)[:, None] * transverse
    q = u ** 2
    cubic = model.k1 * (q[:, 0]*q[:, 1] + q[:, 1]*q[:, 2] + q[:, 2]*q[:, 0])
    cubic += model.k2 * np.prod(q, axis=1)
    shape = np.einsum("ni,ij,nj->n", u, model.D, u)
    zeeman = -u @ model.h
    total = np.array([model.energy_np(v) for v in u])
    np.testing.assert_allclose(cubic + shape + zeeman, total, rtol=1e-14)
    for ax in (terms, field):
        ax.axhline(0., color=ns.RULE, lw=.4, zorder=0)
    for energy, color, label, style, width in [(cubic, ns.BLUE, "Cubic", (0, (4, 1.5)), .9),
                                               (shape, ns.VERMILION, "Shape", (0, (5, 1.5, 1, 1.5)), .9),
                                               (total, ns.BLACK, "Total", "-", 1.1)]:
        terms.plot(angles, (energy-energy[0])/1000, color=color, lw=width, ls=style, label=label)
    terms.set_ylim(bottom=1.38 * (total.min() - total[0]) / 1000)
    terms.legend(ncols=3, loc="lower center", handlelength=2.6)
    terms.set(ylabel=r"$\Delta e$ (kJ m$^{-3}$)")
    terms.tick_params(labelbottom=False)
    field.plot(angles, zeeman-zeeman[0], color=ns.GREEN, lw=1.)
    field.set(xlabel="Angle from [100] toward [031] (°)", ylabel=r"$\Delta e_\mathrm{Z}$ (J m$^{-3}$)",
              xticks=(0, 90, 180, 270, 360), xlim=(0, 360))
    fig.align_ylabels([terms, field])
    for ax, letter in zip((section, terms, field), "abc"):
        ns.panel_label(ax, letter)
    return fig


def lower_hemisphere_xy(directions):
    """Lambert equal-area projection; the equator has unit radius."""
    u = np.asarray(directions)
    if np.any(u[..., 2] > 1e-12):
        raise ValueError("The projection requires directions in the lower hemisphere.")
    return u[..., :2] / np.sqrt(1. - u[..., 2:3])


def cubic_edges(ax):
    """Project directions from the crystal center to the visible edges of a cube."""
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=float)
    for start, adjacent in zip(corners, np.roll(corners, -1, axis=0)):
        equator = np.array([start[0], start[1], 0.])
        for end in (adjacent, equator):
            edge = np.linspace(start, end, 81)
            edge /= np.linalg.norm(edge, axis=1, keepdims=True)
            x, y = lower_hemisphere_xy(edge).T
            line, = ax.plot(x, y, color="white", lw=.8, ls=(0, (3, 2)), zorder=3)
            line.set_path_effects([effects.withStroke(linewidth=1.55, foreground=ns.INK_SOFT)])
    return line


def energy_landscape(model, data):
    fig = plt.figure(figsize=(ns.DOUBLE, 84 * ns.MM), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.3, 1.))
    disk = fig.add_subplot(grid[0])
    barrier = fig.add_subplot(grid[1])
    points = data["points"]
    energies = np.array([p["energy_J_m3"] for p in points])
    u = np.array([p["direction"] for p in points])

    radius, angle = np.meshgrid(np.linspace(0., 1., 129), np.linspace(0., 2*np.pi, 361),
                                indexing="ij")
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    scale = np.sqrt(2. - radius**2)
    directions = np.dstack((x * scale, y * scale, radius**2 - 1.))
    density = np.array([model.energy_np(v) for v in directions.reshape(-1, 3)])
    relative = (density.reshape(radius.shape) - energies.min()) / 1000
    levels = np.linspace(0., 12., 25)
    disk.set_rasterization_zorder(1)
    surface = disk.contourf(x, y, relative, levels=levels, cmap=ns.CMAP_SEQ, zorder=0,
                            extend="max" if relative.max() > levels[-1] else "neither")
    disk.contour(x, y, relative, levels=levels[2:-1:2], colors=ns.BLACK, linewidths=.2, alpha=.35)
    disk.plot([-1, 1], [0, 0], lw=.35, color=ns.INK_SOFT)
    disk.plot([0, 0], [-1, 1], lw=.35, color=ns.INK_SOFT)
    wire = cubic_edges(disk)
    disk.add_patch(Circle((0, 0), 1., fill=False, ec=ns.BLACK, lw=.6, zorder=4))
    disk.text(1.03, 0, "[100]", va="center", fontsize=6)
    disk.text(0., 1.03, "[010]", ha="center", va="bottom", fontsize=6)
    disk.set(xlim=(-1.06, 1.2), ylim=(-1.04, 1.12), aspect="equal")
    disk.set_axis_off()

    indices = np.array([p["index"] for p in points])
    markers = []
    for index, label, marker, face, size in [(0, "Minimum", "o", ns.BLACK, 16),
                                             (1, "Saddle", "D", ns.ORANGE, 13),
                                             (2, "Maximum", "X", "white", 22)]:
        visible = (indices == index) & (u[:, 2] <= 0.)
        px, py = lower_hemisphere_xy(u[visible]).T
        markers.append(disk.scatter(px, py, s=size, marker=marker, facecolor=face,
                                    edgecolor=ns.BLACK, lw=.5, zorder=6, label=label))

    channel = data["connections"][0]
    a, b = channel["minima"]
    saddle = channel["saddle"]
    for path, minimum in zip(channel["descent_paths"], (a, b), strict=True):
        path = np.asarray(path)
        np.testing.assert_allclose(path[-1], u[minimum], rtol=0, atol=1e-6)
        path = path / np.linalg.norm(path, axis=1, keepdims=True)
        px, py = lower_hemisphere_xy(path).T
        descent, = disk.plot(px, py, color=ns.VERMILION, lw=1.1, zorder=5, solid_capstyle="round")
    ids = [a, saddle, b]
    for idx, label in zip(ids, ("A", "S", "B")):
        ns.halo_label(disk, label, lower_hemisphere_xy(u[idx]), xytext=(4, 3.5),
                      textcoords="offset points", zorder=7)
    disk.legend([*markers, wire, descent], ["Minimum", "Saddle", "Maximum", "Cubic edges", "Descent path"],
                loc="upper center", bbox_to_anchor=(.5, .0), ncols=5, handlelength=1.4, columnspacing=1.,
                handletextpad=.35, borderaxespad=0.)
    ns.colorbar(fig, surface, disk, ENERGY, shrink=.62, ticks=[0, 3, 6, 9, 12])

    heights = (energies[ids] - energies.min()) / 1000
    volume = np.pi / 6 * (45e-9)**3
    delta = (energies[saddle] - energies[[a, b]]) * volume / (Boltzmann * (model.T+273.15))
    for x, y, color in zip(range(3), heights, (ns.BLACK, ns.ORANGE, ns.BLACK)):
        barrier.hlines(y, x-.30, x+.30, colors=color, lw=1.6)
    for left, right, dx, y, value in [(0, 1, .45, heights[0], delta[0]),
                                      (1, 2, 1.55, heights[2], delta[1])]:
        barrier.hlines(heights[1], left+.2, right-.2, color=ns.RULE, ls=(0, (1, 1.5)), lw=.6)
        barrier.annotate("", (dx, heights[1]), (dx, y),
                         arrowprops=dict(arrowstyle="<|-|>", mutation_scale=5, color=ns.BLACK, lw=.5,
                                         shrinkA=0, shrinkB=0))
        barrier.text(dx, (y+heights[1])/2, f"{value:.1f} " + r"$k_\mathrm{B}T$",
                     ha="center", va="center",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    barrier.set(xlim=(-.48, 2.48), ylim=(-.11, heights[1]*1.19), xticks=[0, 1, 2],
                xticklabels=["Minimum A", "Saddle S", "Minimum B"], ylabel=ENERGY)
    barrier.spines["bottom"].set_visible(False)
    barrier.tick_params(axis="x", length=0, pad=3)
    for ax, letter in zip((disk, barrier), "ab"):
        ns.panel_label(ax, letter)
    return fig
