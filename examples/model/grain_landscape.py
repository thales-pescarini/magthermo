"""Draw a grain section, its directional energy and a saddle joining two minima."""
import json
import warnings

import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
import numpy as np
from matplotlib.patches import Circle, Ellipse
from scipy.constants import Boltzmann

from gate1_verified_methods import riem


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
    fig = plt.figure(figsize=(9.6, 4.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.22), height_ratios=(2.15, 1))
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
                              angle=short_angle, facecolor="#e0eaf0", edgecolor="#496273", lw=1.5))
    section.axhline(0., color="#adb8be", lw=.7)
    section.axvline(0., color="#adb8be", lw=.7)
    section.plot([-26 * axis[0], 26 * axis[0]], [-26 * axis[1], 26 * axis[1]],
                 color="#a95600", ls="--", lw=1.5)
    for tip, color in [((26, 0), "#23638a"), ((-8, 17), "#1a705c")]:
        section.annotate("", xy=tip, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", lw=2.4, color=color))
    section.text(26, -2.2, r"$\mathbf{B}$", color="#23638a", ha="center", va="top")
    section.text(-9, 18.4, r"$\mathbf{u}$", color="#1a705c", ha="center")
    section.text(24 * axis[0], 24 * axis[1] + 2, "short axis", color="#954d00", ha="center")
    section.plot(0., 0., "o", ms=3, color="#263849")
    section.set(xlim=(-29, 31), ylim=(-28, 28), aspect="equal",
                xlabel="Distance along [100] (nm)", ylabel="Distance along [031] (nm)",
                title="Grain section in the crystal frame")

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
    for energy, color, label, style in [(cubic, "#3969a1", "Cubic", "--"),
                                        (shape, "#b35d19", "Shape", "-."),
                                        (total, "#192f3f", "Total", "-")]:
        terms.plot(angles, (energy-energy[0])/1000, color=color, lw=2, ls=style, label=label)
    terms.set_ylim(bottom=1.38 * (total.min() - total[0]) / 1000)
    terms.legend(ncols=3, loc="lower center", fontsize=9)
    terms.set(ylabel="Change in energy\n(kJ m⁻³)", title="Energy as the moment rotates in this plane")
    terms.tick_params(labelbottom=False)
    field.plot(angles, zeeman-zeeman[0], c="#1a705c", lw=2)
    field.set(xlabel="Angle from [100] toward [031] (°)", ylabel="Zeeman change\n(J m⁻³)",
              xticks=(0, 90, 180, 270, 360), xlim=(0, 360))
    for ax in (terms, field):
        ax.grid(axis="y", color="#dbe2e6", lw=.65)
        ax.set_axisbelow(True)
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
            line, = ax.plot(x, y, color="white", lw=1.45, ls=(0, (3.3, 2.1)), zorder=3)
            line.set_path_effects([effects.withStroke(linewidth=2.4, foreground="#42505c")])
    return line


def energy_landscape(model, data):
    fig = plt.figure(figsize=(9.6, 5.25), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.38, 1.))
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
    surface = disk.contourf(x, y, relative, levels=np.linspace(0., 12., 25), cmap="viridis")
    disk.plot([-1, 1], [0, 0], lw=.65, color="#24343e", alpha=.52)
    disk.plot([0, 0], [-1, 1], lw=.65, color="#24343e", alpha=.52)
    wire = cubic_edges(disk)
    disk.add_patch(Circle((0, 0), 1., fill=False, ec="#223441", lw=1.25, zorder=4))
    disk.text(1.025, 0, "[100]", va="center", fontsize=9)
    disk.text(0., 1.025, "[010]", ha="center", va="bottom", fontsize=9)
    disk.set(xlim=(-1.08, 1.21), ylim=(-1.04, 1.13), aspect="equal",
             title="Magnetization directions in the crystal frame")
    disk.set_axis_off()

    indices = np.array([p["index"] for p in points])
    markers = []
    for index, label, marker, color in [(0, "Minimum", "o", "#f7fbff"), (1, "Saddle", "D", "#ffb260"),
                                       (2, "Maximum", "X", "#ef82a7")]:
        visible = (indices == index) & (u[:, 2] <= 0.)
        px, py = lower_hemisphere_xy(u[visible]).T
        markers.append(disk.scatter(px, py, s=54, marker=marker, facecolor=color,
                                    edgecolor="#14222c", lw=1.05, zorder=6, label=label))

    channel = data["connections"][0]
    a, b = channel["minima"]
    saddle = channel["saddle"]
    for path, minimum in zip(channel["descent_paths"], (a, b), strict=True):
        path = np.asarray(path)
        np.testing.assert_allclose(path[-1], u[minimum], rtol=0, atol=1e-6)
        path = path / np.linalg.norm(path, axis=1, keepdims=True)
        px, py = lower_hemisphere_xy(path).T
        descent, = disk.plot(px, py, color="#ff9c51", lw=2., zorder=5)
        descent.set_path_effects([effects.withStroke(linewidth=3.25, foreground="#333b42")])
    ids = [a, saddle, b]
    for idx, label in zip(ids, ("A", "S", "B")):
        disk.annotate(label, lower_hemisphere_xy(u[idx]), xytext=(6, 6), textcoords="offset points",
                      color="white", weight="bold", fontsize=9, zorder=7,
                      bbox=dict(boxstyle="round,pad=0.1", fc="#1b303d", ec="none", alpha=.85))
    fig.legend([*markers, wire, descent], ["Minimum", "Saddle", "Maximum", "Cubic edges", "Descent path"],
               loc="outside lower center", ncols=5, columnspacing=1.55, handletextpad=.45, fontsize=9)
    fig.colorbar(surface, ax=disk, location="bottom", shrink=.84, pad=.045, aspect=33,
                 ticks=[0, 3, 6, 9, 12], label="Energy above the deepest minimum (kJ m⁻³)")
    heights = (energies[ids] - energies.min()) / 1000
    volume = np.pi / 6 * (45e-9)**3
    delta = (energies[saddle] - energies[[a, b]]) * volume / (Boltzmann * (model.T+273.15))
    for x, y, color in zip(range(3), heights, ("#337298", "#b86416", "#337298")):
        barrier.hlines(y, x-.30, x+.30, colors=color, lw=3)
    for left, right, dx, y, value in [(0, 1, .45, heights[0], delta[0]),
                                      (1, 2, 1.55, heights[2], delta[1])]:
        barrier.hlines(heights[1], left+.2, right-.2, color="#9eaab3", ls=":", lw=1)
        barrier.annotate("", (dx, heights[1]), (dx, y),
                         arrowprops=dict(arrowstyle="<->", color="#53616b", lw=1.4))
        barrier.text(dx, (y+heights[1])/2, f"{value:.1f} " + r"$k_BT$",
                     ha="center", va="center", fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.7))
    barrier.set(xlim=(-.48, 2.48), ylim=(-.11, heights[1]*1.19), xticks=[0, 1, 2],
                xticklabels=["Minimum A", "Saddle S", "Minimum B"],
                ylabel="Relative energy density (kJ m⁻³)", title="Barriers across one saddle")
    return fig
