"""Draw a grain section, its directional energy and a saddle joining two minima."""
import json
import warnings

import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

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


def angular_steps(directions):
    before, after = directions[:-1], directions[1:]
    return np.arctan2(np.linalg.norm(np.cross(before, after), axis=1),
                      np.einsum("ij,ij->i", before, after))


def descent_directions(saddle, saved, minimum):
    """Sample the short spherical arcs between consecutive descent nodes."""
    np.testing.assert_allclose(saved[-1], minimum, rtol=0, atol=1e-6)
    nodes = np.vstack((saddle, saved, minimum))
    nodes /= np.linalg.norm(nodes, axis=1, keepdims=True)
    pieces = [nodes[:1]]
    for start, end, angle in zip(nodes[:-1], nodes[1:], angular_steps(nodes), strict=True):
        if angle == 0.:
            continue
        count = max(1, int(np.ceil(np.degrees(angle) / .25)))
        t = np.arange(1, count + 1)[:, None] / count
        part = (np.sin((1 - t) * angle) * start + np.sin(t * angle) * end) / np.sin(angle)
        pieces.append(part)
    return np.vstack(pieces)


def grain_directions(ax, model):
    """Show the applied field and the long diameters of the illustrated oblate grain."""
    field = model.B / np.linalg.norm(model.B)
    rim = lower_hemisphere_xy(field)
    ax.annotate("", xy=rim, xytext=.68 * rim, zorder=8,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=11, color=ns.BLACK,
                                lw=1.2, shrinkA=0, shrinkB=0,
                                path_effects=[effects.withStroke(linewidth=2.6, foreground="white")]))
    ns.halo_label(ax, "Field [100]", rim, xytext=(0, 6), ha="right",
                  textcoords="offset points", fontsize=6.5, zorder=8)

    # The lower half of the equatorial great circle spans all unoriented long diameters.
    horizontal = np.cross([0., 0., 1.], model.c)
    horizontal /= np.linalg.norm(horizontal)
    downward = np.cross(horizontal, model.c)
    angle = np.linspace(0., np.pi, 241)[:, None]
    equator = np.cos(angle) * horizontal + np.sin(angle) * downward
    x, y = lower_hemisphere_xy(equator).T
    plane_color = "#784078"
    plane, = ax.plot(x, y, color=plane_color, ls=(0, (5, 2)), lw=.9, zorder=4)
    plane.set_path_effects([effects.withStroke(linewidth=1.7, foreground="white")])

    diameter = np.cross(model.c, field)
    diameter /= np.linalg.norm(diameter)
    if diameter[2] > 0.:
        diameter = -diameter
    xy = lower_hemisphere_xy(diameter)
    ax.scatter(*xy, s=23, marker="s", color=plane_color, edgecolor="white", lw=.7, zorder=7)
    ns.halo_label(ax, r"Long diameter $[0\,1\,\overline{3}]$", xy, xytext=(-6, -8),
                  ha="right", textcoords="offset points", fontsize=6.2, zorder=8)
    return plane


def energy_landscape(model, data):
    fig = plt.figure(figsize=(ns.DOUBLE, 91 * ns.MM), layout="constrained")
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
    plane = grain_directions(disk, model)
    disk.text(0., 1.03, "[010]", ha="center", va="bottom", fontsize=6)
    disk.set(xlim=(-1.05, 1.05), ylim=(-1.04, 1.12), aspect="equal")
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
    branch_a, branch_b = [descent_directions(u[saddle], path, u[minimum])
                          for path, minimum in zip(channel["descent_paths"], (a, b), strict=True)]
    path = np.vstack((branch_a[::-1], branch_b[1:]))
    px, py = lower_hemisphere_xy(path).T
    descent, = disk.plot(px, py, color=ns.VERMILION, lw=1.1, zorder=5, solid_capstyle="round")
    ids = [a, saddle, b]
    for idx, label in zip(ids, ("A", "S", "B")):
        ns.halo_label(disk, label, lower_hemisphere_xy(u[idx]), xytext=(4, 3.5),
                      textcoords="offset points", zorder=7)
    disk.legend([*markers, wire, descent, plane],
                ["Minimum", "Saddle", "Maximum", "Cubic edges", "Transition path", "Equatorial plane"],
                loc="upper center", bbox_to_anchor=(.5, .0), ncols=3, handlelength=1.7,
                columnspacing=1., handletextpad=.4, borderaxespad=0.)
    ns.colorbar(fig, surface, disk, ENERGY, shrink=.64, ticks=[0, 3, 6, 9, 12])

    distance = np.degrees(np.r_[0., np.cumsum(angular_steps(path))])
    profile = (np.array([model.energy_np(v) for v in path]) - energies.min()) / 1000
    stations = [0, len(branch_a) - 1, len(path) - 1]
    np.testing.assert_allclose(profile[stations], (energies[ids] - energies.min()) / 1000,
                               rtol=0, atol=1e-9)
    barrier.plot(distance, profile, color=ns.VERMILION, lw=1.45)
    for k, label in zip(stations, ("A", "S", "B"), strict=True):
        is_saddle = label == "S"
        barrier.scatter(distance[k], profile[k], marker="D" if is_saddle else "o",
                        s=18, facecolor=ns.ORANGE if is_saddle else ns.BLACK,
                        edgecolor=ns.BLACK, lw=.5, zorder=3)
        barrier.annotate(label, (distance[k], profile[k]), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
    barrier.set(xlabel="Angular distance along path (°)", ylabel=ENERGY,
                ylim=(-.10, 1.18 * profile.max()))
    for ax, letter in zip((disk, barrier), "ab"):
        ns.panel_label(ax, letter)
    return fig
