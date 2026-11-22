"""Shared typography, colors and export settings for the MAGTHERMO figures."""
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as effects

MM = 1. / 25.4
SINGLE, ONE_HALF, ONE_HALF_WIDE, DOUBLE = 89 * MM, 120 * MM, 136 * MM, 183 * MM

# Okabe-Ito colours (Wong, Nat. Methods 8, 441; 2011). Orange is light on white, so it is used
# only as a filled marker with a black edge or as a thick line.
BLACK, BLUE, VERMILION, GREEN = "#000000", "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE = "#E69F00", "#56B4E9", "#CC79A7"
RULE = "#b3b3b3"     # zero lines and reference axes
FILL = "#ebebeb"     # neutral solid fills
INK_SOFT = "#4d4d4d"  # secondary annotation text and context lines

# One sequential family across the set; low values light except for misfit, where the valley
# of good fits should be the salient feature.
CMAP_SEQ = "YlGnBu"
CMAP_MISFIT = "Blues_r"

PNG_DPI = 300
RASTER_DPI = 600  # resolution of rasterized meshes inside the PDF

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "Nimbus Sans",
                        "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6,
    "legend.title_fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold", "mathtext.sf": "sans", "mathtext.cal": "sans:italic",
    "axes.unicode_minus": True,
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK,
    "xtick.color": BLACK, "ytick.color": BLACK,
    "axes.linewidth": .5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True, "axes.labelpad": 2.,
    "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.titlepad": 4.,
    "axes.prop_cycle": mpl.cycler(color=[BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKY, BLACK]),
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.minor.width": .4, "ytick.minor.width": .4,
    "xtick.major.pad": 2., "ytick.major.pad": 2.,
    "lines.linewidth": 1., "lines.markersize": 3., "lines.markeredgewidth": .5,
    "patch.linewidth": .5, "hatch.linewidth": .5,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": .5,
    "legend.columnspacing": 1.2, "legend.labelspacing": .35, "legend.borderaxespad": .3,
    "legend.borderpad": .2,
    "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 150,
    "figure.constrained_layout.h_pad": 2. / 72, "figure.constrained_layout.w_pad": 2. / 72,
    "figure.constrained_layout.hspace": .03, "figure.constrained_layout.wspace": .03,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


# Mathtext draws sub- and superscripts at 0.7 of the body size, which puts 7 pt labels at 4.9 pt,
# under the 5 pt minimum. 0.75 keeps them at 5.25 pt. The constant is private in matplotlib, so a
# missing attribute only means the default ratio is kept.
SCRIPT_RATIO = .75


def _set_script_ratio(value):
    try:
        from matplotlib import _mathtext
    except ImportError:
        return None
    previous = getattr(_mathtext, "SHRINK_FACTOR", None)
    if previous is not None:
        _mathtext.SHRINK_FACTOR = value
    return previous


def use():
    mpl.rcParams.update(RC)
    _set_script_ratio(SCRIPT_RATIO)


@contextmanager
def style():
    previous = _set_script_ratio(SCRIPT_RATIO)
    try:
        with mpl.rc_context(RC):
            yield
    finally:
        if previous is not None:
            _set_script_ratio(previous)


def tint(color, amount):
    """Mix a colour with white; amount=1 returns the colour itself."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex([1. - amount * (1. - v) for v in (r, g, b)])


def frame(ax):
    """Close the frame of image-like panels (maps, heat maps)."""
    for spine in ax.spines.values():
        spine.set_visible(True)


def halo_label(ax, label, xy, **kwargs):
    """Draw a white outline beneath a selectable text label."""
    outline = ax.annotate(label, xy, **kwargs,
                          path_effects=[effects.Stroke(linewidth=1.6, foreground="white")])
    outline.set_in_layout(False)
    return ax.annotate(label, xy, **kwargs)


def colorbar(fig, mappable, ax, label, location="right", label_on_top=False, **kwargs):
    """Thin colour bar; a short symbol label reads better set horizontally above the bar."""
    options = dict(location=location, fraction=.05, pad=.02, aspect=22, shrink=.8)
    options.update(kwargs)
    bar = fig.colorbar(mappable, ax=ax, **options)
    bar.outline.set_linewidth(.5)
    bar.ax.tick_params(width=.5, length=2., pad=1.5)
    if label_on_top:
        bar.ax.set_title(label, loc="left", pad=3.)
    else:
        bar.set_label(label, labelpad=2.)
    if bar.solids is not None:
        bar.solids.set_rasterized(True)
    return bar


def panel_label(ax, letter):
    """Add an 8 pt bold panel label; save() aligns it with its row and column."""
    fig = ax.figure
    text = ax.text(0., 1., letter, transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=8, fontweight="bold")
    if not hasattr(fig, "_panel_labels"):
        fig._panel_labels = []
    fig._panel_labels.append((ax, text))
    return text


def _span_start(ax, attribute):
    spec = ax.get_subplotspec()
    if spec is None:
        return ("free", id(ax))
    return (id(spec.get_gridspec()), getattr(spec, attribute).start)


def finalize(fig):
    """Freeze the layout and put each panel label at the top-left of its panel's ink.

    Labels in one gridspec column share an x position and labels in one row share a baseline,
    level with any left-aligned panel heading. A panel without y tick labels (shared axis) gets
    its label just left of the frame so it never touches the heading.
    """
    labels = getattr(fig, "_panel_labels", [])
    if labels and fig.get_constrained_layout():
        # Leave room above the first row for the ascenders of the panel letters.
        top_margin = 4.5 / 72. / fig.get_figheight()
        fig.get_layout_engine().set(rect=(0., 0., 1., 1. - top_margin))
    fig.draw_without_rendering()
    if fig.get_layout_engine() is not None and not getattr(fig, "_layout_frozen", False):
        fig.set_layout_engine("none")
        fig._layout_frozen = True
    if not labels:
        return
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    left, top = {}, {}
    gap = 3. / 72. / fig.get_figwidth()
    for ax, text in labels:
        text.set_visible(False)
        x0 = to_figure.transform(ax.get_tightbbox(renderer).p0)[0]
        text.set_visible(True)
        width = text.get_window_extent(renderer).width / fig.bbox.width
        x0 = min(x0, ax.get_position().x0 - width - gap)
        column, row = _span_start(ax, "colspan"), _span_start(ax, "rowspan")
        left[column] = min(left.get(column, 1.), x0)
        top[row] = max(top.get(row, 0.), ax.get_position().y1)
    lift = mpl.rcParams["axes.titlepad"] / 72. / fig.get_figheight()
    for ax, text in labels:
        text.set_transform(fig.transFigure)
        text.set_verticalalignment("baseline")
        text.set_position((left[_span_start(ax, "colspan")], top[_span_start(ax, "rowspan")] + lift))


def save(fig, path, caption=None, vector=True):
    """Write the requested file and, by default, a vector PDF twin with embedded TrueType text.

    Metadata that changes between runs is removed so repeated builds give identical files.
    """
    path = Path(path)
    finalize(fig)
    targets = [path]
    if vector and path.suffix.lower() != ".pdf":
        targets.append(path.with_suffix(".pdf"))
    for target in targets:
        suffix = target.suffix.lower()
        if suffix == ".pdf":
            metadata = {"Creator": None, "Producer": None, "CreationDate": None}
            if caption:
                metadata["Subject"] = caption
            dpi = RASTER_DPI
        elif suffix == ".svg":
            metadata = {"Creator": None, "Date": None}
            if caption:
                metadata["Description"] = caption
            dpi = RASTER_DPI
        else:
            metadata = {"Software": None}
            if caption:
                metadata["Description"] = caption
            dpi = PNG_DPI
        fig.savefig(target, dpi=dpi, metadata=metadata)
    return targets
