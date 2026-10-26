"""Finite interpolation diagnostics; unknown forward error is never set to zero."""
import math
import numpy as np
from scipy.interpolate import CubicSpline

NOISES = (.0025, .005, .01, .02)

def checked_axis(axis):
    x = np.asarray(axis, float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
        raise ValueError('INTERPOLATION_AXIS_INVALID')
    return x

def midpoints(axis):
    x = checked_axis(axis)
    return (x[:-1] + x[1:]) / 2

def cubic(axis, values, query):
    x = checked_axis(axis)
    y, q = np.asarray(values, float), np.asarray(query, float)
    if y.shape[0] != len(x) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('INTERPOLATION_VALUES_INVALID')
    if np.any(q < x[0]) or np.any(q > x[-1]):
        raise ValueError('INTERPOLATION_EXTRAPOLATION_FORBIDDEN')
    return CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=False)(q)

def error_metrics(delta):
    """First axis is the control point; remaining axes are a full vector history."""
    d = np.asarray(delta, float)
    if d.ndim != 3 or d.shape[-1] != 3 or not len(d) or not np.isfinite(d).all():
        raise ValueError('INTERPOLATION_READING_ARRAY_INVALID')
    norms = np.linalg.norm(d, axis=(1, 2))
    reads = np.linalg.norm(d, axis=2)
    i = int(np.argmax(norms))
    return dict(worst_control_index=i, maximum_absolute_L2=float(norms[i]),
        at_four_noises={str(r):dict(maximum_history_L2=float(norms[i]/(r/math.sqrt(2))),
            maximum_reading_L2=float(reads.max()/(r/math.sqrt(2))),
            per_reading_maximum_L2=(reads.max(axis=0)/(r/math.sqrt(2))).tolist())
            for r in NOISES},
        absolute_forward_error_bound=None, combined_error_bound=None, qualified=False)

def tensor_cubic(T_axis, ell_axis, values, T_query, ell_query):
    """Local tensor candidate only; the T stencil is not a full bank."""
    values = np.asarray(values, float)
    if values.ndim != 4 or values.shape[1] != len(ell_axis) or values.shape[-1] != 3:
        raise ValueError('INTERPOLATION_TENSOR_SHAPE_INVALID')
    at_T = cubic(T_axis, values, [float(T_query)])[0]
    return cubic(ell_axis, at_T, ell_query)

def duration_diagnostic(axis, base, direct_mid, direct_new_mid):
    x = checked_axis(axis)
    m = midpoints(x)
    refined = np.sort(np.r_[x, m])
    q = midpoints(refined)
    base, direct_mid, direct_new_mid = map(lambda y: np.asarray(y, float),
                                         (base, direct_mid, direct_new_mid))
    if (direct_mid.shape != (len(m),)+base.shape[1:]
            or direct_new_mid.shape != (len(q),)+base.shape[1:]):
        raise ValueError('INTERPOLATION_CONTROL_SHAPES_DIFFER')
    values = np.empty((len(refined),)+base.shape[1:], float)
    values[::2], values[1::2] = base, direct_mid
    if set(q) & set(refined):
        raise ValueError('INTERPOLATION_CONTROL_IS_A_NODE')
    return dict(candidate='CubicSpline in ell=log10(seconds), not-a-knot; diagnostic only',
        base=error_metrics(cubic(x, base, m)-direct_mid),
        refined=error_metrics(cubic(refined, values, q)-direct_new_mid),
        base_node_count=len(x), base_control_count=len(m),
        refined_node_count=len(refined), new_interval_control_count=len(q),
        refined_controls_are_new_points=True, forward_reference_uncertainty=None,
        population_representation_error_bound=None, qualified=False)
