"""Exact selected Gate 1 / CONN functions; identities in provenance/GATE1_IDENTITY.json.
No historical top-level module or historical driver is imported.
"""
import numpy as np
import math

def hess_amb(pm, m):
    """Ambient Hessian of the R = I energy (analytic; verified against finite differences in the run)."""
    k1, k2, D, _h, _R = pm.np_arrays()
    s = m * m
    H = np.zeros((3, 3))
    for i in range(3):
        j, k = [t for t in range(3) if t != i]
        H[i, i] = 2.0 * (k1 * (s[j] + s[k]) + k2 * s[j] * s[k])
        for kk in range(3):
            if kk != i:
                l = 3 - i - kk
                H[i, kk] = 4.0 * m[i] * m[kk] * (k1 + k2 * s[l])
    return H + 2.0 * D

def tangent_basis(m):
    a = np.array([1.0, 0, 0]) if abs(m[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ m) * m
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(m, e1)

def riem(pm, m):
    g = pm.grad_np(m)
    e1, e2 = tangent_basis(m)
    B = np.stack([e1, e2], axis=1)
    gt = B.T @ g
    Hr = B.T @ (hess_amb(pm, m) - (m @ g) * np.eye(3)) @ B
    return gt, Hr, B

def cube_sphere_seeds(n):
    u = (np.arange(n) + 0.5) / n * 2 - 1
    U, Vv = np.meshgrid(u, u)
    pts = []
    for ax in range(3):
        for sg in (1.0, -1.0):
            P = np.zeros((n * n, 3))
            P[:, ax] = sg
            o = [t for t in range(3) if t != ax]
            P[:, o[0]] = U.ravel()
            P[:, o[1]] = Vv.ravel()
            pts.append(P)
    P = np.concatenate(pts)
    return P / np.linalg.norm(P, axis=1, keepdims=True)

def newton(pm, m, scale, iters=80):
    for _ in range(iters):
        gt, Hr, B = riem(pm, m)
        if np.linalg.norm(gt) <= 1e-9 * scale:
            return m, True
        try:
            step = -np.linalg.solve(Hr, gt)
        except np.linalg.LinAlgError:
            return m, False
        n = np.linalg.norm(step)
        if n > 0.3:
            step *= 0.3 / n
        v = B @ step
        nv = np.linalg.norm(v)
        m = math.cos(nv) * m + (math.sin(nv) / nv) * v if nv > 0 else m
        m /= np.linalg.norm(m)
    gt, _, _ = riem(pm, m)
    return m, bool(np.linalg.norm(gt) <= 1e-9 * scale)

def census(pm, n_seed):
    scale = abs(pm.np_arrays()[0]) + max(np.diag(pm.np_arrays()[2]))
    pts = []
    n_fail = 0
    for s in cube_sphere_seeds(n_seed):
        m, ok = newton(pm, s.copy(), scale)
        if not ok:
            n_fail += 1
            continue
        if all(math.acos(max(-1.0, min(1.0, float(m @ p["m"])))) > 1e-6 for p in pts):
            gt, Hr, _ = riem(pm, m)
            ev = np.linalg.eigvalsh(Hr)
            morse = bool(np.all(np.abs(ev) > 1e-8 * scale))
            pts.append({"m": m, "E": float(pm.energy_np(m)), "eig": ev.tolist(),
                        "index": int(np.sum(ev < 0)), "morse": morse})
    counts = {i: sum(1 for p in pts if p["index"] == i) for i in (0, 1, 2)}
    return {"points": pts, "counts": counts, "euler": counts[0] - counts[1] + counts[2],
            "all_morse": all(p["morse"] for p in pts), "newton_failures": n_fail, "seeds": 6 * n_seed * n_seed}

def same_census(a, b):
    if a["counts"] != b["counts"]:
        return False
    for p in a["points"]:
        if not any(p["index"] == q["index"] and math.acos(max(-1, min(1, float(p["m"] @ q["m"])))) < 1e-6
                   for q in b["points"]):
            return False
    return True

def flow_rhs(model, m, escale=1.0):
    """The declared field, evaluated at the NORMALISED state.

    Projecting with an unnormalised m is not a projection: the radial component then obeys
    d|m|^2/ds = 2 (m.grad e)(|m|^2 - 1)/escale and the deviation grows exponentially, which is a numerical artefact
    of the writing, not of the flow. Normalising inside the right-hand side keeps |m| exactly invariant. The state is
    NOT renormalised between steps (that would hide the error instead of measuring it); the norm deviation is
    reported as a diagnostic.
    """
    mu = np.asarray(m, dtype=float)
    mu = mu / np.linalg.norm(mu)
    g = model.grad_np(mu) / escale
    return -(g - (mu @ g) * mu)

def integrate_branch(model, m0, t_max=200.0, rtol=1e-11, atol=1e-13, escale=1.0,
                     stop_grad=1e-9, max_steps=4000):
    """Integrate one branch from m0. Returns the trajectory and the diagnostics the command requires:
    norm deviation, energy dissipation, tolerances, energy scale, stopping criterion.

    The trajectory is a CANDIDATE producer. No certificate reads its numbers.
    """
    from scipy.integrate import solve_ivp

    def rhs(_s, y):
        return flow_rhs(model, y, escale)

    def event_flat(_s, y):
        yn = y / np.linalg.norm(y)
        g = model.grad_np(yn) / escale
        gt = g - (yn @ g) * yn
        return float(np.linalg.norm(gt)) - stop_grad

    event_flat.terminal = True
    event_flat.direction = -1
    sol = solve_ivp(rhs, (0.0, t_max), np.asarray(m0, dtype=float), method="DOP853",
                    rtol=rtol, atol=atol, dense_output=False, events=event_flat,
                    max_step=t_max / 50.0)
    Y = sol.y.T
    norms = np.linalg.norm(Y, axis=1)
    E = np.array([model.energy_np(y / np.linalg.norm(y)) for y in Y])
    dE = np.diff(E)
    g_end = model.grad_np(Y[-1] / np.linalg.norm(Y[-1]))
    gt_end = g_end - (Y[-1] @ g_end) * Y[-1] / (norms[-1] ** 2)
    return {"m_end": (Y[-1] / norms[-1]).tolist(), "n_steps": int(len(Y)),
            "s_end": float(sol.t[-1]), "stopped_by_event": bool(sol.status == 1),
            "max_norm_deviation": float(np.max(np.abs(norms - 1.0))),
            "energy_start": float(E[0]), "energy_end": float(E[-1]),
            "energy_max_increase": float(np.max(dE)) if len(dE) else 0.0,
            "energy_monotone_decreasing": bool(len(dE) == 0 or np.max(dE) <= 1e-9 * max(1.0, abs(E[0]))),
            "tangential_gradient_end": float(np.linalg.norm(gt_end)),
            "rtol": rtol, "atol": atol, "escale_used": escale, "t_max": t_max,
            "stop_criterion": f"|P_T grad e|/escale < {stop_grad} (terminal event) or s = {t_max}",
            "trajectory": (Y / norms[:, None]).tolist()}
