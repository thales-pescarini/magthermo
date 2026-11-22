"""Test crystal quadrature against duration recovery from synthetic MF curves at 40 °C.
Usage: python examples/mf40/reproduce.py
"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

if __package__:
    from .plot_quadrature import draw_figure
else:
    from plot_quadrature import draw_figure


HERE = Path(__file__).resolve().parent


def load_inputs(protocol):
    banks = {}
    for name, digest in protocol["curve_file_sha256"].items():
        path = HERE / "data" / name
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Curve file checksum mismatch: {name}")
        curve = np.load(path, allow_pickle=False)
        if curve.shape != (121, 2, 20, 3) or curve.dtype != np.dtype("<f8"):
            raise ValueError(f"Unexpected curve layout: {name}")
        if not np.isfinite(curve).all():
            raise ValueError(f"Missing or nonfinite curve: {name}")
        banks[name] = curve

    rng = np.random.Generator(np.random.PCG64(protocol["seed"]))
    shape = (len(protocol["noise_model_order"]), protocol["replicates_per_model"], 60)
    noise = rng.standard_normal(shape, dtype=np.float64)
    digest = sha256(noise.astype("<f8", copy=False).tobytes(order="C")).hexdigest()
    if digest != protocol["noise_sha256_little_endian_float64_c_order"]:
        raise ValueError("The regenerated MF noise differs from the reference matrix.")
    return banks["mf_c16.npy"], banks["mf_c4.npy"], noise


def fit(observations, bank, sigma):
    """Return the best duration index for each observation, using -1 for a tie."""
    estimates = np.empty(len(observations), dtype=int)
    for first in range(0, len(observations), 64):
        batch = observations[first:first + 64]
        residual = batch[:, None, :] - bank[None, :, :]
        residual /= sigma
        np.square(residual, out=residual)
        chi2 = residual.sum(axis=2, dtype=np.float64)
        best = np.argmin(chi2, axis=1)
        minimum = chi2[np.arange(len(batch)), best]
        tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1, np.abs(minimum))
        tied = (chi2 - minimum[:, None] <= tolerance[:, None]).sum(axis=1) > 1
        best[tied] = -1
        estimates[first:first + len(batch)] = best
    return estimates


def compare(reference, coarse, truth, step):
    if np.any(reference < 0) or np.any(coarse < 0):
        return None, None, None
    shift = (coarse - reference) * step
    error = (reference - truth) * step
    d_quad = np.sqrt(np.mean(shift ** 2))
    u_ref = np.sqrt(np.mean(error ** 2))
    return d_quad, u_ref, d_quad / u_ref if u_ref else None


def classify(metrics, unique_reference, step):
    if not unique_reference or any(d is None for d, u, r in metrics):
        return "inconclusive_tie"
    if any(u < step for d, u, r in metrics):
        return "inconclusive_grid"
    if all(r > 1 for d, u, r in metrics):
        return "detected"
    if all(r < 0.3 for d, u, r in metrics):
        return "small"
    return "inconclusive"


def recover(protocol, curves16, curves4, noise):
    sigma = protocol["sigma"]
    block_size = protocol["block_size"]
    grid = protocol["log10_duration_s"]
    rows = []
    for model_index, model in enumerate(protocol["curve_models"]):
        bank16 = np.ascontiguousarray(curves16[:, model_index].reshape(grid["count"], -1))
        bank4 = np.ascontiguousarray(curves4[:, model_index].reshape(grid["count"], -1))
        for truth, curve in enumerate(bank16):
            control = fit(curve[None, :], bank16, sigma)[0]
            control4 = fit(curve[None, :], bank4, sigma)[0]
            if control >= 0 and control != truth:
                raise ValueError("The noiseless C16 curve did not recover its duration.")
            observations = curve + sigma * noise[model_index]
            reference = fit(observations, bank16, sigma)
            coarse = fit(observations, bank4, sigma)
            sections = [("full", slice(None))]
            for first in range(0, len(reference), block_size):
                sections.append((f"block{first // block_size + 1}", slice(first, first + block_size)))

            row = {
                "model": model,
                "true_log10_duration_s": grid["start"] + truth * grid["step"],
                "noiseless_C16_log10_s": grid["start"] + control * grid["step"] if control >= 0 else None,
                "noiseless_C4_log10_s": grid["start"] + control4 * grid["step"] if control4 >= 0 else None,
                "noisy_C16_ties": int(np.sum(reference < 0)),
                "noisy_C4_ties": int(np.sum(coarse < 0)),
            }
            metrics = []
            for name, index in sections:
                if control < 0:
                    values = (None, None, None)
                else:
                    values = compare(reference[index], coarse[index], truth, grid["step"])
                metrics.append(values)
                for quantity, value in zip(("D_quad", "U_ref", "ratio"), values):
                    row[f"{name}_{quantity}"] = value
            row["classification"] = classify(metrics, control >= 0, grid["step"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE, help="Directory for results.csv, quadrature.png and quadrature.pdf")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    curves16, curves4, noise = load_inputs(protocol)
    rows = recover(protocol, curves16, curves4, noise)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draw_figure(rows, protocol, args.output / "quadrature.png")
    groups = {model: [r for r in rows if r["model"] == model] for model in protocol["curve_models"]}
    groups["total"] = rows
    for label, subset in groups.items():
        counts = Counter(r["classification"].split("_")[0] for r in subset)
        print(f"MF/{label}: {len(subset)} cases, {counts['detected']} detected, "
              f"{counts['inconclusive']} inconclusive, {counts['small']} small")


if __name__ == "__main__":
    main()
