# Population response and grid inversion

## Assemble a population curve

For each grain size, shape and internal crystal orientation, `reduce_rotations` averages the specimen-frame vector responses over whole-grain rotations. `reduce_population` combines those averages using the size and shape weights and the solid-angle weights for the internal crystal orientations. Size and shape weights express the grain volume in each cell as a fraction of the parent population volume, so a subset retains its share of the parent magnetic capacity. If a contribution is missing, `mu_known` holds the partial sum, `mu_complete` contains NaNs and the result reports the missing capacity. [Population weights](population.py) · [Crystal quadrature](quadrature.py) · [Reduction functions](integration.py)

## Fit a demagnetization curve

Supply a complete bank for one population, forward model and measurement protocol. The temperature axis is in °C and the duration axis is $\log_{10}(t/\mathrm{s})$. Both axes must be strictly increasing. For PX12, each grid point holds 20 consecutive three-component readings in specimen coordinates. Flatten each curve in measurement order, with $x,y,z$ together at every step. The bank then has shape `(n_temperatures, n_durations, 60)`. Use the same order, protocol and normalization for the observation. Supply a positive noise standard deviation in the same units, either one value or one per component.

```python
from magthermo.inference import GridEstimator
from magthermo.inference_modes import L1

estimator = GridEstimator(temperatures_C, log10_duration_s, curve_bank, sigma=sigma)
fit = estimator.fit(observed_curve, amplitude_mode=L1)
best_pairs = estimator.coordinates(fit["minimizers"][0])
```

`best_pairs` contains every temperature and log-duration pair tied for the smallest weighted sum of squared residuals. The `L1` mode fits one nonnegative multiplier to the whole predicted curve at each grid point. Predictions that differ only by a positive multiplier give the same fit. The default `fixed_scale` mode holds the multiplier at one and is appropriate when the absolute magnetization scale is known. [Estimator](inference.py) · [Estimator tests](tests/test_inference.py)

Run the estimator and integration tests from the repository root.

```sh
PYTHONPATH=src:. python -m unittest magthermo.tests.test_inference magthermo.tests.test_integration
```
