# Inputs for the model figures

From the repository root, after installing the [requirements](../../requirements.txt), run

```sh
PYTHONPATH=src:. python examples/model/figures.py
```

The script draws five PNG files in this directory. The population and forward panels read saved numerical outputs. The energy panels evaluate the grain model at fixed temperature and field. The inverse panels use the public grid estimator on an analytic signal.

## Grain and energy landscape

[Grain states](grain_states.json) gives the ten stationary directions, their energies in J m⁻³, their curvature indices, and the two minima reached by descent from each saddle. A connection refers to zero-based indices in the points array. For the illustrated saddle, the file also includes both saved downhill trajectories at relative tolerance 10⁻¹¹. The two seed resolutions of the archived census agreed. The figure code checks the energies, signs of both tangent curvatures, stationarity against the original solver tolerance, and the trajectory endpoints. [Grain and landscape plots](grain_landscape.py)

The magnetite grain has a polar-to-equatorial axis ratio of exp(−0.18), an axis of symmetry parallel to [531], and a 50 µT field along [100] at 80 °C. Figure 1 shows a central section through the short axis and field. The plotted energy differences use [100] as zero for each term. Figure 2 uses the lowest minimum as zero for the entire sphere. Longitude is measured from [100] toward [010] and latitude from the (001) plane toward [001]. The 45 nm volume-equivalent diameter sets both the dimensions of the cross-section and the volume used to convert a saddle height into a thermal barrier. The two states of the first saved connection are labeled A and B.

## Population and forward curves

The [population weights](population_weights.csv) sum the modeled volume weights of the retained single-domain nodes in bins of length/width and volume-equivalent diameter. Each weight is a fraction of the parent population capacity. The bins sum to 0.8087943853. The window plotted contains 99.57% of that retained weight. The C4 panel is a stereographic projection centered on [100]. Each dot is an integration node and its label is the cell's share of the solid angle of the cubic symmetry sector. Internal crystal and external specimen orientations are both assumed isotropic. [Orientation nodes](../../config/crystal_orientations.json) · [Population parameters](../../config/mf_population.json)

The [forward grid](forward_grid.csv) contains the initial specimen-z component for model F at every temperature from 40 to 300 °C in steps of 20 °C and every log duration from 0 to 15 in steps of 0.125. The [two selected curves](forward_curves.csv) include all 20 readings and all three specimen components at 160 °C for holds of 10³ and 10¹² seconds. The first row of each curve is the initial remanence (NRM) before laboratory demagnetization and has no laboratory plateau. All values are divided by the magnetic capacity of the full parent population. Figure 4 shows x and z. The largest absolute y component of either curve is 1.13 × 10⁻⁵ or less in the same units. [Population and forward plots](population_response.py) · [Laboratory sequence](../../config/protocols.yaml)

## Joint estimator

The [inversion plot](inverse_grid.py) uses the analytic template from the [estimator tests](../../portao_a/tests/test_inference.py). With event temperature in °C and log duration in log₁₀ seconds, the first two components are T/300 and ℓ/15. The other 58 are zero. The input is the noiseless template at T = 160 °C and ℓ = 6. Each component has weight 1/(0.04)² in the residual sum. Every pair on ℓ = 6T/160 gives a proportional template. Fitting one amplitude gives 14 tied grid minima. Fixing it to one preserves both component magnitudes and gives one minimum. Color shows the increase in weighted squared residuals relative to the best fit. The panels illustrate the effect of fitting an amplitude on parameter identifiability.

## Numerical provenance

These SHA-256 digests identify the bytes of the original numerical files used to extract the plotted quantities. The CSV and JSON inputs in this directory suffice to regenerate the figures.

| Archived output | Original file | SHA-256 |
| --- | --- | --- |
| Single-grain census and connections | `136cf7d3bd18cd69c052c902b45630f263cf62ca140cbb858610a516dcee8c7c.json` | `5e4c3f79d382cb8306d814cbbe2c6ba0c5f5cc15bf837d1baca23aa9e707c556` |
| Population quadrature weights | `WEIGHTS.npz` | `7c9136a1af1ec98c3e82b6f61b9aa46d00d08d54dd4e2a36f7452c253c7fa970` |
| Grid design and size coordinates | `DESIGN.json` | `0806124d851579598bdea057e761a065d7ca8111513938bcb0f649ddc3bc748c` |
| Synthetic F and N1 forward curves | `BANK.f31ea6ba0f512296ca576ac7.npz` | `68d6ea0e52f0da4e24da2e8a68f368ed5c447c7feeb71e5b5c3002fcdd05e250` |
