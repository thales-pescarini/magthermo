# MAGTHERMO

Under what conditions can a vector demagnetization curve from a characterized population of single-domain grains constrain the temperature and duration of a thermal event?

MAGTHERMO uses the single-domain energy-landscape and master-equation approach of the SDCC (Cych et al., 2025, and references therein), with its material laws and demagnetizing factors. Its own components are a stationary-point census for arbitrary internal crystal orientation and continuous shape, transport of occupations through state mergers and splits, integration over measured grain populations and joint temperature–duration recovery.

A short, hot event and a longer, cooler event can leave similar remanent intensities. MAGTHERMO tests whether the direction of the surviving remanence and its decay during laboratory demagnetization can distinguish such thermal histories. It predicts the three components of specimen remanence after each laboratory step and compares the predicted sequence with measurements.

## Single-particle energy landscape

Each grain is a uniformly magnetized spheroid. For a unit moment direction $\mathbf u$ in crystal coordinates, cubic magnetocrystalline anisotropy, shape anisotropy and the applied field contribute to the energy density.

$$
e(\mathbf u) =
K_1(u_x^2u_y^2+u_y^2u_z^2+u_z^2u_x^2)
+ K_2 u_x^2u_y^2u_z^2
+ \frac{\mu_0 M_s^2}{2}\mathbf u^{\mathsf T}N\mathbf u
- M_s\,\mathbf B\cdot\mathbf u .
$$

The shape and orientation of the spheroid determine its demagnetizing tensor $N$. Temperature sets $K_1$, $K_2$ and the saturation magnetization $M_s$. The solver locates minima on the sphere of moment directions and follows both downhill paths from each first-order saddle to identify the states it connects. For a grain of volume $V$, escape from minimum $i$ through saddle $s$ requires the energy $V(e_s-e_i)$. The corresponding transition rate is proportional to $\exp[-V(e_s-e_i)/(k_B T_{\mathrm{abs}})]$, with absolute temperature in kelvin. A higher temperature increases the escape probability per unit time for a given barrier. A longer hold allows transitions to accumulate. [Single-grain calculations and units](src/README.md)

![Circular energy landscape in crystal coordinates and energy profiles from a common saddle to two minima](examples/model/landscape.png)

**Single-domain magnetite grain.** The oblate particle has a volume-equivalent diameter of 45 nm, a polar-to-equatorial diameter ratio of 0.835 and a short axis along [531]. At 80 °C in a 50 µT field along [100], minima A and B share saddle S. Panel (a) shows the lower hemisphere in equal-area projection. Dashed white lines trace the projected edges of the cubic unit cell. Panel (b) shows why the escape barrier depends on which minimum the moment occupies.

## Forward calculation

1. **Prepare the grain and apply the thermal event.** Grain size, shape and orientation define the particle. The protocol specifies how to prepare its initial remanence, the applied field, heating and cooling rates, peak temperature and hold time. The solver evolves the probability of occupying each minimum through heating, the isothermal hold and cooling. It transfers occupations when states merge or split under a supported transition rule.
2. **Simulate the measurement.** After the field is removed, the model records the initial remanence. It carries the occupations through successive zero-field laboratory heating and cooling steps. At each return to the measurement temperature, the grain moment is the occupation-weighted sum of the moments in its stable directions. Model F retains the full relaxation spectrum. N1 approximates the response with one relaxation time selected for its contribution to the magnetic signal.
3. **Sum the specimen response.** The model rotates each grain moment into specimen coordinates and sums the contributions of noninteracting grains over size, shape, internal crystal orientation and whole-grain rotation. Repeating the calculation over event temperatures $T$ and log durations $\ell=\log_{10}(t/\mathrm{s})$ gives a grid with one predicted vector curve at each node. [Thermal protocols](config/README.md) · [Forward operator](src/general_engine.py) · [Population integration](portao_a/integration.py)

## Joint inversion

The estimator takes a complete grid of predicted curves, an observed curve from the same protocol and the standard deviation of each measured component. For PX12, each curve contains the initial remanence and 19 subsequent readings, with $x$, $y$ and $z$ retained at each step. At each candidate $(T,\ell)$, the primary analysis (mode `L1`) minimizes the weighted squared residuals over one nonnegative amplitude $a$ shared by the entire curve.

$$
\chi^2(T,\ell)=\min_{a\geq 0}\sum_j
\left[\frac{y_j-a\,m_j(T,\ell)}{\sigma_j}\right]^2 .
$$

Here $j$ indexes both measurement step and vector component, $y_j$ is the observation and $m_j$ the prediction. If two thermal histories produce proportional curves with the same polarity, a fitted amplitude makes them indistinguishable. Changes in direction or relative decay across steps can separate them. The estimator returns all grid points tied for the minimum. A fixed-scale fit is available when the absolute amplitude is known. [Estimator inputs and usage](portao_a/README.md)

The grid search and its handling of ambiguity are implemented and tested. The accuracy of joint temperature and duration recovery on the refined population model and on rock specimens remains under evaluation.

## Installation and tests

Use CPython 3.14.5. From the repository root, create a virtual environment, install the pinned dependencies and run both test suites.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src:. python -m unittest discover -s portao_a/tests
PYTHONPATH=src:. python -m unittest discover -s tests
```

[Model figures and numerical tests](examples/README.md) · [Data sources and checksums](SOURCES.md)

## Attribution

MAGTHERMO uses the material laws and demagnetizing factors in Brendan Cych's Single Domain Comprehensive Calculator v0.4.3. The accompanying paper is by Cych, Paterson, Nagy and Williams (2025). Three upstream Python files are included under CC BY-SA 4.0. The original MAGTHERMO code has no declared license. [SDCC paper](https://doi.org/10.1093/gji/ggaf149) · [Bundled implementation](portable_runtime/README.md) · [License scope and text](LICENSES/README.md)
