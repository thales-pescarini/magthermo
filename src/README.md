# Single-grain calculations

`PointModel` evaluates the energy density of a uniformly magnetized spheroid at a fixed temperature and field. Its argument `s` is the natural logarithm of the polar-to-equatorial diameter ratio. The polar axis `c` and the unit magnetization direction passed to `energy_np` are in crystal coordinates. The field `B` is also in crystal coordinates, in tesla. Temperature `T` is in °C. If the rotation matrix $R$ takes crystal vectors into specimen coordinates, the model field is $R^{\mathsf T}\mathbf B_{\mathrm{specimen}}$.

## Evaluate an energy difference

Run this from the repository root after installing the [dependencies](../requirements.txt). The default material is magnetite. Reversing the moment leaves the cubic and shape terms unchanged, so the difference below isolates the Zeeman term, $2M_s |\mathbf B|$.

```sh
PYTHONPATH=src:. python - <<'PY'
import numpy as np
from e1_core import PointModel

grain = PointModel(s=-0.18, c=[5, 3, 1], T=80.0, B=[50e-6, 0.0, 0.0])
along_field = np.array([1.0, 0.0, 0.0])
difference = grain.energy_np(-along_field) - grain.energy_np(along_field)
print(f"{difference:.2f} J/m³")
PY
```

The result is **46.68 J/m³**. Grain size enters when converting an energy-density difference into a barrier in joules. For volume-equivalent diameter $d$, the grain volume is $\pi d^3/6$. [Energy and transition rates](e1_core.py)

## From energy density to a demagnetization curve

`landscape` locates stationary directions and follows downhill paths from a saddle to determine which minima exchange probability. At a fixed temperature and field, `propagate` advances the occupations over a hold using the directed barriers, grain diameter in nanometres and elapsed time in seconds. [Stationary directions and propagation](e1_core.py)

`GeneralTape.run` assembles preparation, heating in the event field, the isothermal hold, cooling and the sequential zero-field laboratory steps in chronological order. `prepare_response` builds the operators for one diameter and specimen rotation. `evaluate_duration` substitutes a hold of $10^\ell$ seconds and returns the predicted vector at each reading. [Forward calculation](general_engine.py) · [Thermal protocols](../config/protocols.yaml)
