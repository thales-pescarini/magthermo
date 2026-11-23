# Model inputs

| Input | Quantities and conventions |
| --- | --- |
| [Thermal and field protocols](protocols.yaml) | Preparation direction, event field in specimen coordinates (T), heating and cooling rates (°C/s), hold and wait times (s), and laboratory treatment temperatures (°C). The event hold lasts $10^\ell$ seconds. P12, P24 and PX12 specify separate zero-field sequences with 12, 24 and 20 vector readings, respectively, including the initial remanence. |
| [Population parameters](mf_population.json) | For the magnetofossil population, measured length $L$ and width $W$ (nm) define the prolate approximation $d=(LW^2)^{1/3}$ and $s=\ln(L/W)$. The file specifies the joint size and shape distribution and assumes isotropic orientations within grains and in the specimen. |
| [Crystal orientation nodes](crystal_orientations.json) | Unit vectors locate the grain symmetry axis in the cubic crystal frame. Each cell has a solid angle in steradians and a normalized weight. The [whole-grain rotation rules](../portao_a/haar_product.py) independently rotate the crystal into specimen coordinates. |

The [stationary-point search settings](B2_CENSUS_POLICY.json) set the angular seed resolutions used to find minima, saddles and maxima on the unit sphere. The [census repair settings](CENSUS_REPAIR_A2.json) specify the refinements allowed for a particular set of unresolved numerical cases.
