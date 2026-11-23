# Material properties and grain geometry

The single-particle solver calls [`s5_model.material`](STAGE6_E1_WP0_WP1_2026-10-22/legacy_discrete/src/s5_model.py) for the temperature-dependent cubic anisotropy constants and saturation magnetization. The bundled SDCC function `demag_factors` supplies the principal demagnetizing factors. [`PointModel`](../src/e1_core.py) uses these factors and the direction of the spheroid's symmetry axis to express the tensor in crystal coordinates.

The [SDCC files](STAGE6_E1_WP0_WP1_2026-10-22/legacy_discrete/upstream/src/sdcc) are byte-identical to the files at upstream tag `v0.4.3`. The adjacent [model routines](STAGE6_E1_WP0_WP1_2026-10-22/legacy_discrete/src) and [configuration](STAGE6_E1_WP0_WP1_2026-10-22/legacy_discrete/config/core.yaml) belong to MAGTHERMO. [Source revisions and file hashes](../SOURCES.md)
