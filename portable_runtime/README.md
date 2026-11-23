# Material properties and grain geometry

The single-particle solver calls [`material_model.material`](src/material_model.py) for the temperature-dependent cubic anisotropy constants and saturation magnetization. The bundled SDCC function `demag_factors` supplies the principal demagnetizing factors. [`PointModel`](../src/grain_energy.py) uses these factors and the direction of the spheroid's symmetry axis to express the tensor in crystal coordinates.

The [SDCC files](upstream/src/sdcc) are byte-identical to the files at upstream tag `v0.4.3`. The adjacent [model routines](src) and [configuration](config/core.yaml) belong to MAGTHERMO. [Source revisions and file hashes](../SOURCES.md)
