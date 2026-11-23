# Software and data sources

## Single Domain Comprehensive Calculator

Brendan Cych is the author named in the SDCC v0.4.3 package metadata. Brendan Cych, Greig A. Paterson, Lesleis Nagy and Wyn Williams (2025) describe the software in *First principles understanding of single domain magnetizations, Part I: the single domain comprehensive calculator (SDCC) open source library*, Geophysical Journal International. [Paper](https://doi.org/10.1093/gji/ggaf149) · [SDCC source at the revision used](https://github.com/bcych/SDCC/tree/ae8f0338b8bcf9f99588fd980feb7a84b9856ae1)

The forward calculation uses the SDCC temperature-dependent magnetization and cubic anisotropy constants, ellipsoidal demagnetizing factors and energy convention. MAGTHERMO calls the SDCC functions `get_material_parms` and `demag_factors`. The upstream Python files listed below were verified byte for byte against the Git objects at tag `v0.4.3`, commit `ae8f0338b8bcf9f99588fd980feb7a84b9856ae1`. The distribution contains [the verified upstream Python modules](portable_runtime/upstream/src/sdcc). MAGTHERMO implements the population integration, thermal histories and recovery analysis.

| File in SDCC v0.4.3 | Verified SHA-256 |
| --- | --- |
| `src/sdcc/energy.py` | `353101e74bf5780e849b5ad383996f8ef001716d0d1e22e40d4866d2d28bddab` |
| `src/sdcc/materials.py` | `24579c5910ff724e6bda6d7909677500a9c6226d5fc1d23bba949c9291668d18` |
| `src/sdcc/__init__.py` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `LICENSE` | `7abe19ec9bb73b36141b999b861d24ad855e808bafe0f81e84cce28556f6c297` |

The SDCC v0.4.3 README grants Creative Commons Attribution-ShareAlike 4.0 International, and the LICENSE distributed at that revision contains the complete terms. The package metadata has a conflicting MIT classifier. The license file here reproduces the text from the tagged source. It applies to the SDCC material. [License text](LICENSES/SDCC-CC-BY-SA-4.0.txt) · [Upstream README](https://github.com/bcych/SDCC/blob/ae8f0338b8bcf9f99588fd980feb7a84b9856ae1/readme.md) · [Upstream package metadata](https://github.com/bcych/SDCC/blob/ae8f0338b8bcf9f99588fd980feb7a84b9856ae1/pyproject.toml)

## Population measurements and single-domain size limits

To recompute population weights from the original single-domain size workbook, install `xlrd==2.0.2` and set `MAGTHERMO_SD_TABLE` to the local path of `TM_Crit_Sizes.xls`. The [table reader](src/environment.py) also accepts a JSON export with the same sheets. The model figures and the tests on saved curves use bundled inputs.

The MF population uses the grains classified as isotropic in sheet `7-43` of `KR1302-PC6_size.xlsx`. The axisymmetric shape model interprets two-dimensional measurements as a size and shape proxy. The example distributes synthetic demagnetization curves calculated for part of this population. Hekla and Guia are separate populations studied in the wider project.

| Source and use | Original file | Verified SHA-256 |
| --- | --- | --- |
| Usui and Yamazaki (2021), MF morphology. [Data](https://doi.org/10.5281/zenodo.4579925) · [Paper](https://doi.org/10.1029/2021GC009770) | `KR1302-PC6_size.xlsx` | `a2202c7134b0df4aab4436694eedb4f8fd9d053ce1d807dcd799be3e408e4259` |
| Usui and Yamazaki (2021), additional morphology workbook consulted in population development. [Data](https://doi.org/10.5281/zenodo.4579925) | `Site777_size.xlsx` | `430e3b5f0b781f5bc937c8d41070771f0a664807029a3964170d9cb4dc7f8ca2` |
| Cych (2024), single-domain size limits for TM00. [Data](https://doi.org/10.5281/zenodo.10471806) · [Cych et al. paper](https://doi.org/10.1029/2024JB028805) | `TM_Crit_Sizes.xls` | `3709fa5b358cb80a18893a7fdc140eae7975e17fa2355c3f9e26ee685438ca5f` |
| Gergov, Muxworthy, Williams and Cowan (2024 dataset, 2025 paper), Hekla grain morphology. [Data](https://doi.org/10.5281/zenodo.11369780) · [Paper](https://doi.org/10.1029/2024GC011776) | `HeklaGrainMetrics.csv` | `6e3d16d7faf03385ebc836bdf84fcaaa96bfcce942822254b020632bb1c0bf30` |
| Donardelli Bellon and colleagues (2024), Guia dataset archive. [Data](https://doi.org/10.5281/zenodo.10837481) · [Paper](https://doi.org/10.1038/s43247-024-01715-x) | `Dataset.zip` (entire archive) | `eb3b6224e085f4d2e307359d75d2e98e589ba58d0d1c15e69682cce9d1346bad` |
| Donardelli Bellon and colleagues (2024), shape-parameter compilation, sheet `Magnetite`. [Data](https://doi.org/10.5281/zenodo.10837481) | `Magnetic data/Hysteresis and Backfield/Compilation_magnetic_shape_parameters.xlsx` (uncompressed archive member) | `04b662e3be1d4c94b7cc976736be248d2b6a7ed0a8fd4ee6bd70179e0ccd90d4` |

The SHA-256 values were calculated from the original file bytes. The complete Guia archive is 1,447,549,251 bytes. Its MD5 matches the value published on Zenodo (`1c83e7b1e962de7025e2f74a742e3c98`). The extracted workbook is 50,929 bytes and its SHA-256 also matches the copy used for the population analysis.

## Article files

The first two hashes identify the article copies archived with the original model references. The last identifies the local copy of the Guia paper. Publisher downloads obtained on different dates can have different bytes. The DOI identifies the scientific work.

| Article DOI | Local article filename | Verified SHA-256 |
| --- | --- | --- |
| [10.1093/gji/ggaf149](https://doi.org/10.1093/gji/ggaf149) | `ggaf149 (1).pdf` | `c3d628c968393afb2ff4667369ccd74b3c479d0eccd5352974f6b215931f027f` |
| [10.1029/2024JB028805](https://doi.org/10.1029/2024JB028805) | `JGR Solid Earth - 2024 - Cych - Magnetic Domain States and Critical Sizes in the Titanomagnetite Series.pdf` | `37c3e6bce255422db9aa4b62a1f61b22bcda2a86b0c50001c06f4710ac0e75ff` |
| [10.1038/s43247-024-01715-x](https://doi.org/10.1038/s43247-024-01715-x) | `s43247-024-01715-x.pdf` | `948819b8749f2c963933a700f8777b29bde7c9104474629154c1a365bc71ec37` |
