# CalculiX simplified M1 FEA

This directory is **legal open-source FEA output**, not a synthetic Zernike map.

## Model

`products/fea/calculix_m1.py` generates a structured C3D8 mesh:

```text
annular disk: OD 2500 mm, ID 700 mm, t 260 mm
Zerodur: E=90000 MPa, nu=0.24, rho=2.53e-9 t/mm^3
36 bottom supports (6-12-12-6), fixed
24 side supports at mid-thickness, radial fixed
gravity 0 deg (x) / 90 deg (z)
```

## Run

Installed via conda-forge:

```text
conda create -y -n ccx -c conda-forge calculix
```

```bash
python products/fea/calculix_m1.py --out data/fea_cases/calculix
conda run -n ccx ccx -i data/fea_cases/calculix/m1_0deg
conda run -n ccx ccx -i data/fea_cases/calculix/m1_90deg
```

The Windows conda env was used for the committed results:

```text
CalculiX 2.23
3600 nodes, 2736 C3D8 elements
```

## Results

| case | points | dz mean | dz std | dz min | dz max |
|---|---:|---:|---:|---:|---:|
| 0 deg | 720 | ~0 | 30.7 nm | -99.9 nm | +99.9 nm |
| 90 deg | 720 | -170.4 nm | 79.4 nm | -405.0 nm | -80.2 nm |

These are **not** design-spec RMS values: the model is a plate-level
approximation with fixed support nodes. It must not be compared directly with
the 6.5/23 nm design targets until support stiffness/contact/mesh convergence
are modeled.

## Use

- `m1_0deg_top.csv`, `m1_90deg_top.csv`: upload to `POST /api/fea/upload`;
- 66-term Zernike fit is done by the Web product;
- keep `synthetic: false` but `model: simplified` in provenance.
