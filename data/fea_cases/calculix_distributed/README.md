# CalculiX distributed-spring M1 cases

This is a **simplified plate-level** FEA, not the full 2.5 m telescope.

## Model

`products/fea/calculix_m1.py --support distributed`:

- annular Zerodur disk OD 2500 / ID 700 / t 260 mm;
- 3600 nodes, 2736 C3D8 elements;
- **every bottom node** has a z SPRING1 (elastic foundation);
- **every outer-ring node** has global x/y SPRING1 (side support);
- total stiffness `k=1e9`, `k_side=1e8` N/mm, divided per node;
- gravity 0°/90°.

## Results

| case | raw dz std | plane-removed std |
|---|---:|---:|
| 0° | 4.23 nm | 3.23 nm |
| 90° | 8.90 nm | 8.90 nm |

Mesh 28/48/7 (9408 nodes):

| case | raw dz std | plane-removed std |
|---|---:|---:|
| 0° | 4.08 nm | 2.82 nm |
| 90° | 8.93 nm | 8.93 nm |

The two meshes differ by <0.5 nm, much better than point springs (30.7 → 53.5 nm).
This is an **elastic-foundation approximation**, not a real whiffletree/contact.

## Comparison

Design anchors: 6.4873 nm (horizontal), 23.199 nm (vertical).
The simplified model is below both, but the support is distributed springs, not
an actual 36-point whiffletree. Do not claim the design is verified.

## Use

Upload `m1_*_top.csv` to `/api/fea/upload` or add them as manifest cases.
