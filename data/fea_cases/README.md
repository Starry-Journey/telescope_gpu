# Synthetic FEA cases

These files are generated from the public design documents, not from a solved
2.5 m FEA model.  Use them to develop/validate the software chain; replace the
CSV files with real FEA exports before claiming physical sensitivity.

- `manifest.json`: geometry, source pages, target RMS, assumptions
- `*.csv`: `x_mm,y_mm,dz_nm` on the clear aperture
- `perturb.yaml`: Zemax control DOF spec (M2 5-DOF)

Documented anchors used:
- M1: 36 bottom + 24 side supports, horizontal RMS 6.4873e-6 mm,
  vertical RMS 2.3199e-5 mm (PDF pp.113-114)
- M2: 6-point whiffletree, RMS 4.6/4.7 nm (PDF p.124)
- M3: 6-point support, 45 deg RMS 4.230 nm, vertical 1.726 nm (PDF p.131)
- Thermal: +-20 K placeholder 8 nm until real FEA thermal case is supplied
