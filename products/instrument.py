"""Single source for instrument constants used by products/GUI.

Do not duplicate 1250.0 / 0.28 / 632.8 / default defocus in new code.
"""

M1_RADIUS_MM = 1250.0
M1_OBSCURATION = 0.28
M1_DIAMETER_MM = 2500.0
M2_APERTURE_MM = 500.0
M2_OUTER_DIAMETER_MM = 610.0
# PD 离焦 diversity 集（Z4 离焦系数，单位：waves @632.8 nm 波前）。
DEFAULT_DEFOCUS = (-2.0, -1.0, 0.0, 1.0, 2.0)
# 项目参考波长（单位：nm，He-Ne）。波前单位 "waves" 均指 waves @632.8 nm。
DEFAULT_WAVELENGTH_NM = 632.8
DEFAULT_PSF_SIZE = 64
