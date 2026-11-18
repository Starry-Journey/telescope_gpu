from products.optics.forward import forward_psf_numpy
from products.optics.zernike import TERMS
p = forward_psf_numpy([0.1, 0.02, -0.01], 0.5, n=32)
print("psf", p.shape, round(float(p.sum()), 6), "terms", len(TERMS))
