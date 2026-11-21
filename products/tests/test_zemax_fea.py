import unittest

import numpy as np

from products.zemax.zmx_parser import parse_zmx_bytes
from products.sensitivity import service as sens


class ZemaxFeaTest(unittest.TestCase):
    def test_zmx_utf16(self):
        text = "VERS 1\nENPD 2500\nWAVM 1 0.6328 1\nSURF 0\nSURF 1\nSURF 2\n  STOP\n  CURV -1e-4\n  CONI -1.0\n"
        raw = text.encode("utf-16")
        m = parse_zmx_bytes(raw)
        self.assertEqual(m["aperture_mm"], 2500.0)
        self.assertEqual(m["stop"], 2)
        self.assertEqual(len(m["surfaces"]), 3)

    def test_custom_surface_66(self):
        x, y = np.meshgrid(np.linspace(-100, 100, 21), np.linspace(-100, 100, 21))
        r = np.sqrt(x * x + y * y) / 100.0
        z = 5.0 * (2 * r ** 2 - 1) + 0.5 * x / 100.0
        d = sens.custom_surface(x.ravel(), y.ravel(), z.ravel(), 100.0)
        self.assertEqual(len(d["fit"]["coefficients_nm"]), 66)
        self.assertEqual(d["fit"]["display_terms"], 15)
        self.assertFalse(d["case"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
