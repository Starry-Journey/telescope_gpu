import unittest

from products.sensitivity import service as sens


class SensitivityServiceTest(unittest.TestCase):
    def test_cases(self):
        m = sens.list_cases()
        self.assertGreaterEqual(len(m["cases"]), 8)
        self.assertTrue(m["synthetic"])
        self.assertEqual(m["cases"][0]["surface"], "M1")

    def test_surface_zernike(self):
        d = sens.case_detail("M1_gravity_0deg")
        self.assertEqual(d["case"]["surface"], "M1")
        self.assertGreater(len(d["fit"]["coefficients_nm"]), 4)
        self.assertLess(d["fit"]["residual_rms_nm"], 1e-3)
        self.assertIn("Z4", d["fit"]["targets_nm"])
        self.assertEqual(len(d["grid"]["x"]), len(d["grid"]["y"]))

    def test_environment_sensitivity(self):
        s = sens.environment_sensitivity()
        self.assertEqual(s["targets"], ["Z4", "Z8", "Z9"])
        self.assertEqual(s["controls"], ["gravity_deg", "thermal_K"])
        self.assertEqual(len(s["matrix"]), 3)
        self.assertEqual(len(s["matrix"][0]), 2)
        self.assertEqual(s["rank"], 2)
        self.assertEqual(s["source"], "synthetic")

    def test_abaqus_environment_sensitivity(self):
        s = sens.environment_sensitivity(source="abaqus")
        self.assertEqual(s["source"], "abaqus")
        self.assertEqual(s["label"], "FEA-PROVISIONAL")
        self.assertEqual(len(s["matrix"]), 3)
        self.assertEqual(len(s["matrix"][0]), 2)
        self.assertIn(s["rank"], (1, 2))
        self.assertIn("gradient", s)

    def test_bad_source(self):
        with self.assertRaises(ValueError):
            sens.environment_sensitivity(source="nope")


if __name__ == "__main__":
    unittest.main()
