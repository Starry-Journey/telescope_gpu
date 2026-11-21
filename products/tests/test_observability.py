"""Tests for the control-space observability layer and astigmatism plan."""
import math
import unittest

import numpy as np

from products.sensitivity import observability as obs
from products.sensitivity import service as sens


class AnalyzeControlSpaceTest(unittest.TestCase):
    """Relative threshold -> rows/columns dropped, reduced S stays correct."""

    def setUp(self):
        self.targets = ["Z4", "Z5", "Z6", "Z8", "Z9", "Z11"]
        self.dofs = ["focus", "decenter_x", "warp"]
        # focus -> Z4/Z11, decenter_x -> Z8, warp -> weak Z5/Z6 only.
        # Z9 is blind to every DOF.  The warp column (max 2e-3) is below
        # 5% of the global max (1.0), exactly the "translation stage is
        # nearly blind to astigmatism" situation.
        self.S = np.array([
            [1.0, 0.0, 0.0],       # Z4
            [0.0, 0.0, 1.0e-3],    # Z5
            [0.0, 0.0, 2.0e-3],    # Z6
            [0.0, 0.6, 0.01],      # Z8
            [0.0, 0.0, 0.0],       # Z9
            [0.8, 0.0, 0.0],       # Z11
        ])
        self.result = obs.analyze_control_space(
            self.S, self.targets, self.dofs)

    def test_per_dof_and_per_target_max(self):
        self.assertEqual(self.result["per_dof_max"],
                         {"focus": 1.0, "decenter_x": 0.6, "warp": 0.01})
        self.assertEqual(self.result["per_target_max"]["Z4"], 1.0)
        self.assertAlmostEqual(self.result["per_target_max"]["Z8"], 0.6)
        self.assertEqual(self.result["per_target_max"]["Z9"], 0.0)
        self.assertEqual(self.result["per_target"]["Z5"]["max_dof"], "warp")
        self.assertFalse(self.result["per_dof"]["warp"]["observable"])

    def test_blind_row_is_dropped(self):
        # Z9 has no sensitivity at all and Z5/Z6 only through the warp
        # column (below the relative threshold) -> all three rows dropped.
        self.assertEqual(self.result["dropped_rows"], ["Z5", "Z6", "Z9"])
        self.assertEqual(self.result["dropped_cols"], ["warp"])
        self.assertEqual(self.result["observable_dofs"], ["focus", "decenter_x"])
        self.assertEqual(self.result["observable_targets"],
                         ["Z4", "Z8", "Z11"])
        self.assertTrue(self.result["rationale"])

    def test_reduced_matrix_dimensions_and_values(self):
        reduced = self.result["reduced"]
        self.assertEqual(reduced["shape"], [3, 2])
        self.assertEqual(reduced["targets"], ["Z4", "Z8", "Z11"])
        self.assertEqual(reduced["dofs"], ["focus", "decenter_x"])
        np.testing.assert_allclose(
            np.asarray(reduced["matrix"]),
            self.S[np.ix_([0, 3, 5], [0, 1])],
        )
        # full matrix is rank 3; the reduced one is a clean 3x2 rank-2 map.
        self.assertEqual(self.result["rank"], 3)
        self.assertEqual(self.result["reduced_rank"], 2)
        self.assertAlmostEqual(self.result["reduced_condition"],
                               math.sqrt(1.64) / 0.6)

    def test_reduced_matrix_reproduces_kept_targets(self):
        reduced = np.asarray(self.result["reduced_matrix"])
        kept_rows = np.asarray(self.result["kept_rows"])
        kept_cols = np.asarray(self.result["kept_cols"])
        u = np.array([0.3, -0.2])                 # reduced-DOF command
        u_full = np.zeros(len(self.dofs))
        u_full[kept_cols] = u
        # The reduced matrix must be the kept rows of S with the dropped DOFs
        # zeroed, i.e. no dropped row/column leaks into the compensation map.
        np.testing.assert_allclose(
            reduced @ u,
            self.S[np.ix_(kept_rows, range(self.S.shape[1]))] @ u_full,
            atol=1e-15,
        )

    def test_all_zero_matrix_has_no_observable_direction(self):
        r = obs.analyze_control_space(np.zeros((2, 2)),
                                      ["Z5", "Z6"], ["translation_x", "tilt"])
        self.assertEqual(r["dropped_rows"], ["Z5", "Z6"])
        self.assertEqual(r["dropped_cols"], ["translation_x", "tilt"])
        self.assertEqual(r["reduced_shape"], [0, 0])
        self.assertEqual(r["reduced_rank"], 0)
        self.assertEqual(r["reduced_condition"], float("inf"))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            obs.analyze_control_space(self.S, ["Z4"], self.dofs)
        with self.assertRaises(ValueError):
            obs.analyze_control_space(self.S, self.targets, ["focus"])
        with self.assertRaises(ValueError):
            obs.analyze_control_space(self.S, self.targets, self.dofs,
                                      rel_threshold=2.0)
        with self.assertRaises(ValueError):
            obs.analyze_control_space([[np.nan]], ["Z4"], ["focus"])


class AstigmatismPlanTest(unittest.TestCase):
    """FEA plan: orientation from (Z6, Z5), label and recommendation."""

    def test_orientation_label_and_recommendation(self):
        plan = obs.astigmatism_plan(
            {"Z5_nm": 4.0, "Z6_nm": 3.0},
            dofs=["m2_focus", "m2_decenter_x", "astig_warp"],
        )
        expected = math.degrees(math.atan2(3.0, 4.0))
        self.assertEqual(plan["label"], "FEA-PROVISIONAL")
        self.assertTrue(plan["warning"])
        self.assertTrue(plan["recommendation"])
        self.assertAlmostEqual(plan["orientation_deg"], expected, places=9)
        self.assertAlmostEqual(plan["astigmatism_axis_deg"],
                               0.5 * expected, places=9)
        self.assertAlmostEqual(plan["magnitude_nm"], 5.0, places=12)
        self.assertIn("astig_warp", plan["can_correct"])
        self.assertIn("m2_focus", plan["cannot_correct"])
        self.assertIn("m2_decenter_x", plan["cannot_correct"])
        self.assertFalse(plan["drop_astigmatism_row"])
        self.assertGreater(plan["adjuster_count"], 0)
        self.assertEqual(len(plan["azimuths_deg"]), plan["adjuster_count"])
        self.assertIn("deg", plan["recommendation"])
        self.assertIn("nm", plan["recommendation"])
        self.assertGreater(plan["step"]["step_nm_surface"], 0.0)

    def test_axis_convention_for_negative_quadrant(self):
        plan = obs.astigmatism_plan({"Z5_nm": -1.0, "Z6_nm": -1.0},
                                    dofs=["astig_warp"])
        self.assertAlmostEqual(plan["orientation_deg"], -135.0, places=9)
        # axis = orientation/2 mod 180
        self.assertAlmostEqual(plan["astigmatism_axis_deg"], 112.5, places=9)
        self.assertEqual(plan["orientation_deg_360"], 225.0)

    def test_magnitude_phase_input(self):
        plan = obs.astigmatism_plan(
            {"magnitude_nm": 2.0, "orientation_deg": 60.0},
            dofs=["astig_warp"])
        self.assertAlmostEqual(plan["Z5_nm"], 1.0, places=12)
        self.assertAlmostEqual(plan["Z6_nm"], math.sqrt(3.0), places=12)
        self.assertEqual(plan["label"], "FEA-PROVISIONAL")

    def test_default_m2_mechanism_is_blind_to_astigmatism(self):
        # The user requirement: the M2 5-DOF set (focus/decenter/tilt) cannot
        # correct Z5/Z6, so the astigmatism rows must leave its matrix.
        from products.control.service import CONTROL_NAMES

        plan = obs.astigmatism_plan({"Z5_nm": 2.0, "Z6_nm": 1.0})
        self.assertEqual(plan["dofs"], list(CONTROL_NAMES))
        self.assertEqual(plan["can_correct"], [])
        self.assertEqual(plan["cannot_correct"], list(CONTROL_NAMES))
        self.assertTrue(plan["drop_astigmatism_row"])
        self.assertEqual(plan["recommended_dropped_rows"], ["Z5", "Z6"])
        self.assertEqual(plan["adjuster_count"], 4)

    def test_mechanism_sensitivity_matrix_path(self):
        summary = {
            "astigmatism": {"Z5_nm": 1.0, "Z6_nm": 0.0},
            "mechanism_sensitivity": {
                "targets": ["Z5", "Z6"],
                "dofs": ["translation_x", "astig_pair"],
                "matrix": [[0.0, 1.0], [0.0, 0.0]],
            },
        }
        plan = obs.astigmatism_plan(summary)
        self.assertEqual(plan["dofs"], ["translation_x", "astig_pair"])
        self.assertEqual(plan["can_correct"], ["astig_pair"])
        self.assertEqual(plan["cannot_correct"], ["translation_x"])

    def test_engineering_summary_astigmatism_block(self):
        summary = {
            "astigmatism": {
                "Z5_nm": 2.0,
                "Z6_nm": -2.0,
                "case": "M1_abq_gravity_90deg",
                "per_case": {
                    "small": {"Z5_nm": 0.1, "Z6_nm": 0.0},
                    "big": {"Z5_nm": 2.0, "Z6_nm": -2.0},
                },
            }
        }
        plan = obs.astigmatism_plan(summary, dofs=["astig_warp"])
        self.assertEqual(plan["case"], "M1_abq_gravity_90deg")
        self.assertAlmostEqual(plan["orientation_deg"], -45.0, places=9)
        self.assertIn("per_case", plan)
        self.assertEqual(set(plan["per_case"]), {"small", "big"})
        self.assertAlmostEqual(plan["per_case"]["big"]["magnitude_nm"],
                               2.0 * math.sqrt(2.0), places=9)


class EngineeringTermsSensitivityTest(unittest.TestCase):
    """The real Abaqus 6-term engineering matrix and its deliverables."""

    @classmethod
    def setUpClass(cls):
        cls.result = sens.engineering_terms_sensitivity("abaqus")

    def test_six_by_two_matrix_and_six_labels(self):
        self.assertEqual(self.result["targets"],
                         ["Z4", "Z5", "Z6", "Z8", "Z9", "Z11"])
        self.assertEqual(len(self.result["terms"]), 6)
        S = np.asarray(self.result["matrix"], dtype=float)
        self.assertEqual(S.shape, (6, 2))
        self.assertEqual(self.result["shape"], [6, 2])
        self.assertEqual(self.result["controls"], ["gravity_deg", "thermal_K"])
        self.assertEqual(self.result["source"], "abaqus")
        self.assertEqual(self.result["label"], "FEA-PROVISIONAL")
        self.assertIn(self.result["rank"], (1, 2))

    def test_observability_drops_astigmatism_rows(self):
        summary = self.result["observability"]
        self.assertEqual(summary["shape"], [6, 2])
        self.assertEqual(self.result["dropped_rows"], summary["dropped_rows"])
        self.assertIn("Z5", self.result["dropped_rows"])
        self.assertIn("Z6", self.result["dropped_rows"])
        self.assertEqual(len(self.result["reduced_matrix"]),
                         len(self.result["reduced_targets"]))
        self.assertTrue(summary["rationale"])

    def test_astigmatism_plan_is_fea_provisional(self):
        plan = self.result["astigmatism_plan"]
        self.assertEqual(plan["label"], "FEA-PROVISIONAL")
        self.assertTrue(plan["warning"])
        self.assertTrue(plan["recommendation"])
        self.assertGreater(plan["magnitude_nm"], 0.0)
        self.assertIn(plan["case"], self.result["source_cases"])
        self.assertGreaterEqual(plan["adjuster_count"], 1)

    def test_absolute_z5_z6_match_manifest_scale(self):
        # M1_abq gravity-90 Z5/Z6 are small (nm level), unlike its huge Z4.
        z = self.result["zernike_nm"]["M1_abq_gravity_90deg"]
        self.assertAlmostEqual(z["Z4"], -380.527304, places=4)
        self.assertLess(abs(z["Z5"]), 5.0)
        self.assertLess(abs(z["Z6"]), 5.0)
        self.assertGreater(self.result["astigmatism"]["magnitude_nm"], 0.0)

    def test_case_detail_still_66_term(self):
        detail = sens.case_detail("M1_abq_gravity_0deg")
        self.assertEqual(len(detail["fit"]["coefficients_nm"]), 66)
        self.assertEqual(len(detail["fit"]["terms"]), 66)
        self.assertIn("Z4", detail["fit"]["targets_nm"])

    def test_mechanism_dof_override(self):
        r = sens.engineering_terms_sensitivity(
            "abaqus", dofs=["astig_warp", "m2_focus"])
        plan = r["astigmatism_plan"]
        self.assertEqual(plan["can_correct"], ["astig_warp"])
        self.assertEqual(plan["cannot_correct"], ["m2_focus"])
        self.assertFalse(plan["drop_astigmatism_row"])
        self.assertTrue(plan["mount_recommendation"])

    def test_bad_source(self):
        with self.assertRaises(ValueError):
            sens.engineering_terms_sensitivity(source="nope")


class MatrixFromCasesTest(unittest.TestCase):
    def test_abaqus_columns_match_environment_sensitivity(self):
        columns = [
            {"name": "gravity_deg", "unit": "waves/deg",
             "plus": "M1_abq_gravity_90deg",
             "minus": "M1_abq_gravity_0deg", "delta": 90.0},
            {"name": "thermal_K", "unit": "waves/K",
             "plus": "M1_abq_thermal_plus20K",
             "minus": "M1_abq_thermal_minus20K", "delta": 40.0},
        ]
        r = sens.matrix_from_cases(columns, targets=["Z4", "Z8", "Z9"],
                                   label="FEA-PROVISIONAL")
        env = sens.environment_sensitivity(source="abaqus")
        self.assertEqual(np.asarray(r["matrix"]).shape, (3, 2))
        np.testing.assert_allclose(r["matrix"], env["matrix"],
                                   rtol=0.0, atol=1e-15)
        self.assertEqual(r["controls"], ["gravity_deg", "thermal_K"])
        self.assertEqual(len(r["source_cases"]), 4)

    def test_bad_column(self):
        with self.assertRaises(ValueError):
            sens.matrix_from_cases([{"plus": "M1_abq_gravity_0deg",
                                     "minus": "M1_abq_gravity_90deg",
                                     "delta": 0.0}])
        with self.assertRaises(ValueError):
            sens.matrix_from_cases([])


if __name__ == "__main__":
    unittest.main()
