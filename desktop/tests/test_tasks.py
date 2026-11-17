import unittest
from pathlib import Path

from desktop.protocol import build_worker_command, make_job, parse_job

ROOT = Path(__file__).resolve().parents[2]


class WorkerCommandTest(unittest.TestCase):
    def test_source_points_to_entrypoint(self):
        prog, args = build_worker_command("ccd", "job.json", "res.json", frozen=False,
                                          root=ROOT)
        self.assertEqual(Path(args[0]).resolve(),
                         (ROOT / "desktop" / "qt_app.py").resolve())
        self.assertIn("--worker", args)

    def test_frozen_uses_exe(self):
        prog, args = build_worker_command("s", "j", "r", frozen=True, root=ROOT)
        self.assertEqual(args[0], "--worker")

    def test_job_round_trip(self):
        doc = make_job("u", {"target": [1, 2, 3]}, "abc")
        self.assertEqual(parse_job(doc, "u")["target"], [1, 2, 3])
        with self.assertRaises(ValueError):
            parse_job(doc, "ccd")


if __name__ == "__main__":
    unittest.main()
