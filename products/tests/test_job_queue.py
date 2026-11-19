import tempfile
import unittest
from pathlib import Path

from products.jobs.store import JobStore
import products.jobs.store as js


class JobQueueTest(unittest.TestCase):
    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(Path(td) / "jobs.sqlite3")
            jid = store.submit("zemax", {"name": "demo", "data_b64": "abc"})
            self.assertEqual(store.get(jid)["status"], "queued")
            job = store.claim("zemax", "worker-1")
            self.assertEqual(job["id"], jid)
            store.complete(jid, {"mock": True})
            self.assertEqual(store.get(jid)["status"], "done")
            # a second claim must not steal a running/done job
            self.assertIsNone(store.claim("zemax", "worker-2"))

    def test_default_root(self):
        self.assertEqual(js.ROOT, Path(__file__).resolve().parents[2])
        self.assertEqual(js.DB_PATH, js.ROOT / "results" / "jobs.sqlite3")

    def test_lazy_singleton(self):
        """get_store creates the singleton only on first call, honoring TEL_JOB_DB."""
        import importlib
        import os
        import tempfile

        # A long-lived workspace may already have results/jobs.sqlite3 from
        # earlier app/worker runs; the invariant is that *this test* must not
        # create it, not that the file can never exist.
        repo_db = js.ROOT / "results" / "jobs.sqlite3"
        existed_before = repo_db.exists()
        with tempfile.TemporaryDirectory() as td:
            os.environ["TEL_JOB_DB"] = str(Path(td) / "lazy.sqlite3")
            try:
                mod = importlib.import_module("products.jobs.store")
                mod = importlib.reload(mod)  # fresh: singleton must start unset
                self.assertIsNone(mod._store)
                s1 = mod.get_store()
                self.assertIs(mod._store, s1)
                self.assertEqual(s1.path, Path(os.environ["TEL_JOB_DB"]))
                self.assertTrue(s1.path.exists())
                self.assertIs(mod.get_store(), s1)
            finally:
                os.environ.pop("TEL_JOB_DB", None)
                importlib.reload(mod)  # restore default module state
        self.assertFalse(not existed_before and repo_db.exists(),
                         "lazy test must not create a DB inside the repo")


if __name__ == "__main__":
    unittest.main()
