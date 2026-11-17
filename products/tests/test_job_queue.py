import tempfile
import unittest
from pathlib import Path

from products.jobs.store import JobStore


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


if __name__ == "__main__":
    unittest.main()
