import json
import os
import urllib.request

BASE = os.environ.get("MINIAPP_URL", "http://127.0.0.1:8000")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


meta = get("/api/framework")
print("meta", meta["target_names"], meta["control_names"], meta["target"])
s = post("/api/framework/sensitivity", {"step": 0.001})
print("S", s["values"], "rank", s["rank"], "cond", s["condition"], "null", s["nullspace"])
c = post("/api/framework/control", {"target": meta["target"], "lam": 1e-8, "use_bounds": True})
print("ctrl", c["u"], "res", c["residual"], "cost", c["cost"])
print("index", urllib.request.urlopen(BASE + "/", timeout=10).status)
