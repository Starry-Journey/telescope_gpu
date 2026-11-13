import json
import os
import urllib.request

BASE = os.environ.get("MINIAPP_URL", "http://127.0.0.1:8000")


def post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


print("health", urllib.request.urlopen(BASE + "/api/health", timeout=10).read().decode())
sim = post("/api/simulate", {"c4": 0.6, "c8": -0.2, "c9": 0.1, "noise": 0.01})
print("simulate frames", len(sim["frames"]), sim["shape"], sim["frame_peak"][:2])
inv = post("/api/invert", {"c4": 0.6, "c8": -0.2, "c9": 0.1, "noise": 0.01})
print("invert truth", inv["truth"], "cnn", inv["cnn"], "refined", inv["refined"],
      "nfev", inv["nfev"], "cost", inv["cost"])
print("index status", urllib.request.urlopen(BASE + "/", timeout=10).status)
