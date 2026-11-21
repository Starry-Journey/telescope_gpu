import json
import urllib.request

BASE = "http://127.0.0.1:8770"


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


print("health", urllib.request.urlopen(BASE + "/api/health", timeout=10).read().decode())
ccd = post("/api/ccd", {"stars": 40, "seed": 7})
print("ccd", ccd["stars"], ccd["shape"], ccd["image"][:30], len(ccd["image"]))
inv = post("/api/invert", {"c4": 0.6, "c8": -0.2, "c9": 0.1, "noise": 0.01})
print("invert", inv["refined"], inv["nfev"], inv["model_loaded"])
