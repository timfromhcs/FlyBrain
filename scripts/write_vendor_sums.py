#!/usr/bin/env python3
import hashlib
for f in ["src/ui/static/vendor/three.min.js", "src/ui/static/vendor/OrbitControls.js"]:
    print(hashlib.sha256(open(f, "rb").read()).hexdigest() + "  " + f.split("/")[-1])
