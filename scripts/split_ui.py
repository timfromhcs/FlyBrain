#!/usr/bin/env python3
"""Split src/ui/static/index.html monolith into index.html + css/lab.css + js/lab.js.

Local-first: CDN three.js references are replaced with vendored
/static/vendor/ files (pinned r128, hashes in src/ui/static/vendor/SHA256SUMS).
"""
import os
import re

STATIC = os.path.join("src", "ui", "static")
SRC = os.path.join(STATIC, "index.html")

html = open(SRC, encoding="utf-8").read()

# 1. Extract <style>...</style> -> css/lab.css
m = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
assert m, "no <style> block found"
css = m.group(1).strip() + "\n"
os.makedirs(os.path.join(STATIC, "css"), exist_ok=True)
open(os.path.join(STATIC, "css", "lab.css"), "w", encoding="utf-8").write(css)
html = html[:m.start()] + '<link rel="stylesheet" href="/static/css/lab.css">' + html[m.end():]

# 2. Replace CDN scripts with vendored copies
html = html.replace(
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>',
    '<script src="/static/vendor/three.min.js"></script>')
html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>',
    '<script src="/static/vendor/OrbitControls.js"></script>')

# 3. Extract trailing inline <script>...</script> (no src) -> js/lab.js
blocks = list(re.finditer(r"<script>(.*?)</script>", html, re.DOTALL))
assert blocks, "no inline <script> block found"
last = blocks[-1]
js = last.group(1).strip() + "\n"
os.makedirs(os.path.join(STATIC, "js"), exist_ok=True)
open(os.path.join(STATIC, "js", "lab.js"), "w", encoding="utf-8").write(js)
html = (html[:last.start()]
        + '<script src="/static/js/lab.js"></script>'
        + html[last.end():])

open(SRC, "w", encoding="utf-8").write(html)
print(f"index.html now {len(html)} bytes; css {len(css)} bytes; js {len(js)} bytes")
