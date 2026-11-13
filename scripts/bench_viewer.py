"""Time the 3D viewer in real Chrome (GPU): seconds until the scene is built, and frames per second.

    python scripts/bench_viewer.py out/area/viewer [--copies 9]      # needs Google Chrome + websocket-client

Makes a copy of the viewer whose area is tiled `copies` times (a bigger ward, same buildings), adds a
small probe script, opens it in headless Chrome over the DevTools protocol and reads the result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROBE = ('<script>window.__ready=performance.now();setTimeout(function(){var n=0,t0=performance.now();(function f(){n++;'
         'if(performance.now()-t0<5000)requestAnimationFrame(f);else document.body.setAttribute("data-bench",JSON.stringify('
         '{ready_ms:Math.round(window.__ready),fps:+(n/((performance.now()-t0)/1000)).toFixed(1),units:(window.CADASTRE||{units:[]}).units.length}))})()},1500)'
         '</script></body>')


def tiled_copy(viewer: Path, copies: int) -> Path:
    d = Path(tempfile.mkdtemp()) / "viewer"
    shutil.copytree(viewer, d)
    src = (d / "cadastre_data.js").read_text()
    data = json.loads(src[src.index("=") + 1:src.rstrip().rindex(";")])
    b = data["meta"]["bounds"]
    w, h, side = b[2] - b[0], b[3] - b[1], int(round(copies ** 0.5))
    units = []
    for k in range(side * side):
        ox, oy = (k % side) * w, (k // side) * h
        units += [{**u, "poly": [[p[0] + ox, p[1] + oy] for p in u["poly"]] if u.get("poly") else u.get("poly")} for u in data["units"]]
    data.update(units=units, parcels=[], ortho=None)
    data["meta"] = {**data["meta"], "bounds": [b[0], b[1], b[0] + side * w, b[1] + side * h]}
    (d / "cadastre_data.js").write_text("window.CADASTRE=" + json.dumps(data) + ";")
    (d / "index.html").write_text((d / "index.html").read_text().replace("</body>", PROBE))
    return d


def measure(index_html: Path, port=9333, wait=30) -> dict:
    import websocket

    p = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", "--use-angle=metal", "--enable-gpu",
                          "--ignore-gpu-blocklist", "--window-size=1400,900", f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except Exception:
                time.sleep(0.2)
        ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"])
        ws.send(json.dumps({"id": 1, "method": "Page.navigate", "params": {"url": index_html.resolve().as_uri()}}))
        t0 = time.time()
        while time.time() - t0 < wait:
            time.sleep(1)
            ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate",
                                "params": {"expression": "document.body&&document.body.getAttribute('data-bench')", "returnByValue": True}}))
            m = json.loads(ws.recv())
            while m.get("id") != 2:
                m = json.loads(ws.recv())
            v = m["result"]["result"].get("value")
            if v:
                return json.loads(v)
        return {}
    finally:
        p.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("viewers", nargs="+", help="viewer folders (index.html + cadastre_data.js)")
    ap.add_argument("--copies", type=int, default=9)
    a = ap.parse_args()
    for v in a.viewers:
        print(v, measure(tiled_copy(Path(v), a.copies) / "index.html"), flush=True)
