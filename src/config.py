"""Paths. Set HC_DATA to the folder that holds the class folders komp/, nonkomp/ and normal/
(numbered JPEG slices, e.g. komp/1.jpg) and HC_WORK to a working folder for caches, runs, figures
and results. Both default to folders next to src/."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.environ.get("HC_DATA", os.path.join(_HERE, "..", "data")))
WORK = os.path.abspath(os.environ.get("HC_WORK", os.path.join(_HERE, "..", "work")))
for sub in ("cache", "runs", "figures", "results"):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
