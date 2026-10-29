"""Load the CT images once, cache them as uint8 arrays and create the five cross-validation folds."""
import json, os, numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DATA as ROOT, WORK
OUT = os.path.join(WORK, "cache")
CLASSES = ["komp", "nonkomp", "normal"]          # complicated, non-complicated, normal
SEED = 42

paths, labels = [], []
for ci, c in enumerate(CLASSES):
    files = sorted(os.listdir(os.path.join(ROOT, c)), key=lambda f: int(f.split(".")[0]))
    paths += [f"{c}/{f}" for f in files]
    labels += [ci] * len(files)
labels = np.array(labels)

rgb_diff = 0
imgs = np.zeros((len(paths), 512, 512), np.uint8)
for i, p in enumerate(paths):
    im = Image.open(os.path.join(ROOT, p))
    if im.mode == "RGB":
        a = np.asarray(im).astype(int)
        rgb_diff = max(rgb_diff, int(np.abs(a[..., 0] - a[..., 1]).max()), int(np.abs(a[..., 1] - a[..., 2]).max()))
    im = im.convert("L")
    if im.size != (512, 512):
        im = im.resize((512, 512), Image.BILINEAR)
    imgs[i] = np.asarray(im)
np.save(os.path.join(OUT, "images512.npy"), imgs)
np.save(os.path.join(OUT, "labels.npy"), labels)
print("images", imgs.shape, "per class", np.bincount(labels), "max channel diff in RGB files", rgb_diff)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = []
for k, (rest, held) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    test, val = train_test_split(held, test_size=0.5, stratify=labels[held], random_state=SEED + k)
    folds.append({"train": sorted(map(int, rest)), "val": sorted(map(int, val)), "test": sorted(map(int, test))})
    print(f"fold {k+1}: train {len(rest)} val {len(val)} test {len(test)}  test/class {np.bincount(labels[test])}")
json.dump({"paths": paths, "classes": CLASSES, "folds": folds}, open(os.path.join(OUT, "splits.json"), "w"))
