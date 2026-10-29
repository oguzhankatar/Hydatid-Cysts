"""Figs. 8, 9 and 11: Grad-CAM examples, misclassified examples, Grad-CAM vs attention rollout.

Maps come from explain.py (each image explained by the model of the fold in which it was held
out). Heat maps use the colour-blind-safe viridis map with an opacity proportional to the
activation, so that the underlying anatomy remains visible."""
import os, json, numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyArrow
import figstyle as fs

from config import WORK as ROOT
FIG = os.path.join(ROOT, "figures")
X = np.load(os.path.join(ROOT, "cache", "images512.npy"), mmap_mode="r")
G = np.load(os.path.join(ROOT, "cache", "explain", "gradcam.npy")).astype(np.float32)
R = np.load(os.path.join(ROOT, "cache", "explain", "rollout.npy")).astype(np.float32)
P = pd.read_csv(os.path.join(ROOT, "cache", "explain", "heldout_predictions.csv")).set_index("index")
VIR = colormaps["viridis"]


def base(i, size=448):
    return cv2.resize(np.ascontiguousarray(X[i]), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def overlay(i, m, size=448):
    g = base(i, size)
    m = cv2.resize(np.clip(m, 0, 1), (size, size), interpolation=cv2.INTER_LINEAR)
    heat = VIR(m)[..., :3]
    a = 0.75 * m[..., None]
    return np.clip(g[..., None] * (1 - a) + heat * a, 0, 1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def colorbar(fig, rect, label):
    cax = fig.add_axes(rect)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=VIR), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.outline.set_linewidth(0.5)
    cax.set_title(label, fontsize=8, pad=2)


# ------------------------------------------------------------ Fig. 8
NONC = [1991, 1993, 1804, 1411]          # (a)-(d) non-complicated (illustrative, held-out test images)
COMP = [319, 56, 497, 814]               # (e)-(h) complicated (illustrative, held-out test images)
for i in NONC + COMP:
    assert P.loc[i, "label"] == P.loc[i, "pred"], i
fig = plt.figure(figsize=(fs.FULL, 190 * fs.MM))
w, h, gap = 0.215, 0.215, 0.005
for col, (ids, letters) in enumerate(((NONC, "abcd"), (COMP, "efgh"))):
    x0 = 0.04 + col * 0.49
    for row, (i, L) in enumerate(zip(ids, letters)):
        y0 = 0.76 - row * (h + 0.017)
        a1 = fig.add_axes([x0, y0, w, h]); a1.imshow(base(i), cmap="gray", vmin=0, vmax=1); off(a1)
        a2 = fig.add_axes([x0 + w + gap, y0, w, h]); a2.imshow(overlay(i, G[i])); off(a2)
        fig.text(x0 - 0.008, y0 + h / 2, f"({L})", ha="right", va="center")
fig.text(0.04 + w + gap / 2, 0.985, "Non-complicated", ha="center", va="top")
fig.text(0.53 + w + gap / 2, 0.985, "Complicated", ha="center", va="top")
colorbar(fig, [0.38, 0.035, 0.24, 0.012], "Grad-CAM activation")
fs.save(fig, os.path.join(FIG, "Fig8"))

# ------------------------------------------------------------ Fig. 9
CASES = [(1130, (150, 235), (85, 300)), (1841, (205, 215), (130, 300))]
names = ["Complicated", "Non-complicated", "Normal"]
fig = plt.figure(figsize=(fs.FULL, 120 * fs.MM))
info = {}
for r, (i, tip, tail) in enumerate(CASES):
    y0 = 0.53 - r * 0.47
    lab, pred = int(P.loc[i, "label"]), int(P.loc[i, "pred"])
    prob = P.loc[i, ["p_complicated", "p_noncomplicated", "p_normal"]].values.astype(float)
    info[i] = {"label": names[lab], "pred": names[pred], "prob": [round(100 * p, 1) for p in prob], "fold": int(P.loc[i, "fold"]),
               "role": P.loc[i, "role"]}
    a1 = fig.add_axes([0.03, y0, 0.30, 0.40]); a1.imshow(base(i), cmap="gray", vmin=0, vmax=1); off(a1)
    sx = 448 / 512
    a1.add_patch(FancyArrow(tail[0] * sx, tail[1] * sx, (tip[0] - tail[0]) * sx * 0.85, (tip[1] - tail[1]) * sx * 0.85,
                            width=5, head_width=22, head_length=22, color="#E69F00", ec="black", lw=0.6, length_includes_head=True))
    a1.set_title("Input image", pad=3)
    a2 = fig.add_axes([0.43, y0 + 0.05, 0.22, 0.30])
    bars = a2.bar(range(3), prob * 100, color=fs.CLASS_COLORS, edgecolor="black", linewidth=0.5)
    for b, p in zip(bars, prob):
        a2.text(b.get_x() + b.get_width() / 2, p * 100 + 2, f"{100 * p:.1f}%", ha="center", va="bottom")
    a2.set_xticks(range(3)); a2.set_xticklabels(["Comp.", "Non-comp.", "Normal"])
    a2.set_ylim(0, 115); a2.set_yticks([0, 50, 100]); a2.set_ylabel("Softmax probability (%)")
    a2.set_title(f"True label: {names[lab]}\nPredicted label: {names[pred]}", pad=3)
    a3 = fig.add_axes([0.68, y0, 0.30, 0.40]); a3.imshow(overlay(i, G[i])); off(a3)
    a3.set_title("Grad-CAM output", pad=3)
    fig.text(0.005, y0 + 0.20, "(a)" if r == 0 else "(b)", ha="left", va="center")
fs.save(fig, os.path.join(FIG, "Fig9"))
json.dump(info, open(os.path.join(ROOT, "results", "fig9_cases.json"), "w"), indent=1)
print(info)

# ------------------------------------------------------------ Fig. 11 Grad-CAM vs attention rollout
SEL = [1991, 1804, 319, 814]
fig = plt.figure(figsize=(fs.FULL, 150 * fs.MM))
for c, i in enumerate(SEL):
    x0 = 0.02 + c * 0.245
    for r, (img, title) in enumerate(((np.stack([base(i)] * 3, -1), "Input image"), (overlay(i, G[i]), "Grad-CAM"),
                                      (overlay(i, R[i]), "Attention rollout"))):
        ax = fig.add_axes([x0, 0.665 - r * 0.30, 0.235, 0.28]); ax.imshow(img); off(ax)
        if c == 0:
            fig.text(x0 - 0.006, 0.805 - r * 0.30, title, rotation=90, ha="right", va="center")
    fig.text(x0 + 0.1175, 0.985, f"({'abcd'[c]}) {['Non-complicated', 'Non-complicated', 'Complicated', 'Complicated'][c]}", ha="center", va="top")
colorbar(fig, [0.38, 0.015, 0.24, 0.011], "Activation / relevance")
fs.save(fig, os.path.join(FIG, "Fig11"))
