"""Tables and figures of the revised manuscript from the five-fold runs.

Table 5 (per-fold metrics, mean, SD, 95% CI), ablation table, Figs. 5-7."""
import os, json, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

from config import WORK as ROOT
FIG = os.path.join(ROOT, "figures")
OUTJ = os.path.join(ROOT, "results")
os.makedirs(OUTJ, exist_ok=True)
CL = ["comp", "noncomp", "normal"]


def load(exp):
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "runs", exp, "fold*", "metrics.json")))]
    return pd.DataFrame(ms).sort_values("fold") if ms else pd.DataFrame()


def ci95(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


summary = {}
# ---------------------------------------------------------------- Table 5
m = load("main_vit")
cols = [f"{a}_{c}" for c in CL for a in ("pre", "rec", "f1")] + ["test_acc"]
t5 = m[["fold"] + cols].copy()
t5[cols] = t5[cols] * 100
mean, sd = t5[cols].mean(), t5[cols].std(ddof=1)
lo = pd.Series({c: ci95(t5[c])[0] for c in cols}); hi = pd.Series({c: ci95(t5[c])[1] for c in cols})
summary["table5"] = {"per_fold": t5.round(2).to_dict("records"), "mean": mean.round(2).to_dict(), "sd": sd.round(2).to_dict(),
                     "ci_low": lo.round(2).to_dict(), "ci_high": hi.round(2).to_dict()}
cm_total = np.sum([np.array(c) for c in m.cm], axis=0)
k, n = int(np.trace(cm_total)), int(cm_total.sum())
summary["pooled"] = {"correct": k, "n": n, "acc": 100 * k / n, "wilson_low": 100 * wilson(k, n)[0], "wilson_high": 100 * wilson(k, n)[1],
                     "cm": cm_total.tolist()}
summary["training"] = m[["fold", "best_epoch", "stopped_epoch", "best_val_loss", "best_val_acc"]].round(4).to_dict("records")
macro_f1 = m[[f"f1_{c}" for c in CL]].mean(1) * 100
summary["macro_f1"] = {"mean": float(macro_f1.mean()), "sd": float(macro_f1.std(ddof=1))}

# ---------------------------------------------------------------- ablation
ab = []
for exp, name in [("main_vit", "Full slice, no augmentation (proposed)"), ("abl_aug", "Full slice + data augmentation"),
                  ("abl_clahe", "Full slice + CLAHE"), ("body_vit", "Body-segmented slice"), ("abl_liver", "Liver ROI (TotalSegmentator)")]:
    d = load(exp)
    if len(d) < 5:
        continue
    acc = d.test_acc * 100
    mf1 = d[[f"f1_{c}" for c in CL]].mean(1) * 100
    row = {"exp": exp, "variant": name, "acc_mean": acc.mean(), "acc_sd": acc.std(ddof=1), "acc_ci": ci95(acc),
           "macro_f1_mean": mf1.mean(), "macro_f1_sd": mf1.std(ddof=1)}
    for c in CL:
        row[f"f1_{c}_mean"] = d[f"f1_{c}"].mean() * 100
        row[f"rec_{c}_mean"] = d[f"rec_{c}"].mean() * 100
    if exp != "main_vit":
        diff = acc.values - (m.test_acc.values * 100)
        row["acc_diff_mean"] = diff.mean()
        row["acc_diff_ci"] = ci95(diff)
    ab.append(row)
summary["ablation"] = ab
json.dump(summary, open(os.path.join(OUTJ, "summary.json"), "w"), indent=2, default=float)
print(json.dumps({"mean": summary["table5"]["mean"], "pooled": summary["pooled"]}, indent=1, default=float))
for r in ab:
    print(f"{r['variant']:45s} acc {r['acc_mean']:.2f} ± {r['acc_sd']:.2f}  macroF1 {r['macro_f1_mean']:.2f}  "
          f"F1 c/nc/n {r['f1_comp_mean']:.2f}/{r['f1_noncomp_mean']:.2f}/{r['f1_normal_mean']:.2f}"
          + (f"  diff {r['acc_diff_mean']:+.2f} CI {r['acc_diff_ci'][0]:+.2f}..{r['acc_diff_ci'][1]:+.2f}" if 'acc_diff_mean' in r else ""))

# ---------------------------------------------------------------- Fig. 5 training curves
fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 62 * fs.MM))
for k in range(1, 6):
    h = pd.read_csv(os.path.join(ROOT, "runs", "main_vit", f"fold{k}", "history.csv"))
    c = fs.FOLD_COLORS[k - 1]
    be = int(h.val_loss.idxmin())
    axes[0].plot(h.epoch, h.train_acc * 100, color=c, ls="-")
    axes[0].plot(h.epoch, h.val_acc * 100, color=c, ls="--")
    axes[1].plot(h.epoch, h.train_loss, color=c, ls="-")
    axes[1].plot(h.epoch, h.val_loss, color=c, ls="--")
    axes[1].plot(h.epoch[be], h.val_loss[be], marker="o", ms=4, color=c, mec="black", mew=0.5, ls="none")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
for a, t in zip(axes, ("(a)", "(b)")):
    a.text(0.5, -0.30, t, transform=a.transAxes, ha="center", va="top")
    a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fold_h = [Line2D([], [], color=fs.FOLD_COLORS[k], lw=1.2, label=f"Fold {k + 1}") for k in range(5)]
style_h = [Line2D([], [], color="black", ls="-", label="Training"), Line2D([], [], color="black", ls="--", label="Validation"),
           Line2D([], [], color="grey", marker="o", mec="black", mew=0.5, ls="none", ms=4, label="Lowest validation loss")]
fig.legend(handles=fold_h + style_h, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.04), handlelength=1.6, columnspacing=1.0)
fig.subplots_adjust(wspace=0.28)
fs.save(fig, os.path.join(FIG, "Fig5"))

# ---------------------------------------------------------------- Fig. 6 confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(fs.FULL, 48 * fs.MM))
short = ["Comp.", "Non-comp.", "Normal"]
for k, ax in enumerate(axes, 1):
    cm = np.array(m.loc[m.fold == k, "cm"].iloc[0])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_yticklabels([f"{s} (n = {cm[i].sum()})" for i, s in enumerate(short)] if k == 1 else [])
    ax.set_title(f"Fold {k}")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    ax.tick_params(length=2)
axes[0].set_ylabel("True label")
fig.subplots_adjust(wspace=0.12, bottom=0.30)
fig.text(0.56, 0.0, "Predicted label", ha="center", va="bottom")
fs.save(fig, os.path.join(FIG, "Fig6"))

# ---------------------------------------------------------------- Fig. 7 softmax distributions of correct predictions
fig, axes = plt.subplots(1, 5, figsize=(fs.FULL, 50 * fs.MM), sharey=True)
counts = {}
for k, ax in enumerate(axes, 1):
    p = pd.read_csv(os.path.join(ROOT, "runs", "main_vit", f"fold{k}", "test_predictions.csv"))
    ok = p[p.label == p.pred].reset_index(drop=True)
    pc = ok[["p_complicated", "p_noncomplicated", "p_normal"]].values[np.arange(len(ok)), ok.label.values]
    for c in range(3):
        sel = ok.label.values == c
        ax.scatter(np.arange(len(ok))[sel], pc[sel], s=4, color=fs.CLASS_COLORS[c], marker="os^"[c], lw=0, alpha=0.85)
        counts.setdefault(k, {})[c] = int(sel.sum())
    ax.set_title(f"Fold {k}")
    ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("Softmax probability of the true class")
hs = [Line2D([], [], color=fs.CLASS_COLORS[c], marker="os^"[c], ls="none", ms=4, label=fs.CLASS_NAMES[c]) for c in range(3)]
fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.93))
fig.subplots_adjust(wspace=0.08, top=0.84, bottom=0.2)
fig.text(0.5, 0.02, "Index of correctly classified test image", ha="center", va="bottom")
fs.save(fig, os.path.join(FIG, "Fig7"))
json.dump(counts, open(os.path.join(OUTJ, "fig7_counts.json"), "w"))
low = {}
for k in range(1, 6):
    p = pd.read_csv(os.path.join(ROOT, "runs", "main_vit", f"fold{k}", "test_predictions.csv"))
    ok = p[p.label == p.pred]
    pc = ok[["p_complicated", "p_noncomplicated", "p_normal"]].values[np.arange(len(ok)), ok.label.values]
    low[k] = {"below_0.9": int((pc < 0.9).sum()), "by_class_below_0.9": [int(((pc < 0.9) & (ok.label.values == c)).sum()) for c in range(3)],
              "normal_min": float(pc[ok.label.values == 2].min())}
json.dump(low, open(os.path.join(OUTJ, "fig7_lowconf.json"), "w"), indent=1)
print("low-confidence counts:", low)
