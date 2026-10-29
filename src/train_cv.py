"""Five-fold training/evaluation of the ViT classifier and its ablation variants.

Protocol (Section 2.2 / Table 4 of the manuscript):
  * ViT-B/16 (timm vit_base_patch16_224, ImageNet pre-trained), 224x224 input, bilinear resize
  * AdamW (weight decay 0.01), lr 1e-5 (chosen from 1e-4, 3e-5 and 1e-5 by the mean validation loss
    of folds 1-2), batch 32, cross-entropy, max 50 epochs,
    early stopping on validation loss (patience 5); the checkpoint with the lowest
    validation loss is restored and evaluated on the test split of the fold.

Variants (ablation):
  --input full|body|liver   full slice / body-segmented slice / liver-ROI crop
  --preproc none|clahe      optional CLAHE contrast enhancement
  --aug                     on-the-fly data augmentation
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm, cv2
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from config import WORK as ROOT
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True)
p.add_argument("--model", default="vit_base_patch16_224")
p.add_argument("--input", default="full", choices=["full", "body", "liver"])
p.add_argument("--preproc", default="none", choices=["none", "clahe"])
p.add_argument("--aug", action="store_true")
p.add_argument("--size", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--folds", default="1,2,3,4,5")
p.add_argument("--save_weights", action="store_true")
p.add_argument("--split", default="splits.json", help="fold file in HC_WORK/cache (written by prepare_data.py)")
args = p.parse_args()

dev = "cuda"
out_dir = os.path.join(ROOT, "runs", args.exp)
os.makedirs(out_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

imgs = np.load(os.path.join(ROOT, "cache", "images512.npy"))
labels = np.load(os.path.join(ROOT, "cache", "labels.npy"))
split = json.load(open(os.path.join(ROOT, "cache", args.split)))


def build_inputs():
    x = imgs.copy()
    if args.preproc == "clahe":
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = np.stack([cl.apply(im) for im in x])
    if args.input == "body":
        m = np.load(os.path.join(ROOT, "cache", "body_masks.npy"))
        x = x * m
    elif args.input == "liver":
        # liver region of interest: convex hull of the TotalSegmentator liver mask, dilated by
        # 15 px (about 1 cm) so that intrahepatic cysts excluded from the organ label are kept,
        # applied to the slice and cropped to its bounding box (padded to a square)
        m = np.load(os.path.join(ROOT, "cache", "liver_masks.npy"))
        out = np.zeros((len(x), args.size, args.size), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        for i in range(len(x)):
            roi = np.zeros((512, 512), np.uint8)
            cnts, _ = cv2.findContours(m[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts and m[i].sum() >= 500:
                cv2.fillPoly(roi, [cv2.convexHull(np.vstack(cnts))], 1)
                roi = cv2.dilate(roi, kernel)
            else:                                 # no liver found: fall back to the body mask
                roi = np.load(os.path.join(ROOT, "cache", "body_masks.npy"), mmap_mode="r")[i].copy()
            ys, xs = np.nonzero(roi)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = (x[i] * roi)[y0:y1, x0:x1]
            h, w = crop.shape
            s_ = max(h, w)
            sq = np.zeros((s_, s_), np.uint8)
            sq[(s_ - h) // 2:(s_ - h) // 2 + h, (s_ - w) // 2:(s_ - w) // 2 + w] = crop
            out[i] = cv2.resize(sq, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(out).unsqueeze(1)
    t = torch.from_numpy(x).unsqueeze(1).float()
    t = F.interpolate(t, size=(args.size, args.size), mode="bilinear", align_corners=False, antialias=True)
    return t.round().clamp(0, 255).to(torch.uint8)


X = build_inputs().to(dev)                        # (N,1,S,S) uint8 on GPU
Y = torch.from_numpy(labels).long().to(dev)
mean, std = MEAN.to(dev), STD.to(dev)
augment = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
])


def to_model(xb, train):
    xb = xb.float() / 255.0
    if train and args.aug:
        xb = torch.stack([augment(im) for im in xb])
    xb = xb.expand(-1, 3, -1, -1)
    return (xb - mean) / std


@torch.no_grad()
def evaluate(model, idx):
    model.eval()
    losses, probs = [], []
    for s in range(0, len(idx), 128):
        b = idx[s:s + 128]
        logits = model(to_model(X[b], False))
        losses.append(F.cross_entropy(logits, Y[b], reduction="sum").item())
        probs.append(logits.softmax(1).cpu())
    probs = torch.cat(probs).numpy()
    yt = labels[idx.cpu().numpy()]
    return sum(losses) / len(idx), float((probs.argmax(1) == yt).mean()), probs


summary = []
for k in [int(f) for f in args.folds.split(",")]:
    fold = split["folds"][k - 1]
    tr, va, te = (torch.tensor(fold[s], device=dev) for s in ("train", "val", "test"))
    torch.manual_seed(42 + k); np.random.seed(42 + k)
    model = timm.create_model(args.model, pretrained=True, num_classes=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, best_state, wait, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        tl, tc = 0.0, 0
        for s in range(0, len(perm), args.bs):
            b = perm[s:s + args.bs]
            logits = model(to_model(X[b], True))
            loss = F.cross_entropy(logits, Y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += loss.item() * len(b); tc += (logits.argmax(1) == Y[b]).sum().item()
        vl, vacc, _ = evaluate(model, va)
        hist.append({"epoch": ep, "train_loss": tl / len(tr), "train_acc": tc / len(tr), "val_loss": vl, "val_acc": vacc})
        print(f"[{args.exp}] fold {k} ep {ep:2d} train {tl/len(tr):.4f}/{tc/len(tr):.4f} val {vl:.4f}/{vacc:.4f}", flush=True)
        if vl < best:
            best, wait = vl, 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    best_ep = int(np.argmin([h["val_loss"] for h in hist])) + 1
    _, _, probs = evaluate(model, te)
    yt, yp = labels[fold["test"]], probs.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
    acc = accuracy_score(yt, yp)
    fdir = os.path.join(out_dir, f"fold{k}")
    os.makedirs(fdir, exist_ok=True)
    pd.DataFrame(hist).to_csv(os.path.join(fdir, "history.csv"), index=False)
    pd.DataFrame({"index": fold["test"], "path": [split["paths"][i] for i in fold["test"]], "label": yt, "pred": yp,
                  "p_complicated": probs[:, 0], "p_noncomplicated": probs[:, 1], "p_normal": probs[:, 2]}
                 ).to_csv(os.path.join(fdir, "test_predictions.csv"), index=False)
    res = {"fold": k, "stopped_epoch": len(hist), "best_epoch": best_ep, "best_val_loss": best,
           "best_val_acc": hist[best_ep - 1]["val_acc"], "test_acc": acc,
           **{f"{m}_{c}": float(v[i]) for m, v in (("pre", pr), ("rec", rc), ("f1", f1)) for i, c in enumerate(["comp", "noncomp", "normal"])},
           "cm": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(), "minutes": (time.time() - t0) / 60}
    json.dump(res, open(os.path.join(fdir, "metrics.json"), "w"), indent=2)
    if args.save_weights:
        torch.save(best_state, os.path.join(fdir, "best.pt"))
    summary.append(res)
    print(f"[{args.exp}] fold {k} DONE best_ep {best_ep} test_acc {acc:.4f} cm {res['cm']}", flush=True)
    del model, opt, best_state; torch.cuda.empty_cache()

df = pd.DataFrame(summary)
df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
print(df[["fold", "best_epoch", "test_acc"]].to_string(index=False))
print(f"[{args.exp}] mean acc {df.test_acc.mean():.4f} +- {df.test_acc.std():.4f}")
