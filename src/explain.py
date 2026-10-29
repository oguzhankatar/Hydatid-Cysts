"""Held-out predictions, Grad-CAM and attention-rollout maps for every image.

Each image belongs to the held-out part (test or validation split) of exactly one fold and is
explained with the model of that fold, so no map is produced by a model trained on the image.

Grad-CAM (Selvaraju et al.) follows Section 2.4: target layer = blocks[-1].norm1, the class
token is removed and the 196 patch tokens are reshaped to a 14 x 14 grid (Katar & Yildirim).
Attention rollout (Abnar & Zuidema) multiplies the head-averaged attention matrices of the 12
blocks (with the residual identity added) and reads the class-token row.

Outputs (cache/explain/): heldout_predictions.csv, gradcam.npy (N,14,14 upsampled to 224),
and rollout.npy.
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F, timm, cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import WORK as ROOT
OUT = os.path.join(ROOT, "cache", "explain")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
imgs = np.load(os.path.join(ROOT, "cache", "images512.npy"), mmap_mode="r")
labels = np.load(os.path.join(ROOT, "cache", "labels.npy"))
split = json.load(open(os.path.join(ROOT, "cache", "splits.json")))
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(idx):
    t = torch.from_numpy(np.stack([imgs[i] for i in idx])).unsqueeze(1).float().to(dev)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True).round().clamp(0, 255) / 255.0
    return (t.expand(-1, 3, -1, -1) - MEAN) / STD


def reshape_transform(t, h=14, w=14):
    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
    return r.permute(0, 3, 1, 2)


N = len(labels)
gc_maps = np.zeros((N, 224, 224), np.float32)
ro_maps = np.zeros((N, 224, 224), np.float32)
rows = []
for k in range(1, 6):
    fold = split["folds"][k - 1]
    held = [(i, "test") for i in fold["test"]] + [(i, "validation") for i in fold["val"]]
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(os.path.join(ROOT, "runs", "main_vit", f"fold{k}", "best.pt"), map_location="cpu"))
    model.to(dev).eval()
    # attention capture for rollout
    attn_store = []
    for b in model.blocks:
        b.attn.fused_attn = False
        b.attn.attn_drop.register_forward_hook(lambda m, i, o: attn_store.append(o.detach()))
    cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)
    for s in range(0, len(held), 32):
        chunk = held[s:s + 32]
        idx = [i for i, _ in chunk]
        x = prep(idx)
        attn_store.clear()
        with torch.no_grad():
            prob = model(x).softmax(1).cpu().numpy()
        # rollout from the 12 captured attention tensors (B, heads, 197, 197)
        A = torch.stack(attn_store[:12]).mean(2)               # (12, B, 197, 197)
        I = torch.eye(A.size(-1), device=dev)
        R = I.expand(A.size(1), -1, -1).clone()
        for a in A:
            a = a + I
            a = a / a.sum(-1, keepdim=True)
            R = a @ R
        ro = R[:, 0, 1:].reshape(-1, 1, 14, 14)
        ro = F.interpolate(ro, size=(224, 224), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        pred = prob.argmax(1)
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(c)) for c in pred])
        for j, (i, role) in enumerate(chunk):
            gc_maps[i] = g[j]
            r = ro[j] - ro[j].min()
            ro_maps[i] = r / (r.max() + 1e-8)
            rows.append({"index": i, "path": split["paths"][i], "fold": k, "role": role, "label": int(labels[i]),
                         "pred": int(pred[j]), "p_complicated": prob[j, 0], "p_noncomplicated": prob[j, 1], "p_normal": prob[j, 2]})
    print("fold", k, "done", flush=True)
    del model, cam; torch.cuda.empty_cache()

df = pd.DataFrame(rows).sort_values("index")
df.to_csv(os.path.join(OUT, "heldout_predictions.csv"), index=False)
np.save(os.path.join(OUT, "gradcam.npy"), gc_maps.astype(np.float16))
np.save(os.path.join(OUT, "rollout.npy"), ro_maps.astype(np.float16))

print("held-out accuracy (test+val):", round(float((df.label == df.pred).mean()), 4))
