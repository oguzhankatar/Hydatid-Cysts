"""Classify one CT slice with a released fold model and save its Grad-CAM map.

The slice must be exported as in the paper: abdominal soft-tissue window (W 400 HU, L 40 HU), 8-bit gray
levels; it is resized to 224 x 224 with bilinear interpolation, replicated to three channels and normalized
with the ImageNet statistics. Grad-CAM uses the output of blocks[-1].norm1 with the class token removed and the
196 patch tokens reshaped to a 14 x 14 grid (Section 2.4 of the paper).

    python src/predict.py --image slice.jpg --weights hc_vit_b16_fold1.pt --out slice_gradcam.png
"""
import argparse
import numpy as np, torch, torch.nn.functional as F, timm, cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSES = ["complicated", "non-complicated", "normal"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def reshape_transform(t, h=14, w=14):
    return t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", required=True, help="one of the released fold models, e.g. hc_vit_b16_fold1.pt")
    p.add_argument("--out", default=None, help="PNG with the Grad-CAM map overlaid on the slice")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    g = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {a.image}")
    t = torch.from_numpy(g).float()[None, None]
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True).round().clamp(0, 255)
    x = ((t / 255.0).expand(-1, 3, -1, -1) - MEAN) / STD
    x = x.to(a.device)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    model.to(a.device).eval()
    with torch.no_grad():
        prob = model(x).softmax(1)[0].cpu().numpy()
    k = int(prob.argmax())
    for c, v in zip(CLASSES, prob):
        print(f"{c:16s} {v:.4f}")
    print("prediction:", CLASSES[k])

    if a.out:
        cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)
        m = cam(input_tensor=x, targets=[ClassifierOutputTarget(k)])[0]
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_VIRIDIS)
        alpha = 0.75 * m[..., None]
        over = (cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) * (1 - alpha) + heat * alpha).astype(np.uint8)
        cv2.imwrite(a.out, over)
        print("Grad-CAM map written to", a.out)


if __name__ == "__main__":
    main()
