"""Body segmentation: threshold + largest connected component + hole filling.
Removes air, the CT table and any burned-in text outside the patient body."""
import numpy as np, cv2, os
from scipy import ndimage as ndi
from config import WORK as ROOT
imgs = np.load(os.path.join(ROOT, "cache", "images512.npy"))
masks = np.zeros_like(imgs, dtype=np.uint8)
for i, im in enumerate(imgs):
    b = cv2.GaussianBlur(im, (3, 3), 0) > 5
    b = ndi.binary_closing(b, structure=np.ones((5, 5)), iterations=2)     # close the skin outline
    b = ndi.binary_fill_holes(b)
    b = ndi.binary_opening(b, structure=np.ones((9, 9)), iterations=2)     # detach table / thin structures
    lab, n = ndi.label(b)
    if n:
        sizes = ndi.sum(b, lab, range(1, n + 1))
        b = lab == (np.argmax(sizes) + 1)
    b = ndi.binary_fill_holes(b)
    masks[i] = b
np.save(os.path.join(ROOT, "cache", "body_masks.npy"), masks)
area = masks.reshape(len(masks), -1).mean(1)
print("body area fraction: median %.3f, min %.3f, max %.3f" % (np.median(area), area.min(), area.max()))
print("slices with area < 0.15:", int((area < 0.15).sum()))
