"""Liver segmentation with the TotalSegmentator organ model (Wasserthal et al., 2023).

Each 2D slice is resampled to the 1.5 mm grid of the model (512 -> 256 px), mapped back to
approximate Hounsfield units with a standard abdominal window (W 400, L 40; pixels at the lower
window limit, i.e. air, are set to -1024 HU), rotated by 180 degrees to the orientation the model
was trained in, replicated into a thin pseudo-volume and segmented with the nnU-Net model of
TotalSegmentator (part 1, organs; liver = label 5) loaded once. The central slice of the liver
mask is kept, the largest connected component retained and holes filled.

usage: python liver_masks.py all|i,j,k [out.npy]
"""
import os, sys, time, glob, numpy as np, torch, cv2
from scipy import ndimage as ndi
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from config import WORK as ROOT
MODEL = glob.glob(os.path.expanduser(
    "~/.totalsegmentator/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/*3d_fullres"))[0]
W, L = 400.0, 40.0
DEPTH, SP = 24, 1.5
LIVER = 5

imgs = np.load(os.path.join(ROOT, "cache", "images512.npy"), mmap_mode="r")
ids = list(range(len(imgs))) if sys.argv[1] == "all" else [int(x) for x in sys.argv[1].split(",")]
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "cache", "liver_masks.npy")

pred = nnUNetPredictor(tile_step_size=1.0, use_gaussian=True, use_mirroring=False,
                       perform_everything_on_device=True, device=torch.device("cuda"),
                       verbose=False, verbose_preprocessing=False, allow_tqdm=False)
pred.initialize_from_trained_model_folder(MODEL, use_folds=(0,), checkpoint_name="checkpoint_final.pth")

masks = np.zeros((len(imgs), 512, 512), np.uint8)
t0 = time.time()
for n, i in enumerate(ids):
    v = cv2.resize(np.ascontiguousarray(imgs[i]), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    hu = v / 255.0 * W + (L - W / 2)
    hu[v <= 1] = -1024.0
    hu = np.ascontiguousarray(hu[::-1, ::-1])
    vol = np.repeat(hu[None], DEPTH, axis=0)[None]                  # (c, z, y, x)
    seg = pred.predict_single_npy_array(vol, {"spacing": [SP, SP, SP]}, None, None, False)
    m = np.ascontiguousarray((seg[DEPTH // 2] == LIVER)[::-1, ::-1]).astype(np.uint8)
    m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    lab, k = ndi.label(m)
    if k > 1:
        sizes = ndi.sum(m, lab, range(1, k + 1))
        m = lab == (np.argmax(sizes) + 1)
    masks[i] = ndi.binary_fill_holes(m)
    if n % 100 == 0:
        print(f"{n + 1}/{len(ids)}  {(time.time() - t0) / (n + 1):.2f} s/img", flush=True)
        if len(ids) == len(imgs):
            np.save(out_path, masks)
np.save(out_path, masks)
print("done %d images in %.1f min" % (len(ids), (time.time() - t0) / 60))
