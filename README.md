# Explainable Vision Transformer-Based Classification of Liver Hydatid Cysts on CT Images

Code, fold assignments, per-fold results and trained weights for the paper

> **Explainable Vision Transformer-Based Classification of Liver Hydatid Cysts on CT Images: A Deep Learning Approach for Automated Diagnosis and Clinical Decision Support**
> M. E. Ulutas, A. E. Atas, O. Katar, G. Aksoy Eser, Z. Ozpolat, M. Talo, O. Yildirim, R.-S. Tan, U R. Acharya
> *Journal of Bionic Engineering* (under revision, manuscript JBEN-D-25-00926)

A Vision Transformer (ViT-B/16) classifies abdominal CT slices as **complicated hydatid cyst**, **non-complicated hydatid cyst** or **normal**. Grad-CAM, adapted to the ViT, shows the image regions behind each prediction. An ablation study compares the full-slice input with contrast enhancement (CLAHE), data augmentation, body segmentation and a liver region obtained with TotalSegmentator.

## Results (five-fold cross-validation)

| Input | Accuracy (%), mean ± SD | Macro F1-score (%) |
|---|---|---|
| Full slice, no augmentation (**proposed**) | **96.80 ± 0.92** (95% CI 95.65–97.94) | 96.86 ± 0.90 |
| Full slice + CLAHE | 96.97 ± 1.08 | 97.04 ± 1.06 |
| Full slice + data augmentation | 96.14 ± 0.87 | 96.21 ± 0.85 |
| Body-segmented slice | 92.88 ± 2.23 | 92.98 ± 2.18 |
| Liver region (TotalSegmentator) | 86.47 ± 3.73 | 86.53 ± 3.71 |

Per-class F1-scores of the proposed model: complicated 95.92%, non-complicated 94.85%, normal 99.81%. Accuracy over the pooled test images of the five folds: 96.80% (Wilson 95% CI 95.84–97.54%). All numbers are reproduced from the files in `results/` by `src/results.py`.

## Repository layout

```
src/
  config.py         paths (HC_DATA, HC_WORK)
  prepare_data.py   loads the images, caches them and writes the five folds (StratifiedKFold, seed 42)
  train_cv.py       five-fold training and testing of the proposed model and of every ablation variant
  body_masks.py     body segmentation (threshold, closing, hole filling, opening, largest component)
  liver_masks.py    liver segmentation with the TotalSegmentator organ model (ablation only)
  explain.py        Grad-CAM and attention-rollout maps of every held-out image
  results.py        Table 5, ablation table, Figs. 5-7
  fig_gradcam.py    Figs. 8, 9 and 11
  predict.py        classify one slice with a released model and save its Grad-CAM map
  figstyle.py       figure style
results/<run>/fold<k>/  metrics.json, history.csv (training curves), test_predictions.csv
                        (true label, prediction and class probabilities of each test image)
weights_SHA256SUMS.txt  checksums of the released model weights
```

## Data

The dataset is not part of this repository; it **will be made publicly available upon publication of the paper**.

The scripts read numbered 8-bit images from three class folders: `data/komp/` (complicated), `data/nonkomp/` (non-complicated) and `data/normal/` (normal).

## Installation

```bash
conda create -n hc python=3.10.13
conda activate hc
pip install -r requirements.txt
# only for the liver-region ablation, in a separate environment:
pip install -r requirements-liver.txt
```

A CUDA GPU is needed for training (the paper used an NVIDIA GeForce RTX 4090). `predict.py` also runs on a CPU.

## Reproducing the paper

```bash
export HC_DATA=/path/to/data        # folder with komp/, nonkomp/, normal/
export HC_WORK=/path/to/work        # caches, runs, figures and results are written here

python src/prepare_data.py                                   # image cache and the five folds
python src/train_cv.py --exp main_vit --save_weights         # proposed model (lr 1e-5 by default)
python src/train_cv.py --exp abl_clahe --preproc clahe       # ablation: CLAHE
python src/train_cv.py --exp abl_aug --aug                   # ablation: data augmentation
python src/body_masks.py
python src/train_cv.py --exp body_vit --input body           # ablation: body-segmented slice
python src/liver_masks.py all                                # (TotalSegmentator environment)
python src/train_cv.py --exp abl_liver --input liver         # ablation: liver region
python src/results.py                                        # Table 5, ablation table, Figs. 5-7
python src/explain.py                                        # Grad-CAM and attention rollout
python src/fig_gradcam.py                                    # Figs. 8, 9 and 11
```

**Protocol.** ViT-B/16 (`timm` `vit_base_patch16_224`, ImageNet pre-trained), 224 × 224 input with bilinear resizing, AdamW (learning rate 1 × 10⁻⁵, weight decay 0.01), batch size 32, cross-entropy loss, at most 50 epochs, early stopping on the validation loss with a patience of 5; the weights of the epoch with the lowest validation loss are restored and evaluated on the test images of the fold. The learning rate was chosen from 1 × 10⁻⁴, 3 × 10⁻⁵ and 1 × 10⁻⁵ by the mean validation loss of folds 1–2.

To recompute the tables and Figs. 5–7 from the released results without training, copy the run folders: `cp -r results/* $HC_WORK/runs/` and run `python src/results.py`.

## Trained weights

The five fold models of the proposed method are attached to the [v1.0 release](https://github.com/oguzhankatar/Hydatid-Cysts/releases/tag/v1.0) as `hc_vit_b16_fold1.pt` … `hc_vit_b16_fold5.pt` (PyTorch state dicts of `timm` `vit_base_patch16_224` with three output classes, 343 MB each). Check them with `sha256sum -c weights_SHA256SUMS.txt`.

```bash
python src/predict.py --image slice.jpg --weights hc_vit_b16_fold1.pt --out slice_gradcam.png
```

Each fold model was trained on the training images of its own fold. The models are released for research use only and are not a medical device.

## Citation

The citation will be added when the paper is published. Until then, please cite the manuscript title above.

## License

Code: MIT License (see `LICENSE`). Contact: Ozal Yildirim (ozalyildirim@firat.edu.tr), Oguzhan Katar (okatar@firat.edu.tr).
