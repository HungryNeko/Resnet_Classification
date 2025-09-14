# Resnet_Classification

Land-cover classification on the multi-spectral EuroSAT dataset, comparing a plain ResNet-18 with one trained by knowledge distillation from a ResNet-50 teacher.

## Results

| Model | Training | Test accuracy |
| --- | --- | --- |
| ResNet-18 | baseline, no augmentation | 89.57% |
| ResNet-50 (teacher) | 20 epochs | 93.09% |
| ResNet-18 (student) | augmentation + distillation, 10 epochs | **96.65%** |

The distilled student outperforms its own teacher on this split.

## Setup

- **Data:** [EuroSAT](https://github.com/phelber/EuroSAT) multi-spectral version (`EuroSAT_MS`, 13 Sentinel-2 bands, 10 classes). Not included in the repo.
- **Split:** stratified 80/20 train/test, fixed seed, indices saved in `splits/`.
- **Model:** torchvision ResNets with the first convolution changed to take 13 x 32 x 32 input.
- **Distillation:** cross-entropy plus KL divergence on softened logits (T = 4, alpha = 0.5).
- **Augmentation:** random flips and a small random spectral scaling.

## Files

| File | Purpose |
| --- | --- |
| `samedata.py` | Create the fixed train/test split |
| `dataset.py` | Dataset class for the 13-band GeoTIFFs |
| `net.py` | Baseline ResNet-18 |
| `teacher.py` | Train the ResNet-50 teacher |
| `net_pro.py` | Train the ResNet-18 student with distillation |
| `showtif.py` | Preview a sample as an RGB image |
| `report.pdf` | Short write-up |

## Run

```bash
pip install torch torchvision rasterio tqdm matplotlib
# put the dataset in ./EuroSAT_MS
python samedata.py
python net.py        # baseline
python teacher.py    # teacher
python net_pro.py    # distilled student
```

Trained weights (`*.pth`) are included.

*Course project, September 2025.*
