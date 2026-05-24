# SLICE-Net: Spatial-Frequency Adaptive Selection and Inter-domain Correlation Network for Low-Light Image Enhancement

This repository contains a complete PyTorch implementation of **SLICE-Net** for low-light image enhancement.

SLICE-Net is designed around five core components:

1. **Gradient Interaction Block (GIB)** for structure-sensitive gradient guidance.
2. **Gradient-Embedded Spatial Domain Pipeline (GESDP)** using cascaded **Parallel Multi-Kernel Spatial Attention Blocks (PMKSAB)**.
3. **Learnable Frequency Representation Pipeline (LFRP)** using patch-wise 2D-DCT, **Learnable Frequency Selection Module (LSFM)**, **Frequency Adaptive Attention Refinement (FAAR)**, and inverse 2D-DCT.
4. **Inter-Domain Correlation Attention (IDCA)** for explicit spatial-frequency correlation modeling.
5. **Visual Enhancement Block (VEB)** for residual restoration.

<p align="center">
  <img src="assets/slicenet_architecture.png" width="900">
</p>

---

## Repository structure

```text
SLICE-Net-GitHub/
├── assets/
│   └── slicenet_architecture.png
├── configs/
│   └── slicenet_lol.yml
├── datasets/
│   └── paired_image_dataset.py
├── models/
│   ├── blocks.py
│   ├── slicenet.py
│   └── __init__.py
├── scripts/
│   └── prepare_lol_dataset.md
├── utils/
│   ├── image.py
│   ├── io.py
│   ├── losses.py
│   ├── metrics.py
│   └── seed.py
├── train.py
├── test.py
├── inference.py
├── profile_model.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-user/SLICE-Net.git
cd SLICE-Net
conda create -n slicenet python=3.10 -y
conda activate slicenet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

---

## Dataset preparation

The dataloader expects paired low-light and normal-light images with matching filenames.

Example for LOL-v1:

```text
data/LOL-v1/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

Update the paths in:

```bash
configs/slicenet_lol.yml
```

---

## Training

```bash
python train.py --config configs/slicenet_lol.yml
```

Default training settings follow the manuscript-style setup:

- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-7`
- Betas: `(0.9, 0.999)`
- Loss: Charbonnier loss
- Epochs: `600`
- Mixed precision: enabled by default

The best checkpoint is saved at:

```text
checkpoints/best.pth
```

Resume training:

```bash
python train.py --config configs/slicenet_lol.yml --resume checkpoints/epoch_0100.pth
```

---

## Testing

```bash
python test.py \
  --config configs/slicenet_lol.yml \
  --checkpoint checkpoints/best.pth \
  --save_images
```

Enhanced images will be saved in:

```text
results/test_outputs/
```

The testing script reports average PSNR and SSIM.

---

## Single-image or folder inference

```bash
python inference.py \
  --input path/to/low_light_image_or_folder \
  --checkpoint checkpoints/best.pth \
  --output results/inference
```

---

## Model profiling

```bash
python profile_model.py --size 256 --base_channels 32
```

This reports parameter count and FLOPs for a dummy input.

---

## Method overview

Given a low-light image, SLICE-Net extracts structure-aware spatial features through a gradient interaction branch and cascaded multi-kernel spatial attention blocks. In parallel, it processes the input through a learnable frequency pipeline based on patch-wise DCT. The low- and high-frequency components are adaptively selected and refined using LSFM and FAAR. Finally, IDCA models the correlation between spatial and frequency representations, and VEB predicts a residual degradation map for enhancement.

---

## Important notes

This repository provides a clean and reproducible implementation based on the architecture description. Exact reported benchmark numbers may vary depending on dataset split, preprocessing, crop size, seed, training duration, and hardware.

For fair comparison, use the same train/test split, image resolution policy, and evaluation protocol across all methods.

---

## Citation

```bibtex
@article{das2026slicenet,
  title={SLICE-Net: Spatial-Frequency Adaptive Selection and Inter-domain Correlation Network for Low-Light Image Enhancement},
  author={Das, Debashis and Maji, Suman Kumar},
  journal={IEEE Sensors Applications},
  year={2026}
}
```

---

## Contact

For questions, please contact:

```text
Debashis Das
Department of Computer Science and Engineering
Indian Institute of Technology Patna
Email: debashis_2221cs31@iitp.ac.in
```
