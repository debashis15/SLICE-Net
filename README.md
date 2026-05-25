# SLICE-Net

Official PyTorch implementation of **SLICE-Net: Spatial-Frequency Adaptive Selection and Inter-domain Correlation Network for Low-Light Image Enhancement**.

> **Status:** Submitted to **IEEE Sensors Letters**. Citation will be added once available.

---

## Repository structure

```
slice-net/
├── configs/
│   └── default.yaml          # training / model hyperparameters
├── data/
│   ├── __init__.py
│   └── dataset.py            # PairedLLIE, UnpairedLLIE
├── models/
│   ├── __init__.py
│   ├── slicenet.py           # SLICENet, SLICENetConfig, build_model
│   └── modules/
│       ├── dct.py            # block-wise 2D-DCT / IDCT
│       ├── gib.py            # Gradient Interaction Block
│       ├── pmksab.py         # Parallel Multi-Kernel Spatial Attention Block
│       ├── lsfm.py           # Learnable Frequency Selection Module
│       ├── faar.py           # Frequency-Adaptive Attention Refinement
│       ├── idca.py           # Inter-Domain Correlation Attention
│       └── veb.py            # Visual Enhancement Block
└── utils/
    ├── __init__.py
    ├── losses.py             # Charbonnier, Charbonnier + SSIM
    └── metrics.py            # PSNR, SSIM, NIQE
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision, numpy, Pillow, PyYAML
- `pyiqa` (optional, for NIQE)

```bash
pip install torch torchvision numpy Pillow PyYAML pyiqa
```

---

## Quick start

```python
import torch
from models import build_model, SLICENetConfig

cfg = SLICENetConfig(feat_channels=40)
net = build_model(cfg).eval()

low = torch.randn(1, 3, 256, 256)        # low-light input in [0, 1]
with torch.no_grad():
    enhanced = net(low)                  # enhanced output
```

---

## Dataset layout

The `PairedLLIE` dataset expects matched low / high pairs by filename stem:

```
<dataset_root>/
├── low/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── high/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

For NIQE-only evaluation on unpaired benchmarks (DICM, LIME, MEF, NPE), use `UnpairedLLIE` with a flat folder of images.

---

## Pretrained weights & results

Pretrained checkpoints and visual results are hosted on Google Drive:

| Resource | Link |
|---|---|
| Pretrained weights (LOL v1) | [Google Drive](<INSERT_LINK_HERE>) |
| Pretrained weights (LOL v2) | [Google Drive](<INSERT_LINK_HERE>) |
| Pretrained weights (MIT-Adobe 5K) | [Google Drive](<INSERT_LINK_HERE>) |
| Visual results (paired benchmarks) | [Google Drive](<INSERT_LINK_HERE>) |
| Visual results (unpaired benchmarks) | [Google Drive](<INSERT_LINK_HERE>) |

> Replace `<INSERT_LINK_HERE>` with the corresponding shareable Google Drive URL.

To load a downloaded checkpoint:

```python
import torch
from models import build_model

net = build_model()
state = torch.load("slicenet_lolv1.pth", map_location="cpu")
net.load_state_dict(state["model"] if "model" in state else state)
net.eval()
```

---

## Citation

> *To be updated upon acceptance at IEEE Sensors Letters.*

```bibtex
@article{slicenet2026,
  title   = {SLICE-Net: Spatial-Frequency Adaptive Selection and Inter-domain Correlation Network for Low-Light Image Enhancement},
  author  = {Anonymous},
  journal = {IEEE Sensors Letters (under review)},
  year    = {2026}
}
```

---

## License

MIT
