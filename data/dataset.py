from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
try:
    from PIL import Image
except ImportError as e:
    raise ImportError('Pillow is required to load images. `pip install Pillow`.') from e
_IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def _list_images(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXT))

def _load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = torch.from_numpy(_pil_to_array(img))
    return arr.permute(2, 0, 1).float() / 255.0

def _pil_to_array(img):
    import numpy as np
    return np.asarray(img, dtype='uint8')

class PairedLLIE(Dataset):
    def __init__(self, root: str | os.PathLike, patch_size: Optional[int]=256, train: bool=True, low_dir: str='low', high_dir: str='high') -> None:
        self.root = Path(root)
        self.train = train
        self.patch_size = patch_size
        low_folder = self.root / low_dir
        high_folder = self.root / high_dir
        if not low_folder.is_dir() or not high_folder.is_dir():
            raise FileNotFoundError(f"Expected '{low_folder}' and '{high_folder}' to exist. See the project README for the dataset layout.")
        lows = _list_images(low_folder)
        highs = _list_images(high_folder)
        high_by_stem = {h.stem: h for h in highs}
        self.pairs: list[tuple[Path, Path]] = []
        for low_path in lows:
            high_path = high_by_stem.get(low_path.stem)
            if high_path is None:
                continue
            self.pairs.append((low_path, high_path))
        if not self.pairs:
            raise RuntimeError(f'No matching pairs found under {self.root}.')

    def __len__(self) -> int:
        return len(self.pairs)

    def _random_crop(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, H, W = low.shape
        ps = self.patch_size
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            low = torch.nn.functional.pad(low.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            high = torch.nn.functional.pad(high.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
            _, H, W = low.shape
        y = random.randint(0, H - ps)
        x = random.randint(0, W - ps)
        return (low[:, y:y + ps, x:x + ps], high[:, y:y + ps, x:x + ps])

    def _augment(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            low = torch.flip(low, dims=[2])
            high = torch.flip(high, dims=[2])
        if random.random() < 0.5:
            low = torch.flip(low, dims=[1])
            high = torch.flip(high, dims=[1])
        k = random.randint(0, 3)
        if k:
            low = torch.rot90(low, k=k, dims=(1, 2))
            high = torch.rot90(high, k=k, dims=(1, 2))
        return (low, high)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        low_path, high_path = self.pairs[idx]
        low = _load_rgb(low_path)
        high = _load_rgb(high_path)
        if self.train and self.patch_size is not None:
            low, high = self._random_crop(low, high)
            low, high = self._augment(low, high)
        elif self.patch_size is not None:
            _, H, W = low.shape
            ps = self.patch_size
            y = max(0, (H - ps) // 2)
            x = max(0, (W - ps) // 2)
            low = low[:, y:y + ps, x:x + ps]
            high = high[:, y:y + ps, x:x + ps]
        return {'low': low, 'high': high, 'name': low_path.name}

class UnpairedLLIE(Dataset):
    def __init__(self, root: str | os.PathLike, transform: Callable[[torch.Tensor], torch.Tensor] | None=None) -> None:
        self.root = Path(root)
        self.images = _list_images(self.root)
        if not self.images:
            raise RuntimeError(f'No images found under {self.root}.')
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.images[idx]
        img = _load_rgb(path)
        if self.transform is not None:
            img = self.transform(img)
        return {'low': img, 'name': path.name}
