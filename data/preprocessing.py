"""Modul 1: Data Pipeline & Preprocessing gambar untuk Neural Style Transfer.

Menyediakan fungsi untuk memuat gambar ke tensor ternormalisasi ImageNet,
mengembalikan tensor ke array gambar RGB [0.0, 1.0], menyimpan hasil ke disk,
serta verifikasi batch seluruh dataset ke grid visualisasi 2x3.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np  
from PIL import Image
import torch

# Pastikan root repositori ada di sys.path agar import config.py berhasil
# baik saat dijalankan sebagai modul maupun langsung (python data/preprocessing.py).
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    DEFAULT_IMAGE_SIZE = (384, 512)  # Format (H, W) landscape
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

_RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(
    path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Memuat gambar, resize landscape, normalisasi ImageNet, dan kembalikan tensor (1, 3, H, W)."""
    target_h, target_w = image_size
    target_device = torch.device(device)

    # Buka gambar dan konversi paksa ke RGB (3 channel)
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        # PIL resize memakai urutan (width, height) -> hasil gambar berdimensi (W, H) dengan 3 channel
        resized_img = rgb_img.resize((target_w, target_h), resample=_RESAMPLE_LANCZOS)

    # Konversi PIL Image ke array NumPy float32 rentang [0.0, 1.0] -> (H, W, 3)
    img_np = np.asarray(resized_img, dtype=np.float32) / 255.0

    # Konversi ke tensor PyTorch -> (H, W, 3)
    tensor = torch.from_numpy(img_np)
    # Pindahkan sumbu channel ke depan: (H, W, 3) -> (3, H, W)
    tensor = tensor.permute(2, 0, 1).contiguous()
    # Tambahkan dimensi batch: (3, H, W) -> (1, 3, H, W)
    tensor = tensor.unsqueeze(0).to(device=target_device, dtype=torch.float32)

    # Siapkan mean & std ImageNet untuk broadcasting: (3,) -> (1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=target_device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=target_device).view(1, 3, 1, 1)

    # Normalisasi ImageNet per channel: (1, 3, H, W) -> (1, 3, H, W)
    tensor = (tensor - mean) / std
    return tensor


def deprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """Mengembalikan tensor (1, 3, H, W) ke array NumPy (H, W, 3) dalam rentang [0.0, 1.0]."""
    # Lepas dari graph autograd dan pindahkan ke CPU -> (1, 3, H, W) atau (3, H, W)
    img = tensor.detach().cpu().clone()

    if img.dim() == 4:
        # Ambil elemen batch pertama: (B, 3, H, W) -> (3, H, W)
        img = img[0]
    elif img.dim() != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(img.shape)}")

    # Siapkan mean & std ImageNet untuk denormalisasi: (3,) -> (3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(3, 1, 1)

    # Denormalisasi ImageNet: (3, H, W) -> (3, H, W)
    img = img * std + mean
    # Batasi nilai piksel ke rentang valid [0.0, 1.0]: (3, H, W) -> (3, H, W)
    img = torch.clamp(img, 0.0, 1.0)

    # Ubah urutan dimensi ke format gambar NumPy: (3, H, W) -> (H, W, 3)
    img_np = img.permute(1, 2, 0).numpy()
    return img_np


def save_image(tensor: torch.Tensor, output_path: str | Path) -> None:
    """Menyimpan tensor gambar (1, 3, H, W) ke file gambar dan membuat folder tujuan bila belum ada."""
    # Kembalikan tensor ke NumPy array rentang [0.0, 1.0]: (1, 3, H, W) -> (H, W, 3)
    img_np = deprocess_image(tensor)
    # Skalakan ke uint8 [0, 255]: (H, W, 3) float32 -> (H, W, 3) uint8
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(out_path)


def _list_image_files(folder: Path) -> list[Path]:
    """Mengambil daftar file gambar terurut dari folder dataset."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTENSIONS
    )


if __name__ == "__main__":
    content_dir = _ROOT_DIR / "Dataset" / "Content"
    style_dir = _ROOT_DIR / "Dataset" / "Style"
    output_check_path = _ROOT_DIR / "outputs" / "data_check.png"

    content_paths = _list_image_files(content_dir)
    style_paths = _list_image_files(style_dir)

    print("=== Verifikasi Dataset Content ===")
    content_tensors: list[torch.Tensor] = []
    for path in content_paths:
        # Muat tiap gambar konten -> (1, 3, H, W)
        t = load_image(path, image_size=DEFAULT_IMAGE_SIZE, device="cpu")
        content_tensors.append(t)
        print(f"[Content] {path.name} -> {tuple(t.shape)}")

    print("\n=== Verifikasi Dataset Style ===")
    style_tensors: list[torch.Tensor] = []
    for path in style_paths:
        # Muat tiap gambar gaya -> (1, 3, H, W)
        t = load_image(path, image_size=DEFAULT_IMAGE_SIZE, device="cpu")
        style_tensors.append(t)
        print(f"[Style]   {path.name} -> {tuple(t.shape)}")

    if len(content_tensors) >= 3 and len(style_tensors) >= 3:
        # Gabungkan 3 gambar konten horizontal: 3 x (1, 3, H, W) -> (1, 3, H, 3*W)
        top_row = torch.cat(content_tensors[:3], dim=3)
        # Gabungkan 3 gambar gaya horizontal: 3 x (1, 3, H, W) -> (1, 3, H, 3*W)
        bottom_row = torch.cat(style_tensors[:3], dim=3)
        # Gabungkan baris atas dan baris bawah vertikal: 2 x (1, 3, H, 3*W) -> (1, 3, 2*H, 3*W)
        grid_tensor = torch.cat([top_row, bottom_row], dim=2)

        save_image(grid_tensor, output_check_path)
        print(f"\nGrid 2x3 tersimpan ke: {output_check_path} (tensor shape: {tuple(grid_tensor.shape)})")
    else:
        print(
            f"\nPeringatan: Butuh minimal 3 gambar Content dan 3 gambar Style untuk grid 2x3 "
            f"(ditemukan {len(content_tensors)} content, {len(style_tensors)} style)."
        )
