"""Smoke test dan uji integrasi untuk proyek Neural Style Transfer (VGG-19).

Jalankan dari root proyek:
    python -m tests.smoke_test --device cpu
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from config import (
    CONTENT_DIR,
    CONTENT_IMAGES,
    CONTENT_LAYERS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    EXPERIMENT_PAIRS,
    FAST_IMAGE_SIZE,
    STYLE_DIR,
    STYLE_IMAGES,
    STYLE_LAYERS,
)
from models.vgg_extractor import VGGFeatureExtractor


def resolve_device(device_arg: str) -> torch.device:
    """Validasi dan tentukan device eksekusi PyTorch."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Opsi --device cuda dipilih, tetapi CUDA tidak tersedia di mesin ini."
        )
    return torch.device(device_arg)


def test_dataset_files() -> None:
    """Verifikasi keberadaan direktori dan seluruh file gambar pada Dataset."""
    assert CONTENT_DIR.is_dir(), f"Direktori content tidak ditemukan: {CONTENT_DIR}"
    assert STYLE_DIR.is_dir(), f"Direktori style tidak ditemukan: {STYLE_DIR}"

    all_images: dict[str, Path] = {
        **{f"content:{k}": v for k, v in CONTENT_IMAGES.items()},
        **{f"style:{k}": v for k, v in STYLE_IMAGES.items()},
    }
    for label, path in all_images.items():
        assert path.is_file(), f"File dataset tidak ditemukan ({label}): {path}"
        assert path.stat().st_size > 0, f"File dataset kosong ({label}): {path}"
        with Image.open(path) as img:
            width, height = img.size
            assert width > 0 and height > 0, f"Dimensi gambar tidak valid: {path}"

    assert len(EXPERIMENT_PAIRS) == 3, (
        f"Jumlah pasangan eksperimen harus 3, ditemukan: {len(EXPERIMENT_PAIRS)}"
    )
    for pair in EXPERIMENT_PAIRS:
        content_path = Path(pair["content_path"])
        style_path = Path(pair["style_path"])
        assert content_path.is_file(), f"Content pair hilang: {content_path}"
        assert style_path.is_file(), f"Style pair hilang: {style_path}"

    print(
        f"[PASS] Dataset terverifikasi: {len(CONTENT_IMAGES)} content, "
        f"{len(STYLE_IMAGES)} style, {len(EXPERIMENT_PAIRS)} pasangan eksperimen."
    )


def test_vgg_extractor(device: torch.device, image_size: tuple[int, int]) -> VGGFeatureExtractor:
    """Verifikasi kontrak bentuk output, status beku, mode eval, dan autograd VGGFeatureExtractor."""
    extractor = VGGFeatureExtractor().to(device)

    # 1. Pastikan seluruh parameter dibekukan
    params = list(extractor.parameters())
    assert len(params) > 0, "VGGFeatureExtractor tidak memiliki parameter."
    assert all(not p.requires_grad for p in params), (
        "Ditemukan parameter VGGFeatureExtractor dengan requires_grad=True."
    )

    # 2. Pastikan mode eval() terkunci meski train(True) dipanggil
    assert not extractor.training, "VGGFeatureExtractor harus berada di mode eval()."
    extractor.train(True)
    assert not extractor.training, "Mode eval() pada VGGFeatureExtractor tidak terkunci."

    # 3. Uji forward & backward menggunakan tensor dummy
    h, w = image_size
    dummy_input = torch.randn(1, 3, h, w, device=device, requires_grad=True)
    features = extractor(dummy_input)

    expected_layers = set(CONTENT_LAYERS + STYLE_LAYERS)
    assert set(features.keys()) == expected_layers, (
        f"Layer output tidak sesuai. Harapan={expected_layers}, hasil={set(features.keys())}"
    )

    expected_shapes = {
        "conv1_1": (1, 64, h, w),
        "conv2_1": (1, 128, h // 2, w // 2),
        "conv3_1": (1, 256, h // 4, w // 4),
        "conv4_1": (1, 512, h // 8, w // 8),
        "conv4_2": (1, 512, h // 8, w // 8),
        "conv5_1": (1, 512, h // 16, w // 16),
    }
    for layer_name, feat in features.items():
        assert feat.device.type == device.type, (
            f"Device mismatch pada {layer_name}: {feat.device} != {device}"
        )
        assert torch.isfinite(feat).all(), f"Nilai non-finite ditemukan pada {layer_name}"
        if layer_name in expected_shapes:
            assert tuple(feat.shape) == expected_shapes[layer_name], (
                f"Shape {layer_name} salah: {tuple(feat.shape)} != {expected_shapes[layer_name]}"
            )

    dummy_loss = sum(feat.pow(2).mean() for feat in features.values())
    dummy_loss.backward()
    assert dummy_input.grad is not None, "Gradien pada tensor input gagal dihitung."
    assert dummy_input.grad.shape == dummy_input.shape, "Shape gradien input tidak cocok."
    assert all(p.grad is None for p in extractor.parameters()), (
        "Parameter VGGFeatureExtractor tidak boleh menerima gradien."
    )

    print(
        f"[PASS] VGGFeatureExtractor terverifikasi di {device} "
        f"(input={(1, 3, h, w)}, len(features)={len(extractor.features)}, frozen=True, eval_locked=True)."
    )
    return extractor


def _find_callable(module: Any, candidates: list[str]) -> Callable[..., Any] | None:
    """Cari fungsi yang tersedia pada modul berdasarkan daftar kandidat nama."""
    for name in candidates:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    return None


def _call_with_accepted_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Panggil fungsi hanya dengan keyword argument yang diterima signature-nya."""
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return fn(*args, **kwargs)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered_kwargs)


def test_optional_integration(
    extractor: VGGFeatureExtractor,
    device: torch.device,
    image_size: tuple[int, int],
) -> None:
    """Jalankan uji integrasi jika data/preprocessing.py dan/atau models/losses.py tersedia."""
    h, w = image_size
    content_tensor: torch.Tensor | None = None
    style_tensor: torch.Tensor | None = None

    # 1. Integrasi opsional dengan data/preprocessing.py
    try:
        prep_mod = importlib.import_module("data.preprocessing")
    except ModuleNotFoundError:
        print("[SKIP] data.preprocessing belum tersedia; uji integrasi preprocessing dilewati.")
        prep_mod = None

    if prep_mod is not None:
        loader_fn = _find_callable(
            prep_mod,
            [
                "load_image",
                "preprocess_image",
                "load_and_preprocess",
                "load_tensor",
                "image_to_tensor",
            ],
        )
        if loader_fn is None:
            raise AssertionError(
                "data.preprocessing ditemukan, tetapi fungsi loader tidak dikenali."
            )

        sample_pair = EXPERIMENT_PAIRS[0]
        content_tensor = _call_with_accepted_kwargs(
            loader_fn,
            sample_pair["content_path"],
            size=image_size,
            image_size=image_size,
            target_size=image_size,
            device=device,
        ).to(device)
        style_tensor = _call_with_accepted_kwargs(
            loader_fn,
            sample_pair["style_path"],
            size=image_size,
            image_size=image_size,
            target_size=image_size,
            device=device,
        ).to(device)

        assert tuple(content_tensor.shape) == (1, 3, h, w), (
            f"Shape tensor content salah: {tuple(content_tensor.shape)} != {(1, 3, h, w)}"
        )
        assert tuple(style_tensor.shape) == (1, 3, h, w), (
            f"Shape tensor style salah: {tuple(style_tensor.shape)} != {(1, 3, h, w)}"
        )
        assert torch.isfinite(content_tensor).all() and torch.isfinite(style_tensor).all()

        deprocess_fn = _find_callable(
            prep_mod,
            ["deprocess_image", "deprocess", "tensor_to_image", "tensor_to_pil", "im_convert"],
        )
        if deprocess_fn is not None:
            restored = _call_with_accepted_kwargs(deprocess_fn, content_tensor)
            assert restored is not None, "Fungsi deprocess mengembalikan None."

        print("[PASS] Integrasi data.preprocessing + VGGFeatureExtractor terverifikasi.")

    # 2. Integrasi opsional dengan models/losses.py
    try:
        losses_mod = importlib.import_module("models.losses")
    except ModuleNotFoundError:
        print("[SKIP] models.losses belum tersedia; uji integrasi losses dilewati.")
        return

    if content_tensor is None or style_tensor is None:
        content_tensor = torch.randn(1, 3, h, w, device=device)
        style_tensor = torch.randn(1, 3, h, w, device=device)

    target_tensor = content_tensor.clone().detach().requires_grad_(True)
    with torch.no_grad():
        content_feats = extractor(content_tensor)
        style_feats = extractor(style_tensor)
    target_feats = extractor(target_tensor)

    gram_fn = _find_callable(losses_mod, ["gram_matrix", "compute_gram_matrix"])
    if gram_fn is not None:
        sample_feat = target_feats[STYLE_LAYERS[0]]
        gram = gram_fn(sample_feat)
        c = sample_feat.shape[1]
        assert gram.shape in {(c, c), (1, c, c)}, (
            f"Shape Gram Matrix tidak sesuai harapan {(c, c)} atau {(1, c, c)}: {tuple(gram.shape)}"
        )
        assert torch.isfinite(gram).all(), "Gram Matrix mengandung nilai non-finite."

    total_loss_fn = _find_callable(
        losses_mod, ["total_loss", "compute_total_loss", "nst_loss"]
    )
    content_loss_fn = _find_callable(losses_mod, ["content_loss", "compute_content_loss"])
    style_loss_fn = _find_callable(losses_mod, ["style_loss", "compute_style_loss"])

    if total_loss_fn is not None:
        loss_out = _call_with_accepted_kwargs(
            total_loss_fn,
            target_feats,
            content_feats,
            style_feats,
            alpha=DEFAULT_ALPHA,
            beta=DEFAULT_BETA,
            content_layers=CONTENT_LAYERS,
            style_layers=STYLE_LAYERS,
        )
        total = loss_out[0] if isinstance(loss_out, tuple) else loss_out
        assert torch.is_tensor(total) and total.ndim == 0 and torch.isfinite(total), (
            f"Total loss harus skalar finite, diterima: {total}"
        )
        total.backward()
        assert target_tensor.grad is not None, "Gradien target dari total_loss bernilai None."
    elif content_loss_fn is not None and style_loss_fn is not None:
        c_loss = _call_with_accepted_kwargs(
            content_loss_fn,
            target_feats[CONTENT_LAYERS[0]],
            content_feats[CONTENT_LAYERS[0]],
        )
        s_loss = _call_with_accepted_kwargs(
            style_loss_fn,
            target_feats,
            style_feats,
            style_layers=STYLE_LAYERS,
        )
        total = DEFAULT_ALPHA * c_loss + DEFAULT_BETA * s_loss
        assert torch.isfinite(total), f"Total loss non-finite: {total}"
        total.backward()
        assert target_tensor.grad is not None, "Gradien target dari content+style loss None."
    else:
        raise AssertionError(
            "models.losses ditemukan, tetapi fungsi loss NST tidak dikenali."
        )

    print("[PASS] Integrasi VGGFeatureExtractor + models.losses terverifikasi.")


def parse_args() -> argparse.Namespace:
    """Parse argumen baris perintah untuk smoke test."""
    parser = argparse.ArgumentParser(
        description="Smoke test modul NST (dataset, VGGFeatureExtractor, dan integrasi)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Perangkat eksekusi PyTorch (default: cpu).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=list(FAST_IMAGE_SIZE),
        metavar=("H", "W"),
        help="Resolusi dummy/integrasi dalam format H W (default: 192 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image_size = (args.size[0], args.size[1])

    print(f"=== Menjalankan Smoke Test NST (device={device}, size={image_size}) ===")
    test_dataset_files()
    extractor = test_vgg_extractor(device=device, image_size=image_size)
    test_optional_integration(extractor=extractor, device=device, image_size=image_size)
    print("=== Semua pemeriksaan smoke test selesai dengan sukses ===")


if __name__ == "__main__":
    main()
