"""Modul 3: Skrip training end-to-end Neural Style Transfer (NST).

Mengintegrasikan pipeline preprocessing (`data/preprocessing.py`), ekstraktor
fitur VGG-19 beku (`models/vgg_extractor.py`), dan fungsi loss manual
(`models/losses.py`) untuk mengoptimasi piksel gambar target secara langsung.

Contoh penggunaan:
    python train.py --output_dir outputs/exp_test --steps 300
    python train.py --optimizer lbfgs --init noise --steps 100 --save_every 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from config import (
    CONTENT_LAYERS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    EXPERIMENT_PAIRS,
    FINAL_IMAGE_SIZE,
    OUTPUT_DIR,
    STYLE_LAYERS,
)
from data.preprocessing import load_image, save_image
from models.losses import compute_total_loss
from models.vgg_extractor import VGGFeatureExtractor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Mem-parsing argumen baris perintah untuk eksperimen NST."""
    default_pair = EXPERIMENT_PAIRS[0]
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Training Neural Style Transfer (VGG-19) berbasis optimasi piksel."
    )
    parser.add_argument(
        "--content",
        type=str,
        default=str(default_pair["content_path"]),
        help="Path gambar konten (default: pasangan pertama di config.py).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=str(default_pair["style_path"]),
        help="Path gambar gaya (default: pasangan pertama di config.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "exp_default"),
        help="Direktori penyimpanan hasil run (otomatis dibuat jika belum ada).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Jumlah iterasi optimasi (default: 300).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Learning rate optimizer (default: 0.02 untuk Adam).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Bobot content loss (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"Bobot style loss (default: {DEFAULT_BETA}).",
    )
    parser.add_argument(
        "--optimizer",
        type=str.lower,
        choices=["adam", "lbfgs"],
        default="adam",
        help="Pilihan optimizer: 'adam' atau 'lbfgs' (default: adam).",
    )
    parser.add_argument(
        "--init",
        type=str.lower,
        choices=["content", "noise"],
        default="content",
        help="Inisialisasi gambar target: 'content' atau 'noise' (default: content).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=FINAL_IMAGE_SIZE[0],
        help=f"Tinggi gambar target dalam piksel (default: {FINAL_IMAGE_SIZE[0]}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=FINAL_IMAGE_SIZE[1],
        help=f"Lebar gambar target dalam piksel (default: {FINAL_IMAGE_SIZE[1]}).",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        choices=["cuda", "cpu"],
        default=default_device,
        help=f"Perangkat eksekusi PyTorch: 'cuda' atau 'cpu' (default: {default_device}).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Interval cetak progres loss ke terminal (default: 50; 0 untuk nonaktif).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Interval simpan snapshot progres tiap N step (default: 0, hanya hasil akhir).",
    )
    return parser.parse_args(argv)


def resolve_device(device_name: str) -> torch.device:
    """Memvalidasi dan mengembalikan `torch.device` sesuai pilihan pengguna."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Argumen --device cuda dipilih, tetapi CUDA tidak tersedia di sistem ini."
        )
    return torch.device(device_name)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Memvalidasi path file input dan rentang nilai argumen CLI."""
    content_path = Path(args.content)
    style_path = Path(args.style)
    output_dir = Path(args.output_dir)

    if not content_path.is_file():
        raise FileNotFoundError(f"File gambar konten tidak ditemukan: {content_path}")
    if not style_path.is_file():
        raise FileNotFoundError(f"File gambar gaya tidak ditemukan: {style_path}")
    if args.steps <= 0:
        raise ValueError(f"--steps harus bernilai positif, diterima: {args.steps}")
    if args.lr <= 0:
        raise ValueError(f"--lr harus bernilai positif, diterima: {args.lr}")
    if args.height <= 0 or args.width <= 0:
        raise ValueError(
            f"--height dan --width harus positif, diterima: ({args.height}, {args.width})"
        )
    if args.log_every < 0:
        raise ValueError(f"--log_every tidak boleh negatif, diterima: {args.log_every}")
    if args.save_every < 0:
        raise ValueError(f"--save_every tidak boleh negatif, diterima: {args.save_every}")

    return content_path, style_path, output_dir


def initialize_target(content_img: torch.Tensor, init_mode: str) -> torch.Tensor:
    """Menginisialisasi tensor piksel target (1, 3, H, W) dengan `requires_grad=True`."""
    if init_mode == "content":
        # Kloning tensor gambar konten sebagai daun graf autograd: (1, 3, H, W) -> (1, 3, H, W)
        target = content_img.clone().detach().requires_grad_(True)
    elif init_mode == "noise":
        # Bangkitkan white noise Gaussian berdimensi sama dengan konten: (1, 3, H, W)
        target = torch.randn_like(content_img, requires_grad=True)
    else:
        raise ValueError(f"Mode inisialisasi tidak dikenal: {init_mode}")
    return target


def build_optimizer(
    target: torch.Tensor,
    optimizer_name: str,
    lr: float,
) -> torch.optim.Optimizer:
    """Membuat optimizer yang HANYA mengoptimasi parameter piksel `[target]`."""
    if optimizer_name == "adam":
        return torch.optim.Adam([target], lr=lr)
    if optimizer_name == "lbfgs":
        # max_iter=1 agar satu panggilan optimizer.step(closure) setara satu step pada loop
        return torch.optim.LBFGS([target], lr=lr, max_iter=1)
    raise ValueError(f"Optimizer tidak dikenal: {optimizer_name}")


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Menjalankan proses optimasi NST secara end-to-end dan menyimpan seluruh artefak."""
    content_path, style_path, output_dir = validate_args(args)
    device = resolve_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.height, args.width)
    content_layer = CONTENT_LAYERS[0]
    style_weights = {layer: 1.0 / len(STYLE_LAYERS) for layer in STYLE_LAYERS}

    # 1. Muat gambar konten dan gaya -> masing-masing tensor (1, 3, H, W) pada device
    content_img = load_image(content_path, image_size=image_size, device=device)
    style_img = load_image(style_path, image_size=image_size, device=device)

    # 2. Inisialisasi VGGFeatureExtractor dengan bobot beku dan mode eval()
    extractor = VGGFeatureExtractor().to(device)
    for param in extractor.parameters():
        param.requires_grad_(False)
    extractor.eval()

    # 3. Ekstrak fitur referensi konten & gaya tanpa gradien -> dict[str, (1, C_l, H_l, W_l)]
    with torch.no_grad():
        content_features = {
            layer: feat.detach() for layer, feat in extractor(content_img).items()
        }
        style_features = {
            layer: feat.detach() for layer, feat in extractor(style_img).items()
        }

    # 4. Inisialisasi tensor target yang akan dioptimasi -> (1, 3, H, W)
    target = initialize_target(content_img, args.init)

    # 5. Inisialisasi optimizer khusus untuk piksel [target] (bukan parameter VGG)
    optimizer = build_optimizer(target, args.optimizer, args.lr)

    metrics_path = output_dir / "metrics.csv"
    result_path = output_dir / "result.png"
    config_path = output_dir / "run_config.json"

    print(
        f"Memulai training NST | device={device} | optimizer={args.optimizer} | "
        f"init={args.init} | size={image_size} | steps={args.steps} | "
        f"alpha={args.alpha} | beta={args.beta}"
    )

    initial_losses: dict[str, float] | None = None
    final_losses: dict[str, float] = {}
    start_time = time.perf_counter()

    def _compute_step_losses() -> tuple[torch.Tensor, float, float, float]:
        optimizer.zero_grad()
        # Ekstrak aktivasi target -> dict[str, (1, C_l, H_l, W_l)]
        target_features = extractor(target)
        # Hitung total, content, dan style loss -> masing-masing tensor skalar ()
        total_loss, content_loss, style_loss = compute_total_loss(
            target_features=target_features,
            content_features=content_features,
            style_features=style_features,
            alpha=args.alpha,
            beta=args.beta,
            content_layer=content_layer,
            style_weights=style_weights,
        )
        total_loss.backward()
        return (
            total_loss,
            float(total_loss.item()),
            float(content_loss.item()),
            float(style_loss.item()),
        )

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "total_loss", "content_loss", "style_loss", "elapsed_sec"])

        for step in range(1, args.steps + 1):
            if args.optimizer == "adam":
                _, total_val, content_val, style_val = _compute_step_losses()
                optimizer.step()
            else:
                step_losses: dict[str, float] = {}

                def closure() -> torch.Tensor:
                    total_loss, total_f, content_f, style_f = _compute_step_losses()
                    step_losses["total_loss"] = total_f
                    step_losses["content_loss"] = content_f
                    step_losses["style_loss"] = style_f
                    return total_loss

                optimizer.step(closure)
                total_val = step_losses["total_loss"]
                content_val = step_losses["content_loss"]
                style_val = step_losses["style_loss"]

            elapsed_sec = time.perf_counter() - start_time
            writer.writerow(
                [
                    step,
                    f"{total_val:.6f}",
                    f"{content_val:.6f}",
                    f"{style_val:.6f}",
                    f"{elapsed_sec:.4f}",
                ]
            )

            current_losses = {
                "total_loss": total_val,
                "content_loss": content_val,
                "style_loss": style_val,
            }
            if initial_losses is None:
                initial_losses = current_losses
            final_losses = current_losses

            if args.log_every > 0 and (
                step == 1 or step % args.log_every == 0 or step == args.steps
            ):
                print(
                    f"[Step {step:>4}/{args.steps}] "
                    f"total={total_val:.4f} | content={content_val:.4f} | "
                    f"style={style_val:.6f} | elapsed={elapsed_sec:.2f}s"
                )

            if args.save_every > 0 and step % args.save_every == 0:
                snapshot_path = output_dir / f"step_{step:04d}.png"
                # Deproses & simpan snapshot tensor (1, 3, H, W) -> gambar (H, W, 3)
                save_image(target, snapshot_path)

    total_time_sec = time.perf_counter() - start_time
    avg_sec_per_step = total_time_sec / args.steps

    # Simpan hasil akhir target (1, 3, H, W) ke <output_dir>/result.png
    save_image(target, result_path)

    run_config: dict[str, object] = {
        "cli_args": {
            "content": str(content_path),
            "style": str(style_path),
            "output_dir": str(output_dir),
            "steps": args.steps,
            "lr": args.lr,
            "alpha": args.alpha,
            "beta": args.beta,
            "optimizer": args.optimizer,
            "init": args.init,
            "height": args.height,
            "width": args.width,
            "device": args.device,
            "log_every": args.log_every,
            "save_every": args.save_every,
        },
        "device_used": str(device),
        "content_layers": list(CONTENT_LAYERS),
        "style_layers": list(STYLE_LAYERS),
        "style_weights": style_weights,
        "total_time_sec": round(total_time_sec, 4),
        "avg_sec_per_step": round(avg_sec_per_step, 6),
        "initial_loss": initial_losses,
        "final_loss": final_losses,
        "outputs": {
            "result_image": str(result_path),
            "metrics_csv": str(metrics_path),
            "run_config_json": str(config_path),
        },
    }
    if device.type == "cuda":
        run_config["cuda_device_name"] = torch.cuda.get_device_name(device)

    with config_path.open("w", encoding="utf-8") as json_file:
        json.dump(run_config, json_file, indent=2)

    assert initial_losses is not None
    print("\n=== Ringkasan Training NST ===")
    print(f"Total waktu        : {total_time_sec:.2f} detik")
    print(f"Rata-rata per step : {avg_sec_per_step:.4f} detik/step")
    print(
        f"Loss awal (step 1) : total={initial_losses['total_loss']:.4f}, "
        f"content={initial_losses['content_loss']:.4f}, "
        f"style={initial_losses['style_loss']:.6f}"
    )
    print(
        f"Loss akhir (step {args.steps}): total={final_losses['total_loss']:.4f}, "
        f"content={final_losses['content_loss']:.4f}, "
        f"style={final_losses['style_loss']:.6f}"
    )
    print(f"Output gambar      : {result_path}")
    print(f"Output metrik CSV  : {metrics_path}")
    print(f"Output konfigurasi : {config_path}")

    return run_config


def main() -> None:
    """Entrypoint utama skrip CLI training NST."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
