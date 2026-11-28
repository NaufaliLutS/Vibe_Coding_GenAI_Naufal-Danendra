"""Pembuat figur laporan hasil eksperimen Neural Style Transfer (VGG-19).

Skrip ini membaca hasil eksperimen dari 6 subfolder di `outputs/`:
    - pair1_gedung_composition
    - pair2_danau_starry
    - pair3_rumah_greatwave
    - tuning_beta_1e3
    - tuning_beta_1e5
    - tuning_beta_1e7

Menghasilkan figur siap laporan di `outputs/figures/`:
    1. loss_curves_pairs.png       : Kurva content_loss, style_loss, total_loss (3 pasangan)
    2. loss_curves_tuning_beta.png : Kurva perbandingan loss untuk variasi beta
    3. grid_main_pairs.png         : Grid Content - Style - Output untuk 3 pasangan utama
    4. grid_beta_tuning.png        : Grid perbandingan hasil tuning variasi beta

Contoh penggunaan:
    python -m scripts.make_figures
    python -m scripts.make_figures --mode dummy --output-dir outputs/figures_dummy
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from PIL import Image, ImageDraw

# Pastikan root proyek berada pada sys.path saat skrip dijalankan langsung
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CONTENT_IMAGES,
    DEFAULT_ALPHA,
    FINAL_IMAGE_SIZE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    STYLE_IMAGES,
)

REQUIRED_METRICS_COLUMNS: tuple[str, ...] = (
    "step",
    "content_loss",
    "style_loss",
    "total_loss",
)

_RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)

MAIN_PAIR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "folder": "pair1_gedung_composition",
        "title": "Pair 1: Gedung + Composition VII",
        "content_path": CONTENT_IMAGES["gedung"],
        "style_path": STYLE_IMAGES["composition_vii"],
    },
    {
        "folder": "pair2_danau_starry",
        "title": "Pair 2: Danau + The Starry Night",
        "content_path": CONTENT_IMAGES["danau"],
        "style_path": STYLE_IMAGES["the_starry_night"],
    },
    {
        "folder": "pair3_rumah_greatwave",
        "title": "Pair 3: Rumah Bukit + The Great Wave",
        "content_path": CONTENT_IMAGES["rumah_bukit"],
        "style_path": STYLE_IMAGES["the_great_wave"],
    },
)

BETA_TUNING_SPECS: tuple[dict[str, Any], ...] = (
    {
        "folder": "tuning_beta_1e3",
        "label": "β = 1e3",
        "beta": 1e3,
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "folder": "tuning_beta_1e5",
        "label": "β = 1e5",
        "beta": 1e5,
        "color": "#d62728",
        "linestyle": "--",
    },
    {
        "folder": "tuning_beta_1e7",
        "label": "β = 1e7",
        "beta": 1e7,
        "color": "#2ca02c",
        "linestyle": "-.",
    },
)


@dataclass(frozen=True)
class ExperimentRecord:
    """Representasi terstruktur satu hasil eksperimen di `outputs/`."""

    folder: str
    title: str
    content_path: Path
    style_path: Path
    result_path: Path
    metrics: dict[str, np.ndarray]
    alpha: float
    beta: float
    steps: int


def load_metrics_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Memuat dan memvalidasi file `metrics.csv` sesuai kontrak kolom wajib."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File metrics.csv tidak ditemukan: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Header CSV kosong pada file: {csv_path}")

        field_set = {name.strip() for name in reader.fieldnames if name is not None}
        missing = [col for col in REQUIRED_METRICS_COLUMNS if col not in field_set]
        if missing:
            raise ValueError(
                f"Kolom wajib {missing} tidak ditemukan di {csv_path}. "
                f"Kolom tersedia: {sorted(field_set)}"
            )

        rows: list[tuple[int, float, float, float]] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                step = int(float(row["step"]))
                content_loss = float(row["content_loss"])
                style_loss = float(row["style_loss"])
                total_loss = float(row["total_loss"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Nilai numerik tidak valid di {csv_path} baris {line_no}: {row}"
                ) from exc
            rows.append((step, content_loss, style_loss, total_loss))

    if not rows:
        raise ValueError(f"Tidak ada baris data pada file metrics.csv: {csv_path}")

    rows.sort(key=lambda item: item[0])
    steps_arr = np.array([r[0] for r in rows], dtype=np.int64)
    content_arr = np.array([r[1] for r in rows], dtype=np.float64)
    style_arr = np.array([r[2] for r in rows], dtype=np.float64)
    total_arr = np.array([r[3] for r in rows], dtype=np.float64)

    return {
        "step": steps_arr,
        "content_loss": content_arr,
        "style_loss": style_arr,
        "total_loss": total_arr,
    }


def _resolve_repo_path(raw_path: str | Path, base_root: Path) -> Path:
    """Mengubah path relatif/Windows dari run_config.json menjadi Path valid."""
    normalized = str(raw_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return base_root / candidate


def _load_experiment_record(
    run_dir: Path,
    title: str,
    fallback_content: Path,
    fallback_style: Path,
    fallback_beta: float,
    base_root: Path = PROJECT_ROOT,
) -> ExperimentRecord:
    """Memuat metrik, path gambar, dan metadata satu subfolder eksperimen."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Folder eksperimen tidak ditemukan: {run_dir}")

    metrics_path = run_dir / "metrics.csv"
    result_path = run_dir / "result.png"
    config_path = run_dir / "run_config.json"

    if not result_path.is_file():
        raise FileNotFoundError(f"Gambar hasil result.png tidak ditemukan: {result_path}")

    metrics = load_metrics_csv(metrics_path)

    content_path = fallback_content
    style_path = fallback_style
    alpha = DEFAULT_ALPHA
    beta = fallback_beta
    steps = int(metrics["step"][-1])

    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            run_cfg = json.load(handle)
        cli_args = run_cfg.get("cli_args", {})
        if cli_args.get("content"):
            resolved_content = _resolve_repo_path(cli_args["content"], base_root)
            if resolved_content.is_file():
                content_path = resolved_content
        if cli_args.get("style"):
            resolved_style = _resolve_repo_path(cli_args["style"], base_root)
            if resolved_style.is_file():
                style_path = resolved_style
        alpha = float(cli_args.get("alpha", alpha))
        beta = float(cli_args.get("beta", beta))
        steps = int(cli_args.get("steps", steps))

    if not content_path.is_file():
        raise FileNotFoundError(f"Gambar content tidak ditemukan: {content_path}")
    if not style_path.is_file():
        raise FileNotFoundError(f"Gambar style tidak ditemukan: {style_path}")

    return ExperimentRecord(
        folder=run_dir.name,
        title=title,
        content_path=content_path,
        style_path=style_path,
        result_path=result_path,
        metrics=metrics,
        alpha=alpha,
        beta=beta,
        steps=steps,
    )


def load_all_experiments(
    outputs_root: Path,
    base_root: Path = PROJECT_ROOT,
) -> tuple[list[ExperimentRecord], list[ExperimentRecord]]:
    """Memuat seluruh 6 subfolder eksperimen (3 pasangan utama + 3 tuning beta)."""
    main_records: list[ExperimentRecord] = []
    for spec in MAIN_PAIR_SPECS:
        run_dir = outputs_root / spec["folder"]
        main_records.append(
            _load_experiment_record(
                run_dir=run_dir,
                title=spec["title"],
                fallback_content=spec["content_path"],
                fallback_style=spec["style_path"],
                fallback_beta=1e5,
                base_root=base_root,
            )
        )

    tuning_records: list[ExperimentRecord] = []
    default_content = CONTENT_IMAGES["gedung"]
    default_style = STYLE_IMAGES["composition_vii"]
    for spec in BETA_TUNING_SPECS:
        run_dir = outputs_root / spec["folder"]
        tuning_records.append(
            _load_experiment_record(
                run_dir=run_dir,
                title=spec["label"],
                fallback_content=default_content,
                fallback_style=default_style,
                fallback_beta=float(spec["beta"]),
                base_root=base_root,
            )
        )

    return main_records, tuning_records


def _load_display_image(image_path: Path, target_size_wh: tuple[int, int]) -> np.ndarray:
    """Membuka gambar RGB dan menyesuaikannya ke dimensi tampilan seragam (W, H)."""
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        if rgb_img.size != target_size_wh:
            rgb_img = rgb_img.resize(target_size_wh, resample=_RESAMPLE_LANCZOS)
        return np.asarray(rgb_img)


def _format_axis(ax: plt.Axes, use_log: bool = False) -> None:
    """Merapikan grid, skala sumbu Y, dan tampilan spine subplot Matplotlib."""
    if use_log:
        ax.set_yscale("log")
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main_pair_losses(
    main_records: list[ExperimentRecord],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Membuat plot kurva content_loss, style_loss, dan total_loss untuk 3 pasangan utama."""
    fig, axes = plt.subplots(
        nrows=len(main_records),
        ncols=3,
        figsize=(14.5, 10.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(main_records) == 1:
        axes = np.expand_dims(axes, axis=0)

    metric_specs = (
        ("content_loss", "Content Loss", "#1f77b4", False),
        ("style_loss", "Style Loss", "#d62728", True),
        ("total_loss", "Total Loss", "#2ca02c", True),
    )

    for row_idx, record in enumerate(main_records):
        steps = record.metrics["step"]
        for col_idx, (metric_key, metric_label, color, prefer_log) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = record.metrics[metric_key]
            use_log = prefer_log and bool(np.all(values > 0))

            ax.plot(steps, values, color=color, linewidth=1.8)
            _format_axis(ax, use_log=use_log)

            final_val = float(values[-1])
            ax.annotate(
                f"akhir: {final_val:.4g}",
                xy=(float(steps[-1]), final_val),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            )

            if row_idx == 0:
                scale_note = " (skala log)" if use_log else ""
                ax.set_title(f"{metric_label}{scale_note}", fontsize=11.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{record.title}\nLoss", fontsize=9.5, fontweight="bold")
            if row_idx == len(main_records) - 1:
                ax.set_xlabel("Step Optimasi", fontsize=10)

    fig.suptitle(
        "Kurva Konvergensi Loss — 3 Pasangan Eksperimen Utama (384×512, α = 1.0, β = 1e5)",
        fontsize=13.5,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_beta_tuning_losses(
    tuning_records: list[ExperimentRecord],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Membuat plot kurva perbandingan content_loss, style_loss, dan total_loss variasi beta."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(14.5, 4.4),
        constrained_layout=True,
    )

    metric_specs = (
        ("content_loss", "Content Loss", False),
        ("style_loss", "Style Loss", True),
        ("total_loss", "Total Loss", True),
    )

    for col_idx, (metric_key, metric_label, prefer_log) in enumerate(metric_specs):
        ax = axes[col_idx]
        can_log = prefer_log and all(
            bool(np.all(rec.metrics[metric_key] > 0)) for rec in tuning_records
        )

        for spec, rec in zip(BETA_TUNING_SPECS, tuning_records):
            steps = rec.metrics["step"]
            values = rec.metrics[metric_key]
            ax.plot(
                steps,
                values,
                label=f"{spec['label']} (akhir: {values[-1]:.3g})",
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=1.9,
            )

        _format_axis(ax, use_log=can_log)
        scale_note = " (skala log)" if can_log else ""
        ax.set_title(f"{metric_label}{scale_note}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Step Optimasi", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Nilai Loss", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8.5, frameon=True, loc="best")

    fig.suptitle(
        "Perbandingan Kurva Loss pada Variasi Bobot Gaya β ∈ {1e3, 1e5, 1e7} (Gedung + Composition VII)",
        fontsize=13,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_main_pairs_grid(
    main_records: list[ExperimentRecord],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Membuat grid visual Content — Style — Output untuk 3 pasangan eksperimen utama."""
    target_wh = (FINAL_IMAGE_SIZE[1], FINAL_IMAGE_SIZE[0])  # (W, H) = (512, 384)
    fig, axes = plt.subplots(
        nrows=len(main_records),
        ncols=3,
        figsize=(13.5, 10.5),
        constrained_layout=True,
    )
    if len(main_records) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_headers = ("Content", "Style", "Output (Stylized)")

    for row_idx, rec in enumerate(main_records):
        content_img = _load_display_image(rec.content_path, target_wh)
        style_img = _load_display_image(rec.style_path, target_wh)
        output_img = _load_display_image(rec.result_path, target_wh)

        c_final = float(rec.metrics["content_loss"][-1])
        s_final = float(rec.metrics["style_loss"][-1])
        t_final = float(rec.metrics["total_loss"][-1])

        panels = (
            (content_img, f"{rec.content_path.name}"),
            (style_img, f"{rec.style_path.name}"),
            (
                output_img,
                f"Step {rec.steps} | C={c_final:.2f}, S={s_final:.2e}, Total={t_final:.2f}",
            ),
        )

        for col_idx, (img_arr, caption) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(img_arr)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=12.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(rec.title, fontsize=10.5, fontweight="bold", labelpad=8)
            ax.set_xlabel(caption, fontsize=8.5, labelpad=4)

    fig.suptitle(
        "Perbandingan Visual Content — Style — Output (3 Pasangan Utama, 384×512)",
        fontsize=14,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_beta_tuning_grid(
    tuning_records: list[ExperimentRecord],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Membuat grid perbandingan visual Content, Style, dan hasil variasi beta (1e3, 1e5, 1e7)."""
    target_wh = (FINAL_IMAGE_SIZE[1], FINAL_IMAGE_SIZE[0])  # (W, H) = (512, 384)
    reference = tuning_records[0]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2 + len(tuning_records),
        figsize=(16.5, 4.0),
        constrained_layout=True,
    )

    panels: list[tuple[np.ndarray, str, str]] = [
        (
            _load_display_image(reference.content_path, target_wh),
            "Content (Asli)",
            reference.content_path.name,
        ),
        (
            _load_display_image(reference.style_path, target_wh),
            "Style (Referensi)",
            reference.style_path.name,
        ),
    ]

    for spec, rec in zip(BETA_TUNING_SPECS, tuning_records):
        c_final = float(rec.metrics["content_loss"][-1])
        s_final = float(rec.metrics["style_loss"][-1])
        t_final = float(rec.metrics["total_loss"][-1])
        panels.append(
            (
                _load_display_image(rec.result_path, target_wh),
                f"Output ({spec['label']})",
                f"C={c_final:.2f} | S={s_final:.2e} | T={t_final:.2f}",
            )
        )

    for idx, (img_arr, title, caption) in enumerate(panels):
        ax = axes[idx]
        ax.imshow(img_arr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(caption, fontsize=8.5, labelpad=5)

    fig.suptitle(
        "Pengaruh Variasi Bobot Gaya β ∈ {1e3, 1e5, 1e7} (α = 1.0, Gedung + Composition VII)",
        fontsize=13.5,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _create_synthetic_image(path: Path, color: tuple[int, int, int], label: str) -> None:
    """Membuat gambar dummy RGB untuk keperluan uji mandiri skrip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (FINAL_IMAGE_SIZE[1], FINAL_IMAGE_SIZE[0]), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 492, 364), outline=(255, 255, 255), width=4)
    draw.text((40, 40), label, fill=(255, 255, 255))
    img.save(path)


def _create_dummy_metrics_csv(path: Path, beta: float, steps: int = 50) -> None:
    """Membuat file metrics.csv dummy dengan kolom wajib kontrak antarmuka."""
    path.parent.mkdir(parents=True, exist_ok=True)
    step_vals = np.arange(1, steps + 1, dtype=np.int64)
    content_vals = 50.0 * (1.0 - np.exp(-step_vals / 15.0))
    style_vals = 0.03 * np.exp(-step_vals / 12.0) + 1e-4
    total_vals = 1.0 * content_vals + beta * style_vals

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "content_loss", "style_loss", "total_loss"])
        for s, c, st, tot in zip(step_vals, content_vals, style_vals, total_vals):
            writer.writerow([int(s), f"{c:.6f}", f"{st:.6f}", f"{tot:.6f}"])


def _prepare_dummy_workspace(tmp_root: Path) -> Path:
    """Menyiapkan 6 subfolder eksperimen dummy di direktori temporer."""
    outputs_dir = tmp_root / "outputs"
    for idx, spec in enumerate(MAIN_PAIR_SPECS):
        run_dir = outputs_dir / spec["folder"]
        content_img = tmp_root / f"dummy_content_{idx}.png"
        style_img = tmp_root / f"dummy_style_{idx}.png"
        _create_synthetic_image(content_img, (40 + 40 * idx, 90, 160), f"Content {idx+1}")
        _create_synthetic_image(style_img, (160, 70 + 40 * idx, 80), f"Style {idx+1}")
        _create_synthetic_image(run_dir / "result.png", (100, 100 + 30 * idx, 130), spec["title"])
        _create_dummy_metrics_csv(run_dir / "metrics.csv", beta=1e5)
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "cli_args": {
                        "content": str(content_img),
                        "style": str(style_img),
                        "alpha": 1.0,
                        "beta": 1e5,
                        "steps": 50,
                    }
                },
                handle,
            )

    ref_content = tmp_root / "dummy_content_0.png"
    ref_style = tmp_root / "dummy_style_0.png"
    for idx, spec in enumerate(BETA_TUNING_SPECS):
        run_dir = outputs_dir / spec["folder"]
        _create_synthetic_image(
            run_dir / "result.png",
            (60 + 50 * idx, 110, 170 - 30 * idx),
            spec["label"],
        )
        _create_dummy_metrics_csv(run_dir / "metrics.csv", beta=float(spec["beta"]))
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "cli_args": {
                        "content": str(ref_content),
                        "style": str(ref_style),
                        "alpha": 1.0,
                        "beta": float(spec["beta"]),
                        "steps": 50,
                    }
                },
                handle,
            )

    return outputs_dir


def generate_all_figures(
    outputs_root: Path,
    figures_dir: Path,
    base_root: Path = PROJECT_ROOT,
    dpi: int = 300,
) -> list[Path]:
    """Menjalankan seluruh pipeline pembuatan 4 figur laporan."""
    main_records, tuning_records = load_all_experiments(outputs_root, base_root=base_root)

    generated: list[Path] = [
        plot_main_pair_losses(
            main_records,
            figures_dir / "loss_curves_pairs.png",
            dpi=dpi,
        ),
        plot_beta_tuning_losses(
            tuning_records,
            figures_dir / "loss_curves_tuning_beta.png",
            dpi=dpi,
        ),
        plot_main_pairs_grid(
            main_records,
            figures_dir / "grid_main_pairs.png",
            dpi=dpi,
        ),
        plot_beta_tuning_grid(
            tuning_records,
            figures_dir / "grid_beta_tuning.png",
            dpi=dpi,
        ),
    ]
    return generated


def parse_args() -> argparse.Namespace:
    """Parsing argumen CLI untuk scripts/make_figures.py."""
    parser = argparse.ArgumentParser(
        description="Buat figur kurva loss dan grid perbandingan visual NST."
    )
    parser.add_argument(
        "--mode",
        choices=("real", "dummy"),
        default="real",
        help="Gunakan 'real' untuk data di outputs/ atau 'dummy' untuk uji mandiri.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=OUTPUT_DIR,
        help="Direktori induk subfolder eksperimen (default: outputs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "figures",
        help="Direktori tujuan penyimpanan figur (default: outputs/figures/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolusi DPI untuk penyimpanan figur PNG (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point pembuatan figur laporan NST."""
    args = parse_args()

    if args.mode == "dummy":
        with tempfile.TemporaryDirectory(prefix="nst_dummy_figs_") as tmp_dir_str:
            tmp_root = Path(tmp_dir_str)
            dummy_outputs = _prepare_dummy_workspace(tmp_root)
            generated = generate_all_figures(
                outputs_root=dummy_outputs,
                figures_dir=args.output_dir,
                base_root=tmp_root,
                dpi=args.dpi,
            )
    else:
        generated = generate_all_figures(
            outputs_root=args.outputs_root,
            figures_dir=args.output_dir,
            base_root=PROJECT_ROOT,
            dpi=args.dpi,
        )

    print("Figur berhasil dibuat:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
