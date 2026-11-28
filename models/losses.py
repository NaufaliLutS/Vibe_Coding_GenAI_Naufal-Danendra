"""Modul 2: Fungsi loss manual untuk Neural Style Transfer (NST).

Menyediakan perhitungan Gram Matrix, Content Loss, Style Loss, dan Total Loss
secara manual dari nol menggunakan operasi tensor dasar PyTorch, tanpa memakai
nn.MSELoss, F.mse_loss, atau pustaka NST instan.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

# Pastikan root repositori ada di sys.path agar import config.py berhasil
# baik saat dijalankan sebagai modul maupun langsung (python models/losses.py).
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    import config as _config
except ImportError:
    _config = None

DEFAULT_CONTENT_LAYER: str = getattr(
    _config, "CONTENT_LAYER", getattr(_config, "DEFAULT_CONTENT_LAYER", "conv4_2")
)
DEFAULT_STYLE_WEIGHTS: dict[str, float] = getattr(
    _config,
    "STYLE_WEIGHTS",
    getattr(
        _config,
        "DEFAULT_STYLE_WEIGHTS",
        {
            "conv1_1": 0.2,
            "conv2_1": 0.2,
            "conv3_1": 0.2,
            "conv4_1": 0.2,
            "conv5_1": 0.2,
        },
    ),
)
DEFAULT_ALPHA: float = float(
    getattr(_config, "ALPHA", getattr(_config, "DEFAULT_ALPHA", 1.0))
)
DEFAULT_BETA: float = float(
    getattr(_config, "BETA", getattr(_config, "DEFAULT_BETA", 1e5))
)


def compute_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Menghitung Gram Matrix ternormalisasi dari feature map berdimensi (B, C, H, W)."""
    if feature_map.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), got {feature_map.dim()}D tensor "
            f"with shape {tuple(feature_map.shape)}."
        )

    batch_size, channels, height, width = feature_map.shape
    num_elements = channels * height * width
    if num_elements <= 0:
        raise ValueError(
            f"Feature map dimensions must be positive, got shape {tuple(feature_map.shape)}."
        )

    # Ratakan dimensi spasial H dan W menjadi satu dimensi: (B, C, H, W) -> (B, C, H * W)
    features = feature_map.reshape(batch_size, channels, height * width)
    # Transpose dua dimensi terakhir untuk perkalian Gram: (B, C, H * W) -> (B, H * W, C)
    features_t = features.transpose(1, 2)
    # Kalikan matriks fitur per batch dengan transposenya: (B, C, H * W) x (B, H * W, C) -> (B, C, C)
    gram = torch.bmm(features, features_t)
    # Normalisasi Gram Matrix dengan total elemen per sampel (C * H * W): (B, C, C) -> (B, C, C)
    gram = gram / float(num_elements)
    return gram


def compute_content_loss(
    target_features: dict[str, torch.Tensor],
    content_features: dict[str, torch.Tensor],
    content_layer: str = DEFAULT_CONTENT_LAYER,
) -> torch.Tensor:
    """Menghitung Content Loss (MSE manual) pada layer konten yang ditentukan."""
    if content_layer not in target_features:
        raise KeyError(f"Content layer '{content_layer}' not found in target_features.")
    if content_layer not in content_features:
        raise KeyError(f"Content layer '{content_layer}' not found in content_features.")

    # Ambil tensor aktivasi layer konten dari target dan konten: (B, C, H, W)
    target_feat = target_features[content_layer]
    content_feat = content_features[content_layer]

    # Hitung selisih elemen per elemen: (B, C, H, W) - (B, C, H, W) -> (B, C, H, W)
    diff = target_feat - content_feat
    # Kuadratkan setiap elemen selisih: (B, C, H, W) -> (B, C, H, W)
    squared_diff = diff ** 2
    # Hitung rata-rata seluruh elemen menjadi skalar MSE: (B, C, H, W) -> ()
    content_loss = torch.mean(squared_diff)
    return content_loss


def compute_style_loss(
    target_features: dict[str, torch.Tensor],
    style_features: dict[str, torch.Tensor],
    style_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Menghitung total Style Loss berbobot antar layer gaya menggunakan Gram Matrix."""
    if style_weights is None:
        style_layers = list(style_features.keys())
        if not style_layers:
            raise ValueError("style_features cannot be empty when style_weights is None.")
        equal_weight = 1.0 / len(style_layers)
        weights = {layer: equal_weight for layer in style_layers}
    else:
        if not style_weights:
            raise ValueError("style_weights cannot be empty.")
        weights = style_weights

    first_layer = next(iter(weights))
    if first_layer not in target_features:
        raise KeyError(f"Style layer '{first_layer}' not found in target_features.")
    ref_tensor = target_features[first_layer]

    # Inisialisasi akumulator skalar pada device dan dtype yang sama: ()
    style_loss = torch.zeros((), dtype=ref_tensor.dtype, device=ref_tensor.device)

    for layer, weight in weights.items():
        if layer not in target_features:
            raise KeyError(f"Style layer '{layer}' not found in target_features.")
        if layer not in style_features:
            raise KeyError(f"Style layer '{layer}' not found in style_features.")

        # Ambil feature map target dan style pada layer saat ini: (B, C, H, W)
        target_feat = target_features[layer]
        style_feat = style_features[layer]

        # Hitung Gram Matrix ternormalisasi masing-masing: (B, C, H, W) -> (B, C, C)
        gram_target = compute_gram_matrix(target_feat)
        gram_style = compute_gram_matrix(style_feat)

        # Hitung selisih antar Gram Matrix: (B, C, C) - (B, C, C) -> (B, C, C)
        gram_diff = gram_target - gram_style
        # Kuadratkan selisih Gram Matrix: (B, C, C) -> (B, C, C)
        gram_sq_diff = gram_diff ** 2
        # Hitung rata-rata menjadi skalar MSE layer gaya: (B, C, C) -> ()
        layer_style_loss = torch.mean(gram_sq_diff)

        # Kalikan loss layer dengan bobotnya dan jumlahkan ke akumulator: () + () -> ()
        style_loss = style_loss + float(weight) * layer_style_loss

    return style_loss


def compute_total_loss(
    target_features: dict[str, torch.Tensor],
    content_features: dict[str, torch.Tensor],
    style_features: dict[str, torch.Tensor],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    content_layer: str = DEFAULT_CONTENT_LAYER,
    style_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Menghitung total loss NST (alpha * content_loss + beta * style_loss) beserta komponennya."""
    # Hitung skalar content_loss: ()
    content_loss = compute_content_loss(
        target_features=target_features,
        content_features=content_features,
        content_layer=content_layer,
    )
    # Hitung skalar style_loss: ()
    style_loss = compute_style_loss(
        target_features=target_features,
        style_features=style_features,
        style_weights=style_weights,
    )
    # Gabungkan kedua loss dengan bobot alpha dan beta menjadi skalar total_loss: () + () -> ()
    total_loss = float(alpha) * content_loss + float(beta) * style_loss
    return total_loss, content_loss, style_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan unit test models/losses.py pada device: {device}")

    torch.manual_seed(42)

    # Spesifikasi layer VGG-19 standar NST: nama layer -> (B, C, H, W)
    layer_shapes = {
        "conv1_1": (2, 64, 32, 32),
        "conv2_1": (2, 128, 16, 16),
        "conv3_1": (2, 256, 8, 8),
        "conv4_1": (2, 512, 4, 4),
        "conv4_2": (2, 512, 4, 4),
        "conv5_1": (2, 512, 2, 2),
    }

    # 1. Uji bentuk output dan simetri compute_gram_matrix
    sample_feat = torch.randn(layer_shapes["conv2_1"], device=device)
    gram = compute_gram_matrix(sample_feat)
    expected_gram_shape = (layer_shapes["conv2_1"][0], layer_shapes["conv2_1"][1], layer_shapes["conv2_1"][1])
    assert gram.shape == expected_gram_shape, (
        f"Bentuk Gram Matrix tidak sesuai: {gram.shape} != {expected_gram_shape}"
    )
    assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6), (
        "Gram Matrix tidak bersifat simetris!"
    )
    print(f"[PASS] Bentuk Gram Matrix tepat {tuple(gram.shape)} dan simetris.")

    # 2. Uji content_loss dan style_loss ~ 0.0 saat target identik dengan konten/gaya
    content_feats = {
        layer: torch.randn(shape, device=device) for layer, shape in layer_shapes.items()
    }
    style_feats = {
        layer: torch.randn(shape, device=device) for layer, shape in layer_shapes.items()
    }

    zero_content_loss = compute_content_loss(
        target_features={k: v.clone() for k, v in content_feats.items()},
        content_features=content_feats,
        content_layer=DEFAULT_CONTENT_LAYER,
    )
    assert torch.isclose(zero_content_loss, torch.tensor(0.0, device=device), atol=1e-7), (
        f"content_loss seharusnya ~0.0 saat identik, didapat {zero_content_loss.item()}"
    )

    style_only_keys = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    style_subset = {k: style_feats[k] for k in style_only_keys}
    zero_style_loss_equal = compute_style_loss(
        target_features={k: v.clone() for k, v in style_subset.items()},
        style_features=style_subset,
        style_weights=None,
    )
    zero_style_loss_weighted = compute_style_loss(
        target_features={k: v.clone() for k, v in style_feats.items()},
        style_features=style_feats,
        style_weights=DEFAULT_STYLE_WEIGHTS,
    )
    assert torch.isclose(zero_style_loss_equal, torch.tensor(0.0, device=device), atol=1e-7), (
        f"style_loss (equal) seharusnya ~0.0 saat identik, didapat {zero_style_loss_equal.item()}"
    )
    assert torch.isclose(zero_style_loss_weighted, torch.tensor(0.0, device=device), atol=1e-7), (
        f"style_loss (weighted) seharusnya ~0.0 saat identik, didapat {zero_style_loss_weighted.item()}"
    )
    print(
        f"[PASS] Loss identik mendekati 0.0 "
        f"(content={zero_content_loss.item():.2e}, "
        f"style_equal={zero_style_loss_equal.item():.2e}, "
        f"style_weighted={zero_style_loss_weighted.item():.2e})."
    )

    # 3. Uji total_loss positif & aliran gradien saat target berbeda dari konten/gaya
    target_base = {
        layer: torch.randn(shape, device=device, requires_grad=True)
        for layer, shape in layer_shapes.items()
    }
    # Simulasikan feature map target yang diturunkan dari tensor daun dengan requires_grad=True
    target_feats = {layer: feat * 1.0 for layer, feat in target_base.items()}

    total_loss, c_loss, s_loss = compute_total_loss(
        target_features=target_feats,
        content_features=content_feats,
        style_features=style_feats,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        content_layer=DEFAULT_CONTENT_LAYER,
        style_weights=DEFAULT_STYLE_WEIGHTS,
    )

    assert total_loss.dim() == 0 and c_loss.dim() == 0 and s_loss.dim() == 0, (
        "Semua komponen loss harus berupa tensor skalar 0D."
    )
    assert total_loss.item() > 0.0, (
        f"total_loss harus bernilai positif saat target berbeda, didapat {total_loss.item()}"
    )
    print(
        f"[PASS] total_loss positif: total={total_loss.item():.4f}, "
        f"content={c_loss.item():.4f}, style={s_loss.item():.6f}."
    )

    # 4. Uji backward dan integritas gradien (tidak None, tidak NaN, tidak Inf)
    total_loss.backward()
    for layer, leaf_tensor in target_base.items():
        grad = leaf_tensor.grad
        assert grad is not None, f"Gradien bernilai None pada layer '{layer}'!"
        assert not torch.isnan(grad).any(), f"Gradien mengandung NaN pada layer '{layer}'!"
        assert not torch.isinf(grad).any(), f"Gradien mengandung Inf pada layer '{layer}'!"
    print("[PASS] Aliran gradien valid pada seluruh layer (grad is not None, bebas NaN/Inf).")
    print("Seluruh unit test models/losses.py BERHASIL.")
