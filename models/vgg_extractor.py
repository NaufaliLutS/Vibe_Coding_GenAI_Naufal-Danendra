"""Ekstraktor fitur VGG-19 pretrained untuk Neural Style Transfer."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

from config import CONTENT_LAYERS, STYLE_LAYERS, VGG19_LAYER_MAP


class VGGFeatureExtractor(nn.Module):
    """Ekstraktor fitur berbasis VGG-19 dengan bobot beku dan mode eval terkunci.

    Model hanya menyimpan potongan `vgg19.features` hingga indeks layer terdalam
    yang dibutuhkan sehingga lebih hemat memori dan lebih cepat dijalankan di CPU
    maupun CUDA.
    """

    def __init__(
        self,
        layers: Iterable[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        """Inisialisasi VGGFeatureExtractor.

        Args:
            layers: Daftar nama layer yang akan diekstrak. Jika None, memakai
                gabungan `CONTENT_LAYERS` dan `STYLE_LAYERS` dari `config.py`.
            pretrained: Jika True, memuat bobot ImageNet `VGG19_Weights.DEFAULT`.
        """
        super().__init__()

        requested_layers = list(
            dict.fromkeys(
                layers if layers is not None else [*CONTENT_LAYERS, *STYLE_LAYERS]
            )
        )
        if not requested_layers:
            raise ValueError("Daftar layer yang diekstrak tidak boleh kosong.")

        name_to_index = {name: idx for idx, name in VGG19_LAYER_MAP.items()}
        unknown_layers = [name for name in requested_layers if name not in name_to_index]
        if unknown_layers:
            raise ValueError(
                f"Layer tidak dikenal dalam VGG19_LAYER_MAP: {unknown_layers}. "
                f"Pilihan valid: {list(name_to_index.keys())}"
            )

        self.target_layers: list[str] = requested_layers
        self._index_to_name: dict[int, str] = {
            name_to_index[name]: name for name in requested_layers
        }
        self.max_layer_idx: int = max(self._index_to_name)

        weights = VGG19_Weights.DEFAULT if pretrained else None
        full_vgg_features = vgg19(weights=weights).features

        # Potong hanya sampai indeks layer terdalam yang dibutuhkan
        sliced_layers = list(full_vgg_features.children())[: self.max_layer_idx + 1]
        for layer in sliced_layers:
            # Nonaktifkan inplace ReLU agar tensor fitur Conv2d tersimpan aman bagi autograd
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.features: nn.Sequential = nn.Sequential(*sliced_layers)

        # Bekukan seluruh parameter dan kunci ke mode eval
        for param in self.features.parameters():
            param.requires_grad = False
        super().eval()

    def train(self, mode: bool = True) -> "VGGFeatureExtractor":
        """Mengunci model agar tetap selalu berada dalam mode eval()."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Ekstrak peta fitur untuk seluruh layer target.

        Args:
            x: Tensor gambar ternormalisasi ImageNet dengan bentuk (B, 3, H, W).

        Returns:
            Dictionary `{nama_layer: feature_tensor}` untuk setiap layer target.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Input tensor harus berbentuk (B, 3, H, W), diterima: {tuple(x.shape)}"
            )

        extracted: dict[str, torch.Tensor] = {}
        current = x
        for idx, layer in enumerate(self.features):
            current = layer(current)
            if idx in self._index_to_name:
                extracted[self._index_to_name[idx]] = current

        return extracted
