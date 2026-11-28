"""Konfigurasi terpusat untuk proyek Neural Style Transfer (VGG-19).

Modul ini mendefinisikan konstanta path berbasis pathlib.Path relatif terhadap
root proyek, daftar dataset dan pasangan eksperimen, parameter resolusi dan
normalisasi ImageNet, pemetaan layer VGG-19, serta bobot loss default.
"""

from pathlib import Path

# Root proyek dan direktori utama
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
CONTENT_DIR: Path = DATASET_DIR / "Content"
STYLE_DIR: Path = DATASET_DIR / "Style"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# Daftar gambar Content (format landscape)
CONTENT_IMAGES: dict[str, Path] = {
    "gedung": CONTENT_DIR / "sean-pollock-PhYq704ffdA-unsplash.jpg",
    "danau": CONTENT_DIR / "gary-ellis-5vF16lNZ-gI-unsplash.jpg",
    "rumah_bukit": CONTENT_DIR / "saurabh-shukla-umkkXjt1feo-unsplash.jpg",
}

# Daftar gambar Style
STYLE_IMAGES: dict[str, Path] = {
    "composition_vii": STYLE_DIR / "composition-vii-1913.jpg!Large.jpeg",
    "the_starry_night": STYLE_DIR / "the-starry-night-1889.jpg!Large.jpeg",
    "the_great_wave": STYLE_DIR / "the-great-wave-off-kanagawa.jpg!Large.jpeg",
}

# Tiga pasangan eksperimen visual utama
EXPERIMENT_PAIRS: list[dict[str, object]] = [
    {
        "id": "pair1_gedung_composition_vii",
        "name": "Gedung + Composition VII",
        "content_key": "gedung",
        "style_key": "composition_vii",
        "content_path": CONTENT_IMAGES["gedung"],
        "style_path": STYLE_IMAGES["composition_vii"],
    },
    {
        "id": "pair2_danau_starry_night",
        "name": "Danau + The Starry Night",
        "content_key": "danau",
        "style_key": "the_starry_night",
        "content_path": CONTENT_IMAGES["danau"],
        "style_path": STYLE_IMAGES["the_starry_night"],
    },
    {
        "id": "pair3_rumah_bukit_great_wave",
        "name": "Rumah Bukit + The Great Wave",
        "content_key": "rumah_bukit",
        "style_key": "the_great_wave",
        "content_path": CONTENT_IMAGES["rumah_bukit"],
        "style_path": STYLE_IMAGES["the_great_wave"],
    },
]

# Parameter normalisasi ImageNet (urutan kanal RGB)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Resolusi gambar dalam format (H, W) landscape
FAST_IMAGE_SIZE: tuple[int, int] = (192, 256)
FINAL_IMAGE_SIZE: tuple[int, int] = (384, 512)

# Pemetaan indeks modul torchvision.models.vgg19.features ke nama layer konvolusi
VGG19_LAYER_MAP: dict[int, str] = {
    0: "conv1_1",
    5: "conv2_1",
    10: "conv3_1",
    19: "conv4_1",
    21: "conv4_2",
    28: "conv5_1",
}

# Layer ekstraksi fitur untuk Content Loss dan Style Loss
CONTENT_LAYERS: list[str] = ["conv4_2"]
STYLE_LAYERS: list[str] = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

# Bobot loss default dan variasi hyperparameter tuning beta
DEFAULT_ALPHA: float = 1.0
DEFAULT_BETA: float = 1e5
BETA_TUNING_VALUES: list[float] = [1e3, 1e5, 1e7]
