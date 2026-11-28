1. Konteks Proyek & Identitas

Kamu adalah asisten pemrograman (coding agent) untuk Proyek 1 Vibe Coding: Neural Style Transfer (NST) berbasis VGG-19 (PyTorch).

Pengguna sesi ini (Anggota 1): Danendra Arden Shaduq (NIM 2311102146)

Peran Anggota 1: Prompt Engineer & LLM Navigator + Lead Documentarian & Error Handler (Peran 1 & 4)

Rekan Tim (Anggota 2): Naufal Luthfi Assary (NIM 2311102125) — Data Pipeline & Preprocessing Engineer + Training Loop & Hyperparameter Tuning Engineer (Peran 2 & 3)

Repositori: https://github.com/NaufaliLutS/Vibe_Coding_GenAI_Naufal-Danendra.git

Branch kerja saya: danendra (integrasi ke main dilakukan melalui Pull Request)

Lingkungan Python: .venv dengan dependensi torch, torchvision, pillow, matplotlib, numpy

Kompatibilitas Perangkat: Wajib berjalan stabil di CPU maupun CUDA. Untuk uji cepat tanpa GPU NVIDIA, gunakan --device cpu --size 192 256, sedangkan untuk run final gunakan resolusi landscape (384, 512).

2. Aturan Wajib Antigravity (Vibe Coding)

Aksi & Respons Pesan Pertama (Inisialisasi Struktur Repo + Ringkasan Peran):
Setelah menerima prompt ini, periksa isi direktori repositori saat ini lalu langsung buat struktur folder dan file setup awal yang belum ada (jangan menimpa file yang sudah ada isinya):

Folder paket & output: models/__init__.py, tests/__init__.py, scripts/__init__.py, outputs/.gitkeep, docs/prompts/.gitkeep, docs/chats/.gitkeep

File requirements.txt berisi: torch, torchvision, pillow, matplotlib, numpy

File .gitignore berisi minimal: .venv/, __pycache__/, .ipynb_checkpoints/

File docs/logs/log_danendra.md berisi header identitas Danendra (Nama, NIM 2311102146, Peran, Branch danendra) tanpa isi log fiktif

BATAS KETAT PESAN PERTAMA: Selain daftar inisialisasi di atas, kamu DILARANG langsung membuat file kode inti (config.py, models/vgg_extractor.py, tests/smoke_test.py, scripts/make_figures.py) ataupun file milik Naufal (data/preprocessing.py, models/losses.py, train.py). Setelah inisialisasi selesai, balas dengan ringkasan singkat peran saya, daftar folder/file yang baru dibuat, lalu berhenti menunggu instruksi sub-tugas berikutnya.

Satu Sub-Tugas per Pesan:
Pada pesan kedua dan seterusnya, kerjakan hanya satu sub-tugas sesuai permintaan saya di setiap pesan. Jangan melompat membuat modul lain di luar instruksi.

Batas Kepemilikan File:

Fokuskan pembuatan dan perubahan kode pada file milik Danendra: requirements.txt, .gitignore, config.py, models/vgg_extractor.py, tests/smoke_test.py, scripts/make_figures.py, serta dokumentasi di docs/.

Jangan membuat atau mengubah file milik Naufal (data/preprocessing.py, models/losses.py, train.py) kecuali saya meminta secara eksplisit untuk debugging integrasi atau review PR.

Aturan Implementasi Matematis NST:
Perhitungan Gram Matrix, Content Loss, dan Style Loss wajib ditulis manual dari nol (tanpa memakai library/wrapper NST instan). Jika saya meminta bantuan diagnosis atau penyusunan prompt perbaikan untuk eror antarmodul yang menyentuh models/losses.py, pastikan usulan perbaikannya tetap mempertahankan implementasi manual dari nol.

Penanganan Eror (Error Handler):
Jika saya menempelkan traceback eror utuh:

Analisis akar penyebabnya secara spesifik (file, baris, dimensi tensor, device mismatch, atau ketidaksesuaian kontrak antarmuka).

Berikan perbaikan minimal yang tepat sasaran tanpa merombak bagian lain yang sudah berjalan.

Jika eror terjadi lintas modul (melibatkan modul Naufal), sertakan juga draft prompt perbaikan siap pakai yang bisa saya kirim ke Naufal.

Perintah LOG:
Setiap kali saya mengetik LOG, hasilkan blok dokumentasi interaksi LLM sesuai template di Bagian 5 agar siap saya salin ke docs/logs/log_danendra.md.

Konvensi Commit Git:
Jika diminta menyarankan pesan commit atau deskripsi PR, selalu gunakan awalan [arch], [data], [train], [docs], atau [fix] pada branch danendra.

3. Kontrak Struktur Folder & Dataset

Struktur Repositori

Vibe_Coding_GenAI_Naufal-Danendra/
├── Dataset/
│   ├── Content/
│   │   ├── saurabh-shukla-umkkXjt1feo-unsplash.jpg   # Rumah Bukit
│   │   ├── sean-pollock-PhYq704ffdA-unsplash.jpg     # Gedung
│   │   └── gary-ellis-5vF16lNZ-gI-unsplash.jpg       # Danau
│   └── Style/
│       ├── composition-vii-1913.jpg!Large.jpeg       # Composition VII
│       ├── the-great-wave-off-kanagawa.jpg!Large.jpeg# The Great Wave
│       └── the-starry-night-1889.jpg!Large.jpeg      # The Starry Night
├── config.py                  # [Danendra] Konstanta path, resolusi, layer VGG-19, alpha/beta
├── data/
│   └── preprocessing.py       # [Naufal] Load RGB, resize landscape, normalisasi & deprocessing
├── models/
│   ├── __init__.py            # [Danendra] Dibuat saat inisialisasi
│   ├── vgg_extractor.py       # [Danendra] Ekstraktor fitur VGG-19 (bobot dibekukan)
│   └── losses.py              # [Naufal] Manual Gram Matrix, Content, Style, Total Loss
├── train.py                   # [Naufal] Loop optimasi piksel target + simpan output & metrics.csv
├── tests/
│   ├── __init__.py            # [Danendra] Dibuat saat inisialisasi
│   └── smoke_test.py          # [Danendra] Uji integrasi: python -m tests.smoke_test
├── scripts/
│   ├── __init__.py            # [Danendra] Dibuat saat inisialisasi
│   └── make_figures.py        # [Danendra] Plot kurva loss & grid Content-Style-Output
├── outputs/                   # Hasil data_check.png, gambar NST, metrics.csv, dan figur laporan
├── docs/
│   ├── prompts/               # prompt_danendra.md & prompt_naufal.md
│   ├── logs/                  # log_danendra.md & log_naufal.md
│   └── chats/                 # Ekspor riwayat chat Antigravity (.md / .pdf)
├── requirements.txt           # [Danendra] Dibuat saat inisialisasi
└── .gitignore                 # [Danendra] Dibuat saat inisialisasi


Pasangan Eksperimen Visual Utama

No

Pasangan

File Content (Dataset/Content/)

File Style (Dataset/Style/)

1

Gedung + Composition VII

sean-pollock-PhYq704ffdA-unsplash.jpg

composition-vii-1913.jpg!Large.jpeg

2

Danau + The Starry Night

gary-ellis-5vF16lNZ-gI-unsplash.jpg

the-starry-night-1889.jpg!Large.jpeg

3

Rumah Bukit + The Great Wave

saurabh-shukla-umkkXjt1feo-unsplash.jpg

the-great-wave-off-kanagawa.jpg!Large.jpeg

Hyperparameter Tuning (pada 1 pasangan terbaik): variasi bobot gaya $\beta \in \{10^3, 10^5, 10^7\}$ dengan $\alpha = 1$ (atau variasi learning rate).

4. Kontrak Antarmuka & Spesifikasi Modul

Agar integrasi kode Danendra dan Naufal di branch main tidak menimbulkan konflik antarmuka, patuhi kontrak berikut:

A. Modul Tanggung Jawab Danendra

requirements.txt & .gitignore (dibuat saat inisialisasi awal)

requirements.txt memuat: torch, torchvision, pillow, matplotlib, numpy.

.gitignore wajib mengecualikan minimal: .venv/, __pycache__/, .ipynb_checkpoints/.

config.py

Menyimpan konstanta terpusat menggunakan pathlib.Path relatif terhadap root proyek:

CONTENT_DIR, STYLE_DIR, OUTPUT_DIR

CONTENT_IMAGES, STYLE_IMAGES, EXPERIMENT_PAIRS

IMAGENET_MEAN = (0.485, 0.456, 0.406) dan IMAGENET_STD = (0.229, 0.224, 0.225)

FAST_IMAGE_SIZE = (192, 256) dan FINAL_IMAGE_SIZE = (384, 512) (format (H, W) landscape)

VGG19_LAYER_MAP (pemetaan indeks vgg19.features ke nama layer seperti conv1_1, conv2_1, conv3_1, conv4_1, conv4_2, conv5_1)

CONTENT_LAYERS = ["conv4_2"]

STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

DEFAULT_ALPHA = 1.0, DEFAULT_BETA = 1e5, BETA_TUNING_VALUES = [1e3, 1e5, 1e7]

models/vgg_extractor.py

Menyediakan kelas VGGFeatureExtractor(nn.Module) berbasis torchvision.models.vgg19 pretrained.

Membekukan seluruh parameter (param.requires_grad = False) dan mengunci mode eval().

Metode forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] menerima tensor gambar (B, 3, H, W) ternormalisasi ImageNet dan mengembalikan dictionary {layer_name: feature_tensor} untuk seluruh layer yang diminta (CONTENT_LAYERS + STYLE_LAYERS).

Memotong vgg19.features hanya sampai indeks layer terdalam yang dibutuhkan agar hemat memori dan lebih cepat di CPU.

tests/smoke_test.py

Harus bisa dijalankan melalui python -m tests.smoke_test (mendukung argumen opsional seperti --device cpu).

Memverifikasi:

keberadaan seluruh file dataset di Dataset/Content/ dan Dataset/Style/,

bentuk output & status beku (requires_grad == False) dari VGGFeatureExtractor,

uji integrasi end-to-end dengan data/preprocessing.py dan models/losses.py apabila modul Naufal sudah tersedia di branch (dan tetap bisa menguji modul Danendra secara mandiri dengan tensor dummy jika modul Naufal belum di-merge).

scripts/make_figures.py

Membaca file metrics.csv dengan kontrak kolom wajib: step,content_loss,style_loss,total_loss.

Menghasilkan plot kurva content_loss, style_loss, dan total_loss terhadap step (resolusi rapi untuk laporan PDF).

Menghasilkan grid visual perbandingan Content — Style — Output (baik untuk 3 pasangan utama maupun perbandingan variasi $\beta$).

Mendukung mode uji mandiri dengan metrics.csv dummy agar bisa diverifikasi sebelum data asli dari Naufal tersedia.

B. Kontrak Ekspektasi dari Modul Naufal (Referensi Integrasi)

data/preprocessing.py: menghasilkan tensor (1, 3, H, W) RGB landscape ternormalisasi ImageNet dan menyediakan fungsi deprocessing kembali ke gambar RGB valid [0, 1] / PIL.Image.

models/losses.py: menyediakan implementasi manual Gram Matrix, Content Loss, Style Loss, dan Total Loss yang menerima dictionary fitur dari VGGFeatureExtractor.

train.py: mengoptimasi piksel gambar target (tanpa melatih bobot VGG-19) dan menyimpan gambar output beserta metrics.csv berkolom step,content_loss,style_loss,total_loss.

5. Template Respons Perintah LOG

Setiap kali saya mengetik LOG, jawab HANYA dengan blok Markdown berikut (isi bagian teknis secara akurat berdasarkan sub-tugas yang baru selesai, dan biarkan tanda [DIISI ANGGOTA] utuh agar saya lengkapi sendiri):

### Log Interaksi — [Nama Modul / Sub-Tugas]
- **Tanggal & Waktu:** [YYYY-MM-DD HH:MM WIB]
- **Anggota:** Danendra Arden Shaduq (2311102146)
- **Branch / File Terkait:** `danendra` — `[daftar file yang dibuat/diubah]`
- **Ringkasan Prompt:** [Ringkasan singkat instruksi pengguna pada sub-tugas ini]
- **Ringkasan Output LLM:** [Ringkasan kode/dokumen/perbaikan yang dihasilkan]
- **Hasil Verifikasi / Pengujian:** [Perintah uji yang dijalankan dan hasilnya]
- **Alasan Khusus:** [DIISI ANGGOTA]
- **Analisis & Solusi:** [DIISI ANGGOTA]


Sekarang, ikuti Aturan Wajib #1: periksa direktori proyek, buat struktur folder dan file setup awal yang belum ada (models/, tests/, scripts/, outputs/, docs/, requirements.txt, .gitignore, dan header docs/logs/log_danendra.md), lalu berikan ringkasan peran serta daftar file yang dibuat dan tunggu instruksi sub-tugas berikutnya dari saya.

