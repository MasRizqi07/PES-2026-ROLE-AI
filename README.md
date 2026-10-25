# ⚽ PES 2026 Tactical Role Predictor
### Sistem Klasifikasi Karakter & Peran Taktikal Sepak Bola Berbasis Algoritma Scratch K-Nearest Neighbors (KNN)

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML%20Engine-Zero--Library%20(Scratch)-4CAF50?style=for-the-badge)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
[![Unit Testing](https://img.shields.io/badge/CLI%20Tests-100%25%20PASS-success?style=for-the-badge&logo=checkmarx&logoColor=white)](#-hasil-evaluasi-unit-testing-cli)
[![Academic](https://img.shields.io/badge/Course-Kecerdasan%20Artifisial-FF6F00?style=for-the-badge)](https://github.com/MasRizqi07/PES-2026-ROLE-AI)

---

## 📌 Informasi Proyek & Identitas Akademik

Proyek ini dikembangkan sebagai tugas implementasi komprehensif mata kuliah **Kecerdasan Artifisial** dengan fokus pada rekayasa algoritma machine learning deterministik murni tanpa dependensi pihak ketiga (*zero-library machine learning*).

* **Mata Kuliah:** Kecerdasan Artifisial
* **Dosen Pengampu:** Pak Kurniawan
* **Kelompok:** Kelompok 6 — **Wong Urakan**
* **Anggota Tim:**
  1. **Ahmad Rizqi Mubarok** (NIM: `06.2025.1.07900`)
  2. **Wahyu Aditia Putra** (NIM: `06.2025.1.07869`)
* **Repositori GitHub:** [MasRizqi07/PES-2026-ROLE-AI](https://github.com/MasRizqi07/PES-2026-ROLE-AI)

---

## 📖 Deskripsi Singkat

**PES 2026 Tactical Role Predictor** adalah sistem pakar inferensi berbasis web yang mengklasifikasikan kepribadian, gaya bermain, dan preferensi aktivitas seorang pemain ke dalam **4 peran posisi taktikal sepak bola modern** (Pro Evolution Soccer / eFootball 2026):
1. **Forward (Penyerang)**
2. **Midfielder (Gelandang)**
3. **Defender (Pemain Bertahan)**
4. **Goalkeeper (Penjaga Gawang)**

Sistem ini memetakan 5 dimensi psikometrik dan motorik pengguna ke dalam ruang vektor berdimensi lima ($\mathbb{R}^5$), lalu mencari tetangga geometris terdekat menggunakan model **Scratch K-Nearest Neighbors ($K=5$)** yang dibangun murni menggunakan pustaka standar Python.

---

## 🌟 Fitur Utama

- 🧠 **Zero-Library Machine Learning Engine**: Dilarang keras dan sama sekali tidak mengimpor `scikit-learn`, `numpy`, `pandas`, maupun `scipy`. Seluruh kalkulasi vektor, matriks jarak, dan agregasi suara menggunakan modul native Python (`math`, `collections.Counter`, `typing`).
- ⚖️ **Mekanisme Pemecah Seri Deterministik (Tie-Breaking via IDW)**: Mengatasi kebuntuan kondisi seri pada majority voting dengan metode *Inverse Distance Weighting* (IDW) dan pendekatan tetangga rank-1 terdekat.
- 🎯 **Dataset Latih Seimbang (Balanced Classes)**: Terdiri dari 32 data sampel representatif (8 sampel seimbang pada masing-masing 4 kelas taktis) yang mencerminkan arketipe taktik modern.
- 🕸️ **Visualisasi Spider/Radar Chart Simetris (DPI-Aware)**: Kanvas interaktif HTML5 Canvas native beresolusi tinggi yang otomatis menyesuaikan kerapatan piksel layar (HiDPI/Retina Display).
- 📑 **Generator Laporan Akademik Otomatis**: Fitur sekali klik untuk mengekspor laporan komprehensif berformat `.txt` yang mencantumkan identitas pengembang, dosen pengampu, rumus matematika, dan log 5 tetangga terdekat.
- 🛡️ **Robust Error Handling & Logging Terstruktur**: Validasi input ketat (skala 1.0 - 5.0) dengan fallback aman dan pencatatan log terminal transparan pada setiap inferensi.
- 🧪 **Unit Testing Mandiri Berbasis CLI**: Skrip otomatisasi pengujian `test_knn.py` yang memvalidasi kasus ekstrem, integritas dimensi, dan resolusi tie-break dengan status `[PASS / FAIL]`.

---

## 📐 Landasan Teori & Formula Matematis

Algoritma KNN (*K-Nearest Neighbors*) bekerja dengan prinsip *instance-based lazy learning*. Seluruh proses inferensi dieksekusi saat fase prediksi (*query-time*).

```
                      RUANG VEKTOR 5-DIMENSI
  [Agresivitas, Kreativitas, Disiplin, Kerjasama, Refleks]
                            │
                            ▼
           Hitung Euclidean Distance ke 32 Data Latih
                            │
                            ▼
               Sorting Jarak Ascending (Timsort)
                            │
                            ▼
                 Ambil Subset K=5 Tetangga
                            │
            ┌───────────────┴───────────────┐
      Mayoritas Tunggal               Kondisi Seri (Tie)
            │                               │
            ▼                               ▼
       Winner Label                 Inverse Distance
                                    Weighting (IDW)
            │                               │
            └───────────────┬───────────────┘
                            ▼
              Output Prediksi, Confidence (%),
                 Radar Canvas & Laporan
```

### 1. Metrik Jarak Geometris (Euclidean Distance)
Jarak geometris antara vektor input $\mathbf{p} = (p_1, p_2, \dots, p_n)$ dan sampel data latih $\mathbf{q} = (q_1, q_2, \dots, q_n)$ dihitung menggunakan formula Euclidean berdimensi $n=5$:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### 2. Aturan Pemungutan Suara Mayoritas (Majority Voting)
Dari himpunan $K$ tetangga terdekat $N_K = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_K, y_K)\}$, label kelas prediksi $\hat{y}$ ditentukan oleh modus suara terbanyak:

$$\hat{y} = \arg\max_{c \in C} \sum_{i=1}^{K} \mathbb{I}(y_i = c)$$

di mana $C = \{\text{Forward}, \text{Midfielder}, \text{Defender}, \text{Goalkeeper}\}$ dan $\mathbb{I}(\cdot)$ adalah fungsi indikator.

### 3. Tingkat Keyakinan Prediksi (Confidence Score)
Dihitung berdasarkan proporsi suara mayoritas terhadap total nilai $K$:

$$\text{Confidence} = \left( \frac{\sum_{i=1}^{K} \mathbb{I}(y_i = \hat{y})}{K} \right) \times 100\%$$

### 4. Mekanisme Pemecah Seri (Inverse Distance Weighting Tie-Breaker)
Jika dua atau lebih kelas kandidat memiliki perolehan suara terbanyak yang identik pada nilai $K$, pemenang diputuskan melalui akumulasi bobot jarak terbalik (*Inverse Distance Weighting*):

$$W(c) = \sum_{j=1}^{K} \mathbb{I}(y_j = c) \cdot \frac{1}{d_j + \epsilon}$$

dengan $\epsilon = 10^{-6}$ sebagai konstanta penstabil numerik untuk menghindari pembagian dengan nol (*division by zero*). Kelas dengan nilai $W(c)$ terbesar dideklarasikan sebagai pemenang deterministik.

### 5. Normalisasi Min-Max (Opsional untuk Komparasi Fitur)
$$x_{\text{norm}} = \frac{x - \min(X)}{\max(X) - \min(X)}$$

---

## 📊 5 Dimensi Fitur & Profil Karakter

Setiap parameter diukur dalam skala ordinal kontinu **1.0 hingga 5.0**:

| Indeks | Parameter Fitur | Deskripsi Perilaku & Aktivitas Terkait | Bobot Dominan Pada |
| :---: | :--- | :--- | :---: |
| **0** | **Agresivitas & Risiko** | Dorongan menekan, keberanian duel fisik, permainan berisiko tinggi (FPS, Olahraga Ekstrem). | Forward |
| **1** | **Visi & Kreativitas** | Imajinasi spasial, antisipasi langkah lawan, variasi solusi taktis (Catur, Puzzle, Desain). | Midfielder |
| **2** | **Disiplin Defensif** | Kepatuhan posisi, kesabaran zonal, ketahanan mental kalkulatif (Membaca, Analisis Data). | Defender |
| **3** | **Kerjasama Tim** | Sinergi kolektif, komunikasi harmonis, distribusi progresif (Game Co-op, Organisasi). | Midfielder |
| **4** | **Refleks Instan** | Kecepatan respons motorik terhadap krisis mendadak (Tenis Meja, Bela Diri, Reaksi Kilat). | Goalkeeper |

### Matriks Karakteristik Taktik PES 2026

```
    Forward       Midfielder       Defender       Goalkeeper
   (Agresif)       (Kreatif)      (Disiplin)       (Refleks)
     ┌───┐           ┌───┐           ┌───┐           ┌───┐
     │ 5 │           │ 2 │           │ 2 │           │ 1 │  Agresivitas
     │ 4 │           │ 5 │           │ 2 │           │ 2 │  Kreativitas
     │ 1 │           │ 3 │           │ 5 │           │ 4 │  Disiplin
     │ 2 │           │ 5 │           │ 3 │           │ 2 │  Kerjasama
     │ 4 │           │ 3 │           │ 2 │           │ 5 │  Refleks
     └───┘           └───┘           └───┘           └───┘
```

---

## 🗂️ Struktur Direktori Repositori

```text
PES-2026-ROLE-AI/
├── .gitignore                    # Konfigurasi pengabaian file cache Python
├── README.md                     # Dokumentasi akademik & petunjuk teknis
└── pes-role-ai/                  # Direktori utama aplikasi
    ├── app.py                    # Server backend Flask & HTTP route controller
    ├── core_ai.py                # Implementasi algoritma Scratch KNN & IDW Tie-Breaker
    ├── dataset.py                # 32 data latih seimbang & metadata profil taktis
    ├── test_knn.py               # Unit testing mandiri CLI (4 skenario ekstrem)
    ├── static/
    │   └── style.css             # Desain antarmuka responsif tema modern dark
    └── templates/
        └── index.html            # Template antarmuka web, HTML5 Canvas radar & exporter
```

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Prasyarat Sistem
- **Python:** Versi `3.10`, `3.11`, `3.12`, atau `3.14+`.
- **Git:** Versi terbaru untuk kloning repositori.
- **Web Browser:** Chrome, Firefox, Edge, Safari (Mendukung HTML5 Canvas).

### 2. Kloning Repositori
```bash
git clone https://github.com/MasRizqi07/PES-2026-ROLE-AI.git
cd PES-2026-ROLE-AI
```

### 3. Pemasangan Dependensi Web
Aplikasi ini hanya membutuhkan satu dependensi eksternal, yaitu **Flask** untuk antarmuka web. Modul machine learning sama sekali tidak membutuhkan pustaka tambahan!
```bash
pip install flask
```

---

## 🧪 Eksekusi Unit Testing Mandiri CLI

Sebelum menjalankan server web, validasi integritas model AI dapat diuji secara mandiri langsung melalui terminal:

```bash
# Jalankan dari root repositori:
python pes-role-ai/test_knn.py

# ATAU masuk ke subdirektori pes-role-ai:
cd pes-role-ai
python test_knn.py
```

### 📋 Hasil Evaluasi Unit Testing CLI:

```text
================================================================================
 UNIT TESTING ALGORITMA K-NEAREST NEIGHBORS (SCRATCH KNN) - PES 2026
 Kelompok 6: Wong Urakan | Ahmad Rizqi Mubarok & Wahyu Aditia Putra
 Mata Kuliah: Kecerdasan Artifisial | Dosen: Pak Kurniawan
================================================================================

[FASE 1] PENGUJIAN 4 SKENARIO DATA EKSTREM (K=5):
----------------------------------------------------------------------------------------
| No  | Nama Skenario                       | Target     | Aktual     |  Conf   |  Status  |
----------------------------------------------------------------------------------------
|  1  | Penyerang Murni (Forward)           | Forward    | Forward    | 100.0%  |  [PASS]  |
|  2  | Playmaker Visioner (Midfielder)     | Midfielder | Midfielder | 100.0%  |  [PASS]  |
|  3  | Tembok Pertahanan (Defender)        | Defender   | Defender   | 100.0%  |  [PASS]  |
|  4  | Penjaga Gawang Responsif (Goalkeeper) | Goalkeeper | Goalkeeper | 100.0%  |  [PASS]  |
----------------------------------------------------------------------------------------

[FASE 2] LOG 5 TETANGGA TERDEKAT DARI SETIAP SKENARIO EKSTREM:
  - Seluruh kalkulasi jarak Euclidean terverifikasi secara ascending matematis.

[FASE 3] PENGUJIAN MEKANISME DETERMINISTIK TIE-BREAKING (INVERSE DISTANCE WEIGHTING):
  Titik Uji        : [1.2, 1.2]
  Jumlah Voting    : {'Forward': 1, 'Defender': 1} (Kondisi Seri: 1 vs 1 pada K=2)
  Flag Tie-Break   : True
  Pemenang Akhir   : Forward (Expected: Forward via IDW)
  Status Tie-Break : [PASS]

[FASE 4] PENGUJIAN ROBUSTNESS ERROR HANDLING:
  Uji Dimensi Mismatch   : [PASS] - ValueError berhasil ditangkap dengan benar.
  Uji Batas Normalisasi  : [PASS] - ValueError berhasil ditangkap dengan benar.

========================================================================================
 REKAPITULASI HASIL EVALUASI UNIT TESTING AKADEMIK
========================================================================================
 >>> KESIMPULAN: SELURUH PENGUJIAN MEMENUHI STANDAR (100% PASS) <<<
 Algoritma ScratchKNN terbukti robust, deterministik, dan bebas pustaka eksternal.
========================================================================================
```

---

## 🌐 Menjalankan Server Web Flask

Jalankan perintah berikut pada terminal PowerShell / CMD / Bash:

```bash
# Dari root repositori:
python pes-role-ai/app.py
```

Buka peramban (browser) dan akses alamat lokal:
```text
http://127.0.0.1:5000/
```

### Panduan Interaksi Antarmuka:
1. **Atur Slider Parameter**: Geser 5 parameter (Agresivitas, Kreativitas, Disiplin, Kerjasama, Refleks) sesuai nilai yang diinginkan (skala 1.0 - 5.0).
2. **Hitung Prediksi KNN**: Klik tombol hijau untuk mengeksekusi inferensi.
3. **Analisis Hasil**: Periksa *badge* peran, persentase *confidence*, visualisasi radar 5-sumbu, rekomendasi taktik, dan tabel log 5 tetangga terdekat.
4. **Unduh Laporan**: Klik tombol **📄 Unduh Laporan (.txt)** untuk mengunduh arsip hasil komputasi yang terstandarisasi.

---

## 🛠️ Panduan Git Workflow (Push, Commit, Pull)

Gunakan panduan perintah Git berikut untuk menyinkronkan perubahan ke repositori GitHub:

```powershell
# 1. Cek status berkas yang dimodifikasi
git status

# 2. Tambahkan seluruh berkas ke staging area
git add .

# 3. Buat commit dengan pesan deskriptif terstandarisasi
git commit -m "docs: Add comprehensive academic README and gitignore documentation"

# 4. Tarik pembaruan dari repositori remote (jika ada rekan yang melakukan push)
git pull origin main

# 5. Unggah seluruh perubahan ke repositori GitHub
git push origin main
```

---

## 👥 Profil Pengembang & Kontribusi

Proyek akademik ini diselesaikan oleh **Kelompok 6 (Wong Urakan)**:

| Nama Pengembang | NIM | Peran & Kontribusi Utama |
| :--- | :---: | :--- |
| **Ahmad Rizqi Mubarok** | `06.2025.1.07900` | Arsitektur Algoritma Core AI, Matematika Euclidean & IDW Tie-Breaker, Backend Flask, CLI Unit Testing. |
| **Wahyu Aditia Putra** | `06.2025.1.07869` | Desain Antarmuka Web, Skrip Native Canvas Radar Chart, Kurasi Dataset 32 Sampel Taktikal PES 2026. |

*Di bawah bimbingan dan supervisi dosen pengampu:*  
**Pak Kurniawan** — Dosen Mata Kuliah Kecerdasan Artifisial.

---

## 📄 Lisensi & Integritas Akademik

Kode sumber dalam repositori ini dipublikasikan untuk tujuan edukasi dan pemenuhan tugas akademik mata kuliah Kecerdasan Artifisial. Bebas dipelajari, dimodifikasi, dan didistribusikan dengan tetap mencantumkan atribusi kepada pengembang asli (*Academic Attribution & MIT License*).

Copyright © 2026 **Kelompok 6: Wong Urakan** (Ahmad Rizqi Mubarok & Wahyu Aditia Putra).

