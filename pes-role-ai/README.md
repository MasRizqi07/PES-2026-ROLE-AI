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

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori & Pasang Flask
```bash
git clone https://github.com/MasRizqi07/PES-2026-ROLE-AI.git
cd PES-2026-ROLE-AI
pip install flask
```

### 2. Eksekusi Unit Testing Mandiri CLI
```bash
python pes-role-ai/test_knn.py
```

### 3. Menjalankan Server Web Flask
```bash
python pes-role-ai/app.py
```
Akses di browser: `http://127.0.0.1:5000/`

---

## 👥 Profil Pengembang & Kontribusi

Proyek akademik ini diselesaikan oleh **Kelompok 6 (Wong Urakan)**:
* **Ahmad Rizqi Mubarok** (NIM: `06.2025.1.07900`)
* **Wahyu Aditia Putra** (NIM: `06.2025.1.07869`)

*Dosen Pengampu:*  
**Pak Kurniawan** — Dosen Mata Kuliah Kecerdasan Artifisial.

Copyright © 2026 **Kelompok 6: Wong Urakan**

