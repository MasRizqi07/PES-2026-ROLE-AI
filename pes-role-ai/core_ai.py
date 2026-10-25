"""
core_ai.py
=============================================================================
MODUL INTI ALGORITMA K-NEAREST NEIGHBORS (SCRATCH KNN)
Proyek Akademik: PES 2026 Tactical Role Predictor
Kelompok 6 (Wong Urakan):
  - Ahmad Rizqi Mubarok (06.2025.1.07900)
  - Wahyu Aditia Putra  (06.2025.1.07869)
Mata Kuliah: Kecerdasan Artifisial
Dosen Pengampu: Pak Kurniawan

Batasan Ketat:
- ZERO-LIBRARY MACHINE LEARNING (Tanpa scikit-learn, numpy, scipy, pandas).
- Seluruh kalkulasi vektor, metrik geometris, dan voting menggunakan modul native Python.
=============================================================================
"""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple


def euclidean_distance(point_a: List[float], point_b: List[float]) -> float:
    """
    Menghitung jarak geometris Euclidean murni antara dua vektor berdimensi-n.

    Formula Matematis:
        d(p, q) = \\sqrt{ \\sum_{i=1}^{n} (p_i - q_i)^2 }

    Parameter:
        point_a (List[float]): Vektor titik pertama berdimensi n.
        point_b (List[float]): Vektor titik kedua berdimensi n.

    Mengembalikan:
        float: Jarak Euclidean skalar non-negatif.

    Raises:
        ValueError: Jika dimensi kedua vektor tidak sama atau bernilai kosong.
    """
    if len(point_a) != len(point_b):
        raise ValueError(
            f"Ketidaksesuaian dimensi vektor: {len(point_a)} != {len(point_b)}"
        )
    if not point_a:
        raise ValueError("Vektor tidak boleh kosong.")

    squared_diff_sum = 0.0
    for val_a, val_b in zip(point_a, point_b):
        diff = float(val_a) - float(val_b)
        squared_diff_sum += diff * diff

    return math.sqrt(squared_diff_sum)


def min_max_normalize(
    vector: List[float], min_val: float = 1.0, max_val: float = 5.0
) -> List[float]:
    """
    Melakukan normalisasi skala fitur manual ke dalam interval [0.0, 1.0].

    Formula Matematis:
        x_norm = (x - min_val) / (max_val - min_val)

    Parameter:
        vector (List[float]): Daftar nilai skalar yang akan dinormalisasi.
        min_val (float): Batas bawah nilai referensi (default: 1.0).
        max_val (float): Batas atas nilai referensi (default: 5.0).

    Mengembalikan:
        List[float]: Vektor ternormalisasi pada skala [0.0, 1.0].
    """
    if max_val <= min_val:
        raise ValueError("max_val harus bernilai lebih besar daripada min_val.")

    denominator = max_val - min_val
    return [(float(val) - min_val) / denominator for val in vector]


class ScratchKNN:
    """
    Implementasi mandiri (scratch) algoritma K-Nearest Neighbors Classifier.

    Fitur Utama:
    1. Zero-dependency: Hanya menggunakan modul standar Python (math, collections, typing).
    2. Deterministic Tie-Breaking: Mengatasi kondisi suara seri (tie) pada majority voting
       dengan metode Inverse Distance Weighting (IDW) dan pendekatan rank-1 terdekat.
    3. Type Hinting Lengkap & Validasi Integritas Data.
    """

    def __init__(self, k: int = 5) -> None:
        """
        Inisialisasi model ScratchKNN dengan parameter hiper K tetangga.

        Parameter:
            k (int): Jumlah tetangga terdekat yang digunakan dalam voting (default: 5).
        """
        if k < 1:
            raise ValueError("Nilai hiperparameter K harus minimal 1.")
        self.k: int = k
        self.train_data: List[Dict[str, Any]] = []

    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Menyimpan dataset latih ke dalam memori sistem (lazy learner paradigm).

        Parameter:
            dataset (List[Dict[str, Any]]): Daftar kamus data latih, di mana setiap
                elemen memiliki kunci 'features' (List[float]) dan 'label' (str).

        Raises:
            ValueError: Jika format dataset kosong atau tidak valid.
        """
        if not dataset:
            raise ValueError("Dataset latih tidak boleh kosong.")

        for idx, item in enumerate(dataset):
            if "features" not in item or "label" not in item:
                raise ValueError(
                    f"Sampel indeks {idx} tidak memiliki format valid (membutuhkan 'features' dan 'label')."
                )

        self.train_data = dataset

    def euclidean_distance(
        self, point_a: List[float], point_b: List[float]
    ) -> float:
        """Helper method memanggil fungsi euclidean_distance independen."""
        return euclidean_distance(point_a, point_b)

    def predict(self, input_vector: List[float]) -> Dict[str, Any]:
        """
        Memprediksi kelas label berdasarkan input vektor karakteristik taktis.

        Langkah Algoritma:
        1. Menghitung jarak Euclidean antara input_vector ke setiap sampel data latih.
        2. Mengurutkan jarak secara menaik (ascending) menggunakan Python Timsort.
        3. Mengambil subset K tetangga terdekat: N_k = {(d_1, y_1), ..., (d_k, y_k)}.
        4. Melakukan pemungutan suara mayoritas (Majority Voting):
               \\hat{y} = \\arg\\max_{c \\in C} \\sum_{i=1}^{K} I(y_i = c)
        5. Menangani kondisi seri (Tie-Breaking) melalui Inverse Distance Weighting (IDW):
               W(c) = \\sum_{j=1}^{K} I(y_j = c) \\cdot \\frac{1}{d_j + \\epsilon}
           dengan konstanta stabilitas numerik \\epsilon = 10^{-6}.
        6. Menghitung skor confidence:
               Confidence = (suara / K) * 100%

        Parameter:
            input_vector (List[float]): Vektor fitur pengujian (misal: 5 parameter karakter).

        Mengembalikan:
            Dict[str, Any]: Kamus hasil inferensi berisi:
                - 'predicted_role' (str): Label kelas hasil klasifikasi.
                - 'confidence' (float): Persentase keyakinan suara mayoritas.
                - 'neighbors' (List[Tuple[float, str]]): Daftar (jarak, label) dari K tetangga.
                - 'nearest_neighbors' (List[Tuple[float, str]]): Alias kompatibilitas.
                - 'tie_resolved' (bool): Indikator apakah tie-breaker diaktifkan.
                - 'vote_counts' (Dict[str, int]): Rincian perolehan suara tiap kelas pada K.

        Raises:
            RuntimeError: Jika model belum dilatih (fit) dengan dataset.
            ValueError: Jika panjang input_vector tidak sesuai dengan data latih.
        """
        if not self.train_data:
            raise RuntimeError("Model ScratchKNN belum dilatih. Panggil fit() terlebih dahulu.")

        expected_dim = len(self.train_data[0]["features"])
        if len(input_vector) != expected_dim:
            raise ValueError(
                f"Dimensi vektor input ({len(input_vector)}) tidak sesuai dengan dimensi dataset ({expected_dim})."
            )

        # 1 & 2. Hitung jarak Euclidean ke seluruh sampel data latih
        distances: List[Tuple[float, str]] = []
        for sample in self.train_data:
            dist = euclidean_distance(input_vector, sample["features"])
            distances.append((dist, str(sample["label"])))

        # Urutkan berdasarkan jarak terkecil ke terbesar (Ascending Sort)
        distances.sort(key=lambda item: item[0])

        # 3. Ambil K tetangga terdekat
        effective_k = min(self.k, len(distances))
        k_nearest = distances[:effective_k]

        # 4. Perhitungan suara mayoritas (Majority Voting)
        labels_in_k = [label for _, label in k_nearest]
        vote_counts = Counter(labels_in_k)

        max_vote = max(vote_counts.values())
        top_candidates = [
            label for label, count in vote_counts.items() if count == max_vote
        ]

        tie_resolved = False

        # 5. Mekanisme Resolusi Seri (Tie-Breaking Mechanism)
        if len(top_candidates) == 1:
            winner_label = top_candidates[0]
        else:
            tie_resolved = True
            epsilon = 1e-6
            # Hitung Inverse Distance Weighting untuk setiap kelas yang seri
            candidate_weights: Dict[str, float] = {}
            for cand in top_candidates:
                weight_sum = 0.0
                for dist, label in k_nearest:
                    if label == cand:
                        weight_sum += 1.0 / (dist + epsilon)
                candidate_weights[cand] = weight_sum

            # Pilih kandidat dengan akumulasi bobot jarak terdekat tertinggi
            # Jika bobot identik, tie-break kedua memilih tetangga berjarak absolut minimum
            def tie_sorter(candidate: str) -> Tuple[float, float]:
                # Inversi tanda bobot agar bobot terbesar berada di urutan pertama
                # Jarak minimum tetangga pertama kandidat sebagai penentu kedua
                first_occurrence_dist = next(
                    dist for dist, label in k_nearest if label == candidate
                )
                return (-candidate_weights[candidate], first_occurrence_dist)

            top_candidates.sort(key=tie_sorter)
            winner_label = top_candidates[0]

        # 6. Kalkulasi metrik confidence (suara pemenang / K) * 100
        winner_votes = vote_counts[winner_label]
        confidence = (winner_votes / effective_k) * 100.0

        return {
            "predicted_role": winner_label,
            "confidence": round(confidence, 2),
            "neighbors": k_nearest,
            "nearest_neighbors": k_nearest,  # Kompatibilitas mundur
            "tie_resolved": tie_resolved,
            "vote_counts": dict(vote_counts),
        }
