"""
test_knn.py
=============================================================================
UNIT TESTING MANDIRI BERBASIS CLI - EVALUASI ALGORITMA SCRATCH KNN
Proyek Akademik: PES 2026 Tactical Role Predictor
Kelompok 6 (Wong Urakan):
  - Ahmad Rizqi Mubarok (06.2025.1.07900)
  - Wahyu Aditia Putra  (06.2025.1.07869)
Mata Kuliah: Kecerdasan Artifisial
Dosen Pengampu: Pak Kurniawan

Fungsi Pengujian:
1. Memvalidasi 4 skenario data ekstrem secara deterministik tanpa server web.
2. Memvalidasi mekanisme pemecah voting seri (Tie-Breaking via IDW).
3. Memvalidasi penanganan error dimensi dan integritas data input.
4. Mencetak laporan hasil dalam tabel ASCII terformat rapi.
=============================================================================
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# Pastikan modul internal dapat diimpor tanpa dependensi CWD
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core_ai import ScratchKNN, euclidean_distance, min_max_normalize
from dataset import TRAINING_DATA


def run_test_suite() -> bool:
    """
    Mengeksekusi seluruh pengujian skenario ekstrem dan integritas algoritma.

    Mengembalikan:
        bool: True jika seluruh pengujian berstatus PASS, False jika ada yang FAIL.
    """
    print("\n" + "=" * 80)
    print(" UNIT TESTING ALGORITMA K-NEAREST NEIGHBORS (SCRATCH KNN) - PES 2026")
    print(" Kelompok 6: Wong Urakan | Ahmad Rizqi Mubarok & Wahyu Aditia Putra")
    print(" Mata Kuliah: Kecerdasan Artifisial | Dosen: Pak Kurniawan")
    print("=" * 80)

    # Inisialisasi Model dengan K = 5
    knn = ScratchKNN(k=5)
    knn.fit(TRAINING_DATA)

    # 4 Skenario Wajib Kasus Ekstrem Karakteristik
    test_cases: List[Dict[str, Any]] = [
        {
            "id": 1,
            "nama": "Penyerang Murni (Forward)",
            "vector": [5.0, 1.0, 1.0, 1.0, 2.0],
            "target": "Forward",
            "keterangan": "Agresivitas tinggi, disiplin & visi rendah",
        },
        {
            "id": 2,
            "nama": "Playmaker Visioner (Midfielder)",
            "vector": [1.0, 5.0, 2.0, 5.0, 2.0],
            "target": "Midfielder",
            "keterangan": "Kreativitas & kerjasama tim dominan",
        },
        {
            "id": 3,
            "nama": "Tembok Pertahanan (Defender)",
            "vector": [1.0, 1.0, 5.0, 2.0, 1.0],
            "target": "Defender",
            "keterangan": "Disiplin defensif & kesabaran maksimal",
        },
        {
            "id": 4,
            "nama": "Penjaga Gawang Responsif (Goalkeeper)",
            "vector": [1.0, 1.0, 2.0, 1.0, 5.0],
            "target": "Goalkeeper",
            "keterangan": "Refleks krisis puncak, agresi menyerang minimal",
        },
    ]

    print("\n[FASE 1] PENGUJIAN 4 SKENARIO DATA EKSTREM (K=5):")
    print("-" * 88)
    header = f"| {'No':^3} | {'Nama Skenario':<35} | {'Target':<10} | {'Aktual':<10} | {'Conf':^7} | {'Status':^8} |"
    print(header)
    print("-" * 88)

    all_passed = True
    results_detail = []

    for tc in test_cases:
        res = knn.predict(tc["vector"])
        actual = res["predicted_role"]
        confidence = res["confidence"]
        status = "PASS" if actual == tc["target"] else "FAIL"

        if status == "FAIL":
            all_passed = False

        row = f"| {tc['id']:^3} | {tc['nama']:<35} | {tc['target']:<10} | {actual:<10} | {confidence:>5.1f}% | [{status:^4}] |"
        print(row)
        results_detail.append((tc, res))

    print("-" * 88)

    # [FASE 2] PENGUJIAN DETAIL LOG TETANGGA TERDEKAT (VERIFIKASI EUCLIDEAN DISTANCE)
    print("\n[FASE 2] LOG 5 TETANGGA TERDEKAT DARI SETIAP SKENARIO EKSTREM:")
    for tc, res in results_detail:
        print(f"\n* Skenario #{tc['id']} : {tc['nama']}")
        print(f"  Vektor Input : {tc['vector']}")
        print(f"  Prediksi     : {res['predicted_role']} (Confidence: {res['confidence']}%)")
        print("  Tetangga Terdekat:")
        for rank, (dist, label) in enumerate(res["neighbors"], start=1):
            print(f"    - Rank {rank}: Jarak Euclidean = {dist:.4f} | Label = {label}")

    # [FASE 3] PENGUJIAN MEKANISME RESOLUSI SERI (TIE-BREAKING VIA IDW)
    print("\n" + "-" * 88)
    print("[FASE 3] PENGUJIAN MEKANISME DETERMINISTIK TIE-BREAKING (INVERSE DISTANCE WEIGHTING):")
    # Siapkan dataset sintetis dengan 2 kelas berbobot seimbang
    synthetic_knn = ScratchKNN(k=2)
    synthetic_data = [
        {"features": [1.0, 1.0], "label": "Forward"},
        {"features": [3.0, 3.0], "label": "Defender"},
    ]
    synthetic_knn.fit(synthetic_data)

    # Uji titik [1.2, 1.2] yang lebih dekat ke Forward daripada Defender
    query_point = [1.2, 1.2]
    tie_res = synthetic_knn.predict(query_point)

    tie_status = "PASS" if (tie_res["predicted_role"] == "Forward" and tie_res["tie_resolved"]) else "FAIL"
    if tie_status == "FAIL":
        all_passed = False

    print(f"  Titik Uji        : {query_point}")
    print(f"  Jumlah Voting    : {tie_res['vote_counts']} (Kondisi Seri: 1 vs 1 pada K=2)")
    print(f"  Flag Tie-Break   : {tie_res['tie_resolved']}")
    print(f"  Pemenang Akhir   : {tie_res['predicted_role']} (Expected: Forward via IDW)")
    print(f"  Status Tie-Break : [{tie_status}]")

    # [FASE 4] PENGUJIAN ROBUSTNESS & VALIDASI INPUT ERROR
    print("\n" + "-" * 88)
    print("[FASE 4] PENGUJIAN ROBUSTNESS ERROR HANDLING:")
    error_tests_passed = True
    try:
        # Uji vektor dengan dimensi tidak cocok (3 elemen vs 5 elemen)
        knn.predict([1.0, 2.0, 3.0])
        print("  Uji Dimensi Mismatch   : [FAIL] - Tidak memunculkan ValueError.")
        error_tests_passed = False
    except ValueError:
        print("  Uji Dimensi Mismatch   : [PASS] - ValueError berhasil ditangkap dengan benar.")

    try:
        # Uji normalisasi dengan rentang salah (max <= min)
        min_max_normalize([1.0, 2.0], min_val=5.0, max_val=1.0)
        print("  Uji Batas Normalisasi  : [FAIL] - Tidak memunculkan ValueError.")
        error_tests_passed = False
    except ValueError:
        print("  Uji Batas Normalisasi  : [PASS] - ValueError berhasil ditangkap dengan benar.")

    if not error_tests_passed:
        all_passed = False

    # REKAPITULASI HASIL TESTING
    print("\n" + "=" * 88)
    print(" REKAPITULASI HASIL EVALUASI UNIT TESTING AKADEMIK")
    print("=" * 88)
    if all_passed:
        print(" >>> KESIMPULAN: SELURUH PENGUJIAN MEMENUHI STANDAR (100% PASS) <<<")
        print(" Algoritma ScratchKNN terbukti robust, deterministik, dan bebas pustaka eksternal.")
        print("=" * 88 + "\n")
        return True
    else:
        print(" >>> KESIMPULAN: ADA PENGUJIAN YANG GAGAL (FAIL) <<<")
        print("=" * 88 + "\n")
        return False


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)

