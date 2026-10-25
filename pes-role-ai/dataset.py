"""
dataset.py
=============================================================================
DATASET & METADATA PROFIL TAKTIK PES 2026
Proyek Akademik: PES 2026 Tactical Role Predictor
Kelompok 6 (Wong Urakan):
  - Ahmad Rizqi Mubarok (06.2025.1.07900)
  - Wahyu Aditia Putra  (06.2025.1.07869)
Mata Kuliah: Kecerdasan Artifisial
Dosen Pengampu: Pak Kurniawan

Representasi Vektor Fitur (Skala 1.0 - 5.0):
  Indeks 0: Agresivitas & Pengambilan Risiko (FPS, Olahraga Ekstrem)
  Indeks 1: Visi Strategis & Kreativitas (Catur, Puzzle, Desain/Seni)
  Indeks 2: Disiplin Defensif & Kesabaran (Membaca, Analisis Data, Koleksi)
  Indeks 3: Kerjasama Tim & Distribusi (Game Co-op, Organisasi)
  Indeks 4: Refleks Instan & Tanggap Krisis (Tenis Meja, Reaksi Cepat)

Total Sampel: 32 data latih seimbang (8 sampel per kelas pada 4 kategori peran).
=============================================================================
"""

from typing import Any, Dict, List

# Dataset Latih Terdistribusi Seimbang (Balanced Classes: 8 x 4 = 32 Sampel)
TRAINING_DATA: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # KELAS 1: PENYERANG (FORWARD) - 8 Sampel
    # Ciri: Agresivitas tinggi, Refleks tajam, Disiplin defensif minimal.
    # -------------------------------------------------------------------------
    {"features": [5.0, 3.0, 1.0, 2.0, 4.0], "label": "Forward"},  # Poacher Klasik
    {"features": [5.0, 4.0, 2.0, 2.0, 3.0], "label": "Forward"},  # Prolific Winger
    {"features": [4.0, 3.0, 2.0, 1.0, 5.0], "label": "Forward"},  # Target Man Fisikal
    {"features": [5.0, 2.0, 1.0, 3.0, 4.0], "label": "Forward"},  # Goal Poacher Kecepatan
    {"features": [4.0, 4.0, 1.0, 3.0, 3.0], "label": "Forward"},  # Inside Forward
    {"features": [5.0, 5.0, 1.0, 2.0, 3.0], "label": "Forward"},  # Complete Forward
    {"features": [4.0, 4.0, 2.0, 3.0, 4.0], "label": "Forward"},  # Second Striker (SS)
    {"features": [5.0, 2.0, 1.0, 1.0, 4.0], "label": "Forward"},  # Finisher Instingtif

    # -------------------------------------------------------------------------
    # KELAS 2: GELANDANG (MIDFIELDER) - 8 Sampel
    # Ciri: Visi strategis & kreativitas prima, kerjasama tim tinggi.
    # -------------------------------------------------------------------------
    {"features": [2.0, 5.0, 3.0, 5.0, 3.0], "label": "Midfielder"},  # Orchestrator (CMF)
    {"features": [3.0, 5.0, 2.0, 5.0, 2.0], "label": "Midfielder"},  # Classic No. 10 (AMF)
    {"features": [2.0, 4.0, 4.0, 5.0, 3.0], "label": "Midfielder"},  # Box-to-Box (CMF)
    {"features": [3.0, 4.0, 3.0, 4.0, 2.0], "label": "Midfielder"},  # Roaming Playmaker
    {"features": [1.0, 5.0, 3.0, 5.0, 4.0], "label": "Midfielder"},  # Creative Wide Midfielder
    {"features": [3.0, 5.0, 4.0, 4.0, 3.0], "label": "Midfielder"},  # Anchor / Deep Pivot
    {"features": [2.0, 5.0, 3.0, 4.0, 3.0], "label": "Midfielder"},  # Mezzala Taktikal
    {"features": [1.0, 5.0, 2.0, 5.0, 2.0], "label": "Midfielder"},  # Pure Maestro Distribusi

    # -------------------------------------------------------------------------
    # KELAS 3: BERTAHAN (DEFENDER) - 8 Sampel
    # Ciri: Disiplin defensif & kesabaran maksimal, kalkulasi posisi terukur.
    # -------------------------------------------------------------------------
    {"features": [3.0, 2.0, 5.0, 3.0, 2.0], "label": "Defender"},  # Build-up Centre Back
    {"features": [2.0, 2.0, 5.0, 4.0, 3.0], "label": "Defender"},  # Defensive Fullback
    {"features": [4.0, 1.0, 5.0, 2.0, 2.0], "label": "Defender"},  # The Destroyer (CB Agresif)
    {"features": [2.0, 3.0, 5.0, 3.0, 1.0], "label": "Defender"},  # Sweeper / Libero
    {"features": [3.0, 1.0, 4.0, 4.0, 3.0], "label": "Defender"},  # Stopper Fisikal
    {"features": [1.0, 2.0, 5.0, 3.0, 2.0], "label": "Defender"},  # Positional Defender
    {"features": [2.0, 1.0, 5.0, 3.0, 2.0], "label": "Defender"},  # Low-Block Specialist
    {"features": [1.0, 1.0, 5.0, 2.0, 1.0], "label": "Defender"},  # Iron Fortress CB

    # -------------------------------------------------------------------------
    # KELAS 4: PENJAGA GAWANG (GOALKEEPER) - 8 Sampel
    # Ciri: Refleks instan puncak, ketenangan krisis tinggi, minim agresi menyerang.
    # -------------------------------------------------------------------------
    {"features": [1.0, 2.0, 4.0, 2.0, 5.0], "label": "Goalkeeper"},  # Defensive Goalkeeper
    {"features": [2.0, 1.0, 4.0, 1.0, 5.0], "label": "Goalkeeper"},  # Shot-Stopper Reaktif
    {"features": [1.0, 3.0, 3.0, 2.0, 5.0], "label": "Goalkeeper"},  # Offensive Goalkeeper (Sweeper GK)
    {"features": [2.0, 2.0, 5.0, 2.0, 5.0], "label": "Goalkeeper"},  # Command-the-Box GK
    {"features": [1.0, 1.0, 4.0, 3.0, 5.0], "label": "Goalkeeper"},  # Modern Hybrid GK
    {"features": [2.0, 2.0, 3.0, 1.0, 5.0], "label": "Goalkeeper"},  # Acrobatic Crisis Responser
    {"features": [1.0, 2.0, 5.0, 1.0, 5.0], "label": "Goalkeeper"},  # Penalty Specialist GK
    {"features": [1.0, 1.0, 3.0, 1.0, 5.0], "label": "Goalkeeper"},  # Lightning Reflex GK
]

# Metadata Taktikal Lengkap PES 2026
TACTICAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "Forward": {
        "role_pes": "Centre Forward (CF) / Prolific Winger (LWF/RWF) / Goal Poacher",
        "formation": "4-3-3 Attacking atau 3-4-3 Fluida",
        "playstyle": "Quick Counter / Gegenpressing Intensif",
        "desc": (
            "Karakter eksekutor tajam berani mengambil risiko tinggi di sepertiga akhir lapangan. "
            "Memiliki dorongan instingtif untuk menekan garis pertahanan lawan secara agresif, "
            "memanfaatkan ruang sempit, dan menuntaskan peluang dalam hitungan detik."
        ),
        "instructions": (
            "Gaya Menyerang: Serangan Balik Cepat (Quick Counter); Area Membangun Serangan: Tengah & Sayap; "
            "Penyusunan Posisi: Fleksibel; Garis Pertahanan: Tinggi (Gegenpress)."
        ),
    },
    "Midfielder": {
        "role_pes": "Classic No. 10 (AMF) / Orchestrator (CMF) / Box-to-Box (CMF)",
        "formation": "4-2-3-1 Modern atau 4-3-3 Ball Possession",
        "playstyle": "Possession Game / Fluid Tiki-Taka",
        "desc": (
            "Arsitek tempo dan metronom sirkulasi bola tim. Karakter ini unggul dalam visi spasial makro, "
            "akurasi distribusi progresi, serta kemampuan menghubungkan transisi bertahan menuju fase "
            "penetrasi dengan perhitungan matang dan kerjasama kolektif."
        ),
        "instructions": (
            "Gaya Menyerang: Penguasaan Bola (Possession); Jarak Dukungan: Pendek; "
            "Penyusunan Posisi: Terstruktur; Instruksi Lanjutan: Centering Targets / Tiki-Taka."
        ),
    },
    "Defender": {
        "role_pes": "Build-up Centre Back (CB) / The Destroyer (CB) / Defensive Fullback",
        "formation": "3-5-2 Solid atau 5-3-2 Low-Block",
        "playstyle": "Long Ball Counter / Low-Block Defense",
        "desc": (
            "Pilar dinding pertahanan yang mengutamakan kedisiplinan posisi, kesabaran taktis, dan "
            "intersepsi terukur. Memiliki resistensi tinggi terhadap provokasi lawan serta fokus penuh "
            "menjaga kerapatan garis belakang untuk meminimalisasi celah berbahaya."
        ),
        "instructions": (
            "Gaya Bertahan: All-Out Defense; Area Tekanan: Tengah; "
            "Garis Pertahanan: Rendah (Deep Block); Kepadatan: Sangat Rapat (Compactness Level 8-9)."
        ),
    },
    "Goalkeeper": {
        "role_pes": "Offensive Goalkeeper (Sweeper GK) / Defensive Goalkeeper (GK)",
        "formation": "4-4-2 Solid Compact Defense atau 4-3-3 Balance",
        "playstyle": "All-Out Defense / Counter Attack Support",
        "desc": (
            "Garda terakhir penyelamat krisis dengan refleks motorik kilat. Mampu membaca arah ancaman "
            "secara instan, mengambil keputusan krusial di bawah tekanan ekstrim, dan mempertahankan "
            "fokus mental tanpa kompromi sepanjang 90 menit pertandingan."
        ),
        "instructions": (
            "Gaya Bertahan: Penjagaan Garis Gawang Ketat; Distribusi: Tendangan Jauh Akurat / Lemparan Cepat; "
            "Karakteristik Khusus: Penyelamatan Refleks Jarak Dekat & Antisipasi Bola Mati."
        ),
    },
}
