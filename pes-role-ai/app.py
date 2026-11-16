import logging
import os
import sys
from typing import Any

from flask import Flask, jsonify, render_template, request

# Pastikan path absolut konsisten di berbagai level direktori eksekusi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core_ai import ScratchKNN
from dataset import TACTICAL_PROFILES, TRAINING_DATA

# Inisialisasi Flask dengan path eksplisit untuk folder templates dan static
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Konfigurasi Logging Terminal Akademik
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Kelompok 6 - PES AI) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PES_Tactical_AI")

# Inisialisasi Model AI KNN Scratch dengan K = 5
KNN_K_VALUE = 5
knn_classifier = ScratchKNN(k=KNN_K_VALUE)
knn_classifier.fit(TRAINING_DATA)
logger.info(
    "Model ScratchKNN berhasil diinisialisasi dengan K=%d dan %d data latih seimbang.",
    KNN_K_VALUE,
    len(TRAINING_DATA),
)

# Definisi nama-nama fitur untuk validasi dan logging
FEATURE_NAMES: list[tuple[str, str]] = [
    ("q1", "Agresivitas"),
    ("q2", "Kreativitas"),
    ("q3", "Disiplin"),
    ("q4", "Kerjasama"),
    ("q5", "Refleks"),
]


def validate_and_parse_input(
    raw_value: str | None, field_label: str, default_val: float = 3.0
) -> float:
    """
    Memvalidasi dan mengonversi nilai input form ke tipe float dalam rentang [1.0, 5.0].

    Parameter:
        raw_value (Optional[str]): Nilai mentah yang diterima dari HTTP POST form.
        field_label (str): Nama label fitur untuk pesan diagnostik error.
        default_val (float): Nilai fallback aman jika input kosong (default: 3.0).

    Mengembalikan:
        float: Nilai numerik valid pada rentang [1.0, 5.0].

    Raises:
        ValueError: Jika tipe data bukan numerik atau berada di luar batasan [1.0, 5.0].
    """
    if raw_value is None or raw_value.strip() == "":
        return default_val

    try:
        parsed = float(raw_value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Input '{field_label}' bernilai '{raw_value}', yang bukan merupakan angka valid."
        ) from exc

    if parsed < 1.0 or parsed > 5.0:
        raise ValueError(
            f"Input '{field_label}' bernilai {parsed:.2f}. Skala parameter harus berada dalam rentang [1.0 - 5.0]."
        )

    return parsed


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handler utama antarmuka pengguna:
    - GET: Merender halaman formulir input dengan status awal bersih.
    - POST: Memvalidasi parameter karakter, mengeksekusi inferensi KNN,
            mencatat log ke terminal, dan merender visualisasi hasil.
    """
    prediction_result: dict[str, Any] | None = None
    input_scores: dict[str, float] | None = None
    error_message: str | None = None

    if request.method == "POST":
        try:
            # 1. Parsing dan validasi ketat setiap parameter karakter
            parsed_scores: dict[str, float] = {}
            input_vector: list[float] = []

            for field_key, field_label in FEATURE_NAMES:
                val = validate_and_parse_input(
                    request.form.get(field_key), field_label, default_val=3.0
                )
                parsed_scores[field_label] = val
                input_vector.append(val)

            input_scores = parsed_scores

            # 2. Eksekusi inferensi K-Nearest Neighbors mandiri
            inference = knn_classifier.predict(input_vector)
            role = inference["predicted_role"]
            profile = TACTICAL_PROFILES.get(role, {})

            prediction_result = {
                "role": role,
                "confidence": inference["confidence"],
                "role_pes": profile.get("role_pes", "Peran Taktis Dinamis"),
                "formation": profile.get("formation", "Formasi Fleksibel"),
                "playstyle": profile.get("playstyle", "Gaya Main Adaptif"),
                "desc": profile.get("desc", "Deskripsi profil tidak tersedia."),
                "instructions": profile.get(
                    "instructions", "Instruksi standar lapangan."
                ),
                "neighbors": inference["neighbors"],
                "tie_resolved": inference["tie_resolved"],
                "vote_counts": inference.get("vote_counts", {}),
            }

            # 3. Pencatatan log terstruktur ke terminal
            logger.info(
                "-----------------------------------------------------------------"
            )
            logger.info("INFERENSI PREDIKSI PERAN TAKTIS PES 2026")
            logger.info("Vektor Fitur Input : %s", input_vector)
            logger.info("Skor Parameter     : %s", parsed_scores)
            logger.info(
                "K-Tetangga (K=%d)  : %s",
                KNN_K_VALUE,
                [f"{label} (d={dist:.4f})" for dist, label in inference["neighbors"]],
            )
            logger.info("Distribusi Suara   : %s", inference.get("vote_counts"))
            logger.info(
                "Resolusi Seri (Tie): %s",
                "YA (IDW)" if inference["tie_resolved"] else "TIDAK",
            )
            logger.info(
                "Hasil Prediksi     : %s (Confidence: %.2f%%)",
                role,
                inference["confidence"],
            )
            logger.info(
                "-----------------------------------------------------------------"
            )

        except ValueError as val_err:
            error_message = str(val_err)
            logger.warning("Validasi input gagal: %s", error_message)
        except Exception as err:
            error_message = f"Terjadi kesalahan internal pada sistem inferensi: {err}"
            logger.exception("Pengecualian tak terduga pada server backend:")

    return render_template(
        "index.html",
        result=prediction_result,
        inputs=input_scores,
        error=error_message,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint AJAX JSON untuk inferensi prediksi peran taktis PES 2026.
    Menerima form fields q1-q5 via POST dan mengembalikan hasil prediksi
    dalam format JSON murni (tanpa render template).
    """
    try:
        # 1. Parsing dan validasi ketat setiap parameter karakter
        parsed_scores: dict[str, float] = {}
        input_vector: list[float] = []

        for field_key, field_label in FEATURE_NAMES:
            val = validate_and_parse_input(
                request.form.get(field_key), field_label, default_val=3.0
            )
            parsed_scores[field_label] = val
            input_vector.append(val)

        # 2. Eksekusi inferensi K-Nearest Neighbors mandiri
        inference = knn_classifier.predict(input_vector)
        role = inference["predicted_role"]
        profile = TACTICAL_PROFILES.get(role, {})

        # 3. Pencatatan log terstruktur ke terminal
        logger.info(
            "-----------------------------------------------------------------"
        )
        logger.info("INFERENSI PREDIKSI PERAN TAKTIS PES 2026 (JSON API)")
        logger.info("Vektor Fitur Input : %s", input_vector)
        logger.info("Skor Parameter     : %s", parsed_scores)
        logger.info(
            "K-Tetangga (K=%d)  : %s",
            KNN_K_VALUE,
            [f"{label} (d={dist:.4f})" for dist, label in inference["neighbors"]],
        )
        logger.info("Distribusi Suara   : %s", inference.get("vote_counts"))
        logger.info(
            "Resolusi Seri (Tie): %s",
            "YA (IDW)" if inference["tie_resolved"] else "TIDAK",
        )
        logger.info(
            "Hasil Prediksi     : %s (Confidence: %.2f%%)",
            role,
            inference["confidence"],
        )
        logger.info(
            "-----------------------------------------------------------------"
        )

        # 4. Bangun response JSON
        return jsonify(
            {
                "success": True,
                "role": role,
                "confidence": inference["confidence"],
                "role_pes": profile.get("role_pes", "Peran Taktis Dinamis"),
                "formation": profile.get("formation", "Formasi Fleksibel"),
                "playstyle": profile.get("playstyle", "Gaya Main Adaptif"),
                "desc": profile.get("desc", "Deskripsi profil tidak tersedia."),
                "instructions": profile.get(
                    "instructions", "Instruksi standar lapangan."
                ),
                "tie_resolved": inference["tie_resolved"],
                "vote_counts": inference.get("vote_counts", {}),
                "neighbors": [
                    {"rank": i + 1, "label": label, "distance": round(dist, 4)}
                    for i, (dist, label) in enumerate(inference["neighbors"])
                ],
                "inputs": parsed_scores,
            }
        )

    except ValueError as val_err:
        error_msg = str(val_err)
        logger.warning("Validasi input gagal (JSON API): %s", error_msg)
        return jsonify({"success": False, "error": error_msg}), 400
    except Exception as err:
        error_msg = f"Terjadi kesalahan internal pada sistem inferensi: {err}"
        logger.exception("Pengecualian tak terduga pada server backend (JSON API):")
        return jsonify({"success": False, "error": error_msg}), 500


if __name__ == "__main__":
    logger.info(
        "Server web PES 2026 Tactical Role Predictor siap dijalankan di http://127.0.0.1:5000"
    )
    app.run(debug=True, host="127.0.0.1", port=5000)
