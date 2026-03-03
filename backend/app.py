# ═══════════════════════════════════════════
# FloraVision — Flask App
# Serves the full frontend + REST API
# Entry point: http://127.0.0.1:5000
# ═══════════════════════════════════════════

import os
import re
import threading
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
try:
    import gdown
except ImportError:
    gdown = None

from model import load_model, predict


# APP SETUP


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)
CORS(app)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# MODEL AUTO-DOWNLOAD


MODEL_PATH     = os.path.join(BASE_DIR, "resnet50_flower_model.pth")
GDRIVE_FILE_ID = "16GChkDpZgbmdSCyQFffOHx0D_rHtaEsj"


def _extract_drive_file_id(file_id_or_url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", file_id_or_url or "")
    if match:
        return match.group(1)
    return file_id_or_url.strip()

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        print("Model file already present — skipping download.")
        return True
    print("Model file not found — downloading from Google Drive...")
    if gdown is None:
        print("ERROR downloading model: gdown is not installed.")
        return False
    try:
        file_id = _extract_drive_file_id(GDRIVE_FILE_ID)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
        return True
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        traceback.print_exc()
        return False


# LOAD MODEL AT STARTUP


model = None
MODEL_READY = False
MODEL_ERROR = None
MODEL_LOCK = threading.Lock()


def ensure_model_loaded() -> bool:
    global model, MODEL_READY, MODEL_ERROR
    if MODEL_READY and model is not None:
        return True

    with MODEL_LOCK:
        if MODEL_READY and model is not None:
            return True

        try:
            download_model_if_needed()
            model = load_model()
            MODEL_READY = True
            MODEL_ERROR = None
            print("Model loaded successfully. 102 classes available.")
            return True
        except Exception as e:
            MODEL_READY = False
            MODEL_ERROR = str(e)
            print(f"ERROR loading model: {e}")
            traceback.print_exc()
            return False


# FRONTEND ROUTES


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/about")
@app.route("/about.html")
def about():
    return send_from_directory(FRONTEND_DIR, "about.html")

@app.route("/identify")
@app.route("/identify.html")
def identify():
    return send_from_directory(FRONTEND_DIR, "identify.html")



# API ROUTES


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_READY,
        "model_error": MODEL_ERROR
    }), 200


@app.route("/predict", methods=["POST"])
def predict_route():
    if not ensure_model_loaded():
        return jsonify({
            "error": "Model not loaded. Please ensure resnet50_flower_model.pth is in the backend/ folder.",
            "detail": MODEL_ERROR
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send an image in a 'file' field."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select a valid image file."}), 400

    mime_type = file.content_type or ""
    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({
            "error": f"Unsupported file type '{mime_type}'. Please upload a JPG, PNG, or WEBP image."
        }), 415

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        size_mb = len(image_bytes) / (1024 * 1024)
        return jsonify({
            "error": f"File too large ({size_mb:.1f}MB). Maximum size is 10MB."
        }), 413

    if len(image_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = predict(image_bytes, model)
        return jsonify(result), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed. Please try a different image.",
            "detail": str(e)
        }), 500



# ENTRY POINT


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
