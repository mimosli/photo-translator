print('hello world')

from flask import Flask, request, jsonify, render_template, url_for
import os
import logging
from ocr import extract_text
from translate import translate_with_deepl  # or use translate_with_gpt
from PIL import Image

# set up JSON-style logs
logging.basicConfig(
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    level=logging.INFO
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def upload_form():
    """Zeigt das HTML-Formular zum Hochladen."""
    return render_template("upload.html")

@app.route("/translate", methods=["POST"])
def translate_image():
    """
    Empfängt ein Bild:
      1) speichert es,
      2) extrahiert per OCR deutschen Text,
      3) übersetzt mit DeepL,
      4) loggt Upload-Metadaten,
      5) rendert result.html.
    """
    # 1) file checks
    if "image" not in request.files:
        return "No file part", 400
    f = request.files["image"]
    if f.filename == "":
        return "No selected file", 400

    # 2) save file
    filename = f.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    # 3) OCR
    german_text = extract_text(filepath).strip()

    # 4) translate
    translated_text = translate_with_deepl(german_text).strip()

    # 5) log metadata
    metadata = {
        "client_ip":    request.remote_addr,
        "filename":     filename,
        "user_agent":   request.headers.get("User-Agent")
    }
    logging.info(f"upload_metadata: {metadata}")

    # 6) render result
    return render_template(
        "result.html",
        original_image  = url_for("static", filename=f"uploads/{filename}"),
        extracted_text  = german_text,
        translated_text = translated_text
    )

@app.route("/healthz")
def healthz():
    return jsonify(status="ok", version="1.0"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
