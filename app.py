print('hello world')

from flask import Flask, request, jsonify, redirect, render_template, send_from_directory, url_for
import os
from ocr import extract_text
from translate import translate_with_deepl  # or use translate_with_gpt
from PIL import Image


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#@app.route('/')
#def index():
#    return 'It works, yeyy!'

@app.route("/")
def upload_form():
    """
    Zeigt das HTML-Formular zum Hochladen.
    """
    return render_template("upload.html")

@app.route("/translate", methods=["POST"])
def translate_image():
    """
    Empfängt das Hochlade-Formular, speichert das Bild,
    führt OCR aus und übersetzt den Text.
    Dann wird result.html gerendert.
    """
    if "image" not in request.files:
        return "No file part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    # Speichere das Originalbild
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # OCR: Deutschen Text erkennen
    german_text = extract_text(filepath).strip()

    # Übersetzen lassen (DeepL)
    translated_text = translate_with_deepl(german_text).strip()

    # Bereite Daten für das Template vor
    return render_template(
        "result.html",
        original_image=url_for("static", filename=f"uploads/{filename}"),
        extracted_text=german_text,
        translated_text=translated_text
    )

if __name__ == "__main__":
    # Im Debug-Modus laufen lassen, Port 5001 (oder 5000)
    app.run(host="0.0.0.0", port=5001, debug=True)


