print('hello world')

from flask import Flask, request, jsonify, redirect, render_template, send_from_directory
import os
from ocr import extract_text
from translate import translate_with_deepl  # or use translate_with_gpt
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#@app.route('/')
#def index():
#    return 'It works, yeyy!'

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the image
    img = Image.open(filepath)
    img = img.convert('L')  # Example: convert to grayscale
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
    img.save(processed_path)

    return f'''
    <h1>Image processed!</h1>
    <p>Original:</p>
    <img src="/static/uploads/{file.filename}" width="300">
    <p>Processed:</p>
    <img src="/static/uploads/processed_{file.filename}" width="300">
    '''

@app.route("/translate", methods=["POST"])
def translate_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    german_text = extract_text(filepath)
    translated_text = translate_with_deepl(german_text)  # or translate_with_gpt

    return jsonify({
        "original_text": german_text.strip(),
        "translated_text": translated_text.strip()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)








