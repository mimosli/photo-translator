from PIL import Image
import pytesseract

def extract_text(image_path):
    print(pytesseract.image_to_string(Image.open(image_path), lang='deu'))
    return pytesseract.image_to_string(Image.open(image_path), lang='deu')



sample_text = extract_text("/home/ubuntu/photo-translator/static/uploads/Screenshot 2025-05-26 174727.png")
print(sample_text)