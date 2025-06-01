from PIL import Image
import pytesseract

def extract_text(image_path: str) -> str:
    """
    Liest den deutschen Text aus dem Bild unter `image_path` aus.
    """
    text = pytesseract.image_to_string(Image.open(image_path), lang="deu")
    return text


#sample_text = extract_text("/home/ubuntu/photo-translator/static/uploads/Screenshot 2025-05-26 174727.png")
#print(sample_text)
