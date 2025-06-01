from PIL import Image
import pytesseract

def extract_text(image_path: str) -> str:
    """
    Liest den deutschen Text aus dem Bild unter `image_path` aus.
    """
    # -c preserve_interword_spaces=1 lässt Pytesseract Leerzeichen weniger stark zusammenziehen
    custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(Image.open(image_path), lang="deu", config=custom_config)
    return text


#sample_text = extract_text("/home/ubuntu/photo-translator/static/uploads/Screenshot 2025-05-26 174727.png")
#print(sample_text)
