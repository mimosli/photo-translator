from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Lädt das Bild und wendet Graustufen, Median-Blur und adaptive Thresholding an,
    um die OCR-Genauigkeit zu verbessern.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return thresh


def extract_text(image_path: str) -> str:
    """
    Basis-OCR mit Preprocessing und automatischer Block-Erkennung (PSM 3).
    - preserve_interword_spaces für exakte Wortabstände
    - gängiges PSM 3, um mehrere Text-Blöcke zu erkennen
    """
    img = preprocess_image(image_path)
    config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(
        img,
        lang="deu",
        config=config
    )
    return text


def extract_text_with_layout(image_path: str) -> str:
    """
    Fortgeschrittene OCR mit Layout:
    - Preprocessing für bessere Erkennung
    - Gruppenbildung per block_num und line_num
    - Trennt Absätze über unterschiedliche Blöcke (block_num)
    - Fügt nach jedem Block eine Leerzeile ein
    """
    img = preprocess_image(image_path)
    config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(
        img,
        lang="deu",
        config=config,
        output_type=Output.DICT
    )

    # Struktur: blocks[block_num][line_num] -> Liste[(left, word)]
    blocks = {}
    n = len(data['text'])
    for i in range(n):
        w = data['text'][i].strip()
        if not w:
            continue
        blk = data['block_num'][i]
        ln = data['line_num'][i]
        left = data['left'][i]
        blocks.setdefault(blk, {}).setdefault(ln, []).append((left, w))

    lines = []
    for blk in sorted(blocks.keys()):
        for ln in sorted(blocks[blk].keys()):
            words = sorted(blocks[blk][ln], key=lambda x: x[0])
            line = " ".join(word for (_, word) in words)
            lines.append(line)
        # Leerzeile trennend nach jedem Block
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)

# Auskommentierte Tests
# text_simple = extract_text("static/uploads/poetry.png")
# text_layout = extract_text_with_layout("static/uploads/poetry.png")
# print(text_simple)
# print("--- Layout ---")
# print(text_layout)
