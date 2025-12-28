# the idea of that simple ocr code is to keep things simple and using OpenCV and pytesseract only, but with improvement logics such as clear preprocessing and lyout-aware text extraction
# 

""" 
- Optional page detection + perspective warp (best for photos of a page)
- Deskew (fixes slight rotation)
- Better denoise (NLMeans)
- Adaptive threshold + small morphology
- Optional upscale 
- Text-region OCR (OCR only the text blocks, not the whole photo)
- Confidence + quality gate (so you can detect garbage and ask the user to retake) """


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

# -----------------------------
# Configuration knobs (tweak)
# -----------------------------

@dataclass
class OcrSettings:
    lang: str = "deu"
    oem: int = 3

    # PSM: 6 = single uniform block, 3 = automatic layout
    # For poems, 6 is often better than 3 once you crop / deskew.
    psm_full: int = 6 # poems are only printed not handwritten
    psm_block: int = 6

    preserve_spaces: bool = True

    # Preprocessing
    use_page_warp: bool = True        # try to detect page and warp
    use_deskew: bool = True           # fix slight rotation
    use_adaptive_thresh: bool = True  # better under uneven lighting
    denoise_h: int = 14               # NLMeans strength (10-20 typical)
    morph_close: bool = True
    morph_kernel: Tuple[int, int] = (2, 2)
    upscale_if_small: bool = True
    min_upscale: float = 1.6          # 1.3–2.5 typical
    max_dim_after: int = 2600         # keep images from becoming huge

    # Text region detection (OCR only text)
    use_text_regions: bool = True
    min_region_area: int = 800        # filter tiny blobs
    max_region_count: int = 40        # safety cap

    # Quality gate (reject garbage)
    min_chars: int = 20
    min_alpha_ratio: float = 0.55     # letters / all characters
    min_mean_conf: float = 35.0       # average tesseract confidence


DEFAULT = OcrSettings()


# -----------------------------
# Utilities
# -----------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts shape (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _try_page_warp(bgr: np.ndarray) -> np.ndarray:
    """
    Attempts to find the page contour and warp it.
    Falls back to original if no good quad found.
    """
    img = bgr.copy()
    h, w = img.shape[:2]
    # Downscale for contour detection speed
    scale = 900.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        small = img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    best_quad = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            best_quad = approx.reshape(4, 2)
            break

    if best_quad is None:
        return bgr

    # Scale quad back to original size if we resized
    if scale < 1.0:
        best_quad = best_quad / scale

    try:
        return _four_point_transform(bgr, best_quad.astype("float32"))
    except Exception:
        return bgr

def _tess_config(psm: int = 6) -> str:
    # user_defined_dpi helps when DPI metadata is missing
    return f'--oem 3 --psm {psm} -c preserve_interword_spaces=1 -c user_defined_dpi=300'


def _mean_conf(data: dict) -> float:
    confs = []
    for c in data.get("conf", []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except:
            pass
    return float(np.mean(confs)) if confs else 0.0

def _deskew(gray: np.ndarray) -> np.ndarray:
    # Estimate skew with Otsu + minAreaRect
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th
    coords = np.column_stack(np.where(inv > 0))
    if coords.size < 300:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.7:
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast boost – great for printed pages."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _preprocess_candidates(image_path: str) -> list[np.ndarray]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Use page warp here too (helps book photos)
    bgr = _try_page_warp(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # denoise while preserving edges
    gray = cv2.fastNlMeansDenoising(gray, h=12)
    gray = _enhance_contrast(gray)
    gray = _deskew(gray)

    # Candidate A: grayscale (often best for printed text)
    cand_gray = gray

    # Candidate B: Otsu (clean binary)
    _, cand_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Candidate C: Adaptive (better for uneven lighting)
    cand_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=9
    )

    # Tiny close to connect broken strokes (light touch)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cand_otsu = cv2.morphologyEx(cand_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    cand_adapt = cv2.morphologyEx(cand_adapt, cv2.MORPH_CLOSE, kernel, iterations=1)

    return [cand_gray, cand_otsu, cand_adapt]

def _maybe_upscale(gray: np.ndarray, settings: OcrSettings) -> np.ndarray:
    if not settings.upscale_if_small:
        return gray

    h, w = gray.shape[:2]
    # Heuristic: if max dimension is small-ish, upscale
    if max(h, w) < 1200:
        scale = settings.min_upscale
    else:
        scale = 1.0

    if scale <= 1.01:
        return gray

    new_w = int(w * scale)
    new_h = int(h * scale)

    # guardrail
    max_dim = max(new_w, new_h)
    if max_dim > settings.max_dim_after:
        factor = settings.max_dim_after / max_dim
        new_w = int(new_w * factor)
        new_h = int(new_h * factor)

    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _binarize(gray: np.ndarray, settings: OcrSettings) -> np.ndarray:
    if settings.use_adaptive_thresh:
        # Adaptive handles shadows better
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=9
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if settings.morph_close:
        kx, ky = settings.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return th


def preprocess_image(image_path: str, settings: OcrSettings = DEFAULT) -> np.ndarray:
    """
    Loads image and applies:
    page warp -> grayscale -> denoise -> deskew -> optional upscale -> threshold -> morphology
    Returns a binary image suitable for OCR (uint8 0/255).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    if settings.use_page_warp:
        bgr = _try_page_warp(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Denoise (keeps edges better than median blur for many phone photos)
    if settings.denoise_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=settings.denoise_h)

    if settings.use_deskew:
        gray = _deskew(gray)

    gray = _maybe_upscale(gray, settings)
    th = _binarize(gray, settings)
    return th


# -----------------------------
# Text region extraction
# -----------------------------

def _find_text_regions(binary: np.ndarray, settings: OcrSettings) -> List[Tuple[int, int, int, int]]:
    """
    Finds text-like regions on the binary image.
    Returns list of bounding boxes (x, y, w, h) sorted top-to-bottom.
    """
    # We want foreground = white for morphology grouping
    inv = 255 - binary

    # Connect characters into words/lines a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    connected = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    H, W = binary.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < settings.min_region_area:
            continue
        # Filter super thin lines / weird shapes
        if h < 12 or w < 30:
            continue
        # Filter regions that are almost entire page (often background)
        if w > 0.98 * W and h > 0.98 * H:
            continue
        boxes.append((x, y, w, h))

    # Sort by y, then x
    boxes.sort(key=lambda b: (b[1], b[0]))

    # Safety cap
    if len(boxes) > settings.max_region_count:
        boxes = boxes[: settings.max_region_count]

    return boxes


def _tesseract_config(settings: OcrSettings, psm: int) -> str:
    parts = [f"--oem {settings.oem}", f"--psm {psm}"]
    if settings.preserve_spaces:
        parts.append("-c preserve_interword_spaces=1")
    return " ".join(parts)


def _quality_metrics(text: str, confs: List[int]) -> Tuple[float, float, int]:
    """
    Returns: (alpha_ratio, mean_conf, char_count)
    alpha_ratio: fraction of letters among all non-space chars
    """
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if not cleaned:
        return 0.0, 0.0, 0
    letters = sum(ch.isalpha() for ch in cleaned)
    alpha_ratio = letters / max(1, len(cleaned))

    valid_confs = [c for c in confs if c >= 0]
    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
    return alpha_ratio, mean_conf, len(cleaned)


# -----------------------------
# Public OCR functions
# -----------------------------

def extract_text(image_path: str, settings: OcrSettings = DEFAULT) -> str:
    """
    OCR that uses improved preprocessing.
    Uses full-image OCR by default (faster, simpler).
    """
    img = preprocess_image(image_path, settings=settings)
    config = _tesseract_config(settings, psm=settings.psm_full)
    text = pytesseract.image_to_string(img, lang=settings.lang, config=config)
    return text


def extract_text_with_layout(image_path: str, settings: OcrSettings = DEFAULT) -> str:
    """
    OCR that preserves a line/block-ish structure by using image_to_data and grouping.
    Uses improved preprocessing.
    """
    img = preprocess_image(image_path, settings=settings)
    config = _tesseract_config(settings, psm=settings.psm_full)

    data = pytesseract.image_to_data(
        img,
        lang=settings.lang,
        config=config,
        output_type=Output.DICT
    )

    blocks = {}
    n = len(data["text"])
    for i in range(n):
        w = (data["text"][i] or "").strip()
        if not w:
            continue
        blk = data["block_num"][i]
        ln = data["line_num"][i]
        left = data["left"][i]
        blocks.setdefault(blk, {}).setdefault(ln, []).append((left, w))

    lines: List[str] = []
    for blk in sorted(blocks.keys()):
        for ln in sorted(blocks[blk].keys()):
            words = sorted(blocks[blk][ln], key=lambda x: x[0])
            line = " ".join(word for (_, word) in words)
            lines.append(line)
        lines.append("")  # blank line between blocks

    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def extract_text_regions(image_path: str, settings: OcrSettings = DEFAULT) -> Tuple[str, dict]:
    """
    Best-quality mode: detect text regions and OCR region-by-region.
    Returns (text, metrics_dict).
    """
    binary = preprocess_image(image_path, settings=settings)
    if not settings.use_text_regions:
        # fallback to full
        text = extract_text_with_layout(image_path, settings=settings)
        return text, {"mode": "full"}

    boxes = _find_text_regions(binary, settings=settings)
    if not boxes:
        text = extract_text_with_layout(image_path, settings=settings)
        return text, {"mode": "full_fallback", "regions": 0}

    config = _tesseract_config(settings, psm=settings.psm_block)

    texts: List[str] = []
    confs_all: List[int] = []

    for (x, y, w, h) in boxes:
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(binary.shape[1], x + w + pad)
        y1 = min(binary.shape[0], y + h + pad)
        roi = binary[y0:y1, x0:x1]

        # Use image_to_data to also get confidence
        d = pytesseract.image_to_data(
            roi,
            lang=settings.lang,
            config=config,
            output_type=Output.DICT
        )
        words = []
        for i in range(len(d["text"])):
            wtxt = (d["text"][i] or "").strip()
            if not wtxt:
                continue
            words.append(wtxt)
            try:
                confs_all.append(int(float(d["conf"][i])))
            except Exception:
                pass

        # Joining words as a single line-ish region
        if words:
            texts.append(" ".join(words))

    full_text = "\n".join(texts).strip()

    alpha_ratio, mean_conf, char_count = _quality_metrics(full_text, confs_all)
    metrics = {
        "mode": "regions",
        "regions": len(boxes),
        "alpha_ratio": alpha_ratio,
        "mean_conf": mean_conf,
        "char_count": char_count,
    }
    return full_text, metrics


def extract_text_best(image_path: str, lang: str = "deu") -> str:
    """
    Runs OCR on multiple preprocess variants and picks the best by mean confidence.
    """
    config = _tess_config(psm=6)

    best_text = ""
    best_conf = -1.0

    for img in _preprocess_candidates(image_path):
        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)
        text = "\n".join([t for t in (data.get("text") or []) if t and t.strip()])
        conf = _mean_conf(data)

        # Prefer higher confidence; break ties with longer text
        if conf > best_conf or (abs(conf - best_conf) < 1e-6 and len(text) > len(best_text)):
            best_conf = conf
            best_text = pytesseract.image_to_string(img, lang=lang, config=config)

    return best_text.strip()
