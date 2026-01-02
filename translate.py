import os
import re
import deepl
import inspect
from dotenv import load_dotenv
from glossary import apply_glossary
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # make results deterministic

load_dotenv()
DEEPL_KEY = (os.getenv("DEEPL_API_KEY") or "").strip()
if not DEEPL_KEY:
    raise RuntimeError("DEEPL_API_KEY is missing")

GLOSSARY_DE_EN = (os.getenv("DEEPL_GLOSSARY_ID_DE_EN") or "").strip()
GLOSSARY_FR_EN = (os.getenv("DEEPL_GLOSSARY_ID_FR_EN") or "").strip()

GLOSSARY_ID  = (os.getenv("DEEPL_GLOSSARY_ID") or "").strip()
USE_GLOSSARY = bool(GLOSSARY_ID)

# -----------------------------
# OCR text cleanup helpers
# -----------------------------

_WEIRD_CHAR_MAP = {
    "\u00ad": "",   # soft hyphen
    "\ufb01": "fi", # ligatures
    "\ufb02": "fl",
    "“": '"', "”": '"',
    "„": '"', "«": '"', "»": '"',
    "’": "'", "‘": "'",
    "—": "-", "–": "-",
    "•": "-", "·": "-",
    "…": "...",
    "∕": "/",
    "¦": "|",
}


def detect_source_lang(text: str) -> str | None:
    """
    Returns DeepL-style language codes: 'DE' or 'FR', or None if uncertain.
    """
    try:
        lang = detect(text)
    except Exception:
        return None

    if lang == "de":
        return "DE"
    if lang == "fr":
        return "FR"
    return None

def get_glossary_id_for_pair(src: str, tgt: str) -> str | None:
    if src == "DE" and tgt == "EN":
        return GLOSSARY_DE_EN or None
    if src == "FR" and tgt == "EN":
        return GLOSSARY_FR_EN or None
    return None

def _normalize_chars(s: str) -> str:
    for a, b in _WEIRD_CHAR_MAP.items():
        s = s.replace(a, b)
    return s


def _fix_hyphenation(s: str) -> str:
    """
    Fix word breaks across line breaks:
      'Wort-\\nbruch' -> 'Wortbruch'
    Keep real dash lines intact by only removing hyphen right before a newline
    when both sides look like letters.
    """
    # letters (incl. umlauts) on both sides
    s = re.sub(r"([A-Za-zÄÖÜäöüß])-\n([A-Za-zÄÖÜäöüß])", r"\1\2", s)
    return s


def _remove_line_artifacts(s: str) -> str:
    """
    Remove common OCR junk while being conservative (poetry needs punctuation).
    """
    # Replace common "pipe" artifacts used as separators
    s = s.replace("|", " ")

    # Remove repeated underscores / long runs of punctuation
    s = re.sub(r"[_]{3,}", " ", s)
    s = re.sub(r"[=]{3,}", " ", s)

    # Remove lonely single-character lines that are often noise
    lines = s.splitlines()
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        if len(stripped) == 1 and stripped in {"-", "_", "—", "–", "."}:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def _normalize_whitespace_poetry(s: str) -> str:
    """
    Keep line breaks (important for poems), but normalize:
    - trailing spaces
    - excessive blank lines
    - multiple spaces inside a line
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    out_lines = []
    blank_run = 0
    for ln in s.split("\n"):
        # collapse multiple spaces/tabs inside a line
        ln2 = re.sub(r"[ \t]{2,}", " ", ln).rstrip()

        if ln2.strip() == "":
            blank_run += 1
            # allow at most 1 consecutive blank line (stanza spacing)
            if blank_run <= 1:
                out_lines.append("")
            continue

        blank_run = 0
        out_lines.append(ln2)

    # Trim leading/trailing blank lines
    while out_lines and out_lines[0] == "":
        out_lines.pop(0)
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "\n".join(out_lines)


def cleanup_ocr_for_translation(text: str) -> str:
    """
    The main cleanup function:
    1) normalize odd chars
    2) fix hyphenation across line breaks
    3) remove artifacts
    4) normalize whitespace while keeping poem structure
    """
    if not text:
        return ""

    s = text
    s = _normalize_chars(s)
    s = _fix_hyphenation(s)
    s = _remove_line_artifacts(s)
    s = _normalize_whitespace_poetry(s)

    # If OCR accidentally returns many empty lines, keep it sane:
    # (already handled via blank_run, but this is a final guard)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def ocr_quality_hint(text: str) -> str | None:
    """
    Optional: detect very noisy OCR and return a hint.
    You can show this in UI or logs.
    """
    if not text or len(text.strip()) < 10:
        return "OCR found very little text. Try brighter light and fill the frame with the poem."

    cleaned = "".join(ch for ch in text if not ch.isspace())
    if not cleaned:
        return "OCR found no readable characters."

    letters = sum(ch.isalpha() for ch in cleaned)
    alpha_ratio = letters / max(1, len(cleaned))

    # Heuristic: too many symbols -> likely garbage
    if alpha_ratio < 0.45:
        return "OCR quality seems low (many symbols). Try a sharper photo with less background."

    return None


# -----------------------------
# DeepL translation
# -----------------------------

def translate_to_english(text: str) -> tuple[str, str | None]:
    cleaned = cleanup_ocr_for_translation(text)
    corrected = apply_glossary(cleaned)

    source_lang = detect_source_lang(corrected)  # 'DE', 'FR', or None
    translator = deepl.Translator(DEEPL_KEY)

    params = {
        "text": corrected,
        "target_lang": "EN-GB",
    }
    if source_lang:
        params["source_lang"] = source_lang

    # Apply DeepL glossary only if:
    # - we detected a supported language
    # - we have the right glossary for that pair
    if source_lang and USE_GLOSSARY:
        sig = inspect.signature(translator.translate_text)
        if "glossary" in sig.parameters:
            glossary_id = get_glossary_id_for_pair(source_lang, "EN")  # you implement this
            if glossary_id:
                params["glossary"] = glossary_id

    result = translator.translate_text(**params)
    translated = result.text

    # If you omitted source_lang, DeepL may still tell you what it detected:
    detected = getattr(result, "detected_source_lang", None)
    return translated, (source_lang or detected)
