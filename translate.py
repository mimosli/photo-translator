from __future__ import annotations

import os
import re
import inspect
import deepl
from dotenv import load_dotenv
from glossary import apply_glossary

load_dotenv()

# -----------------------------
# Environment / config
# -----------------------------

def _require_deepl_key():
    if not os.getenv("DEEPL_API_KEY"):
        raise RuntimeError("DEEPL_API_KEY is missing")

DEEPL_KEY = os.getenv("DEEPL_API_KEY")

# Optional glossaries (DeepL glossary IDs)
GLOSSARY_DE_EN = (os.getenv("DEEPL_GLOSSARY_ID_DE_EN") or "").strip()
GLOSSARY_FR_EN = (os.getenv("DEEPL_GLOSSARY_ID_FR_EN") or "").strip()
GLOSSARY_ES_EN = (os.getenv("DEEPL_GLOSSARY_ID_ES_EN") or "").strip()
GLOSSARY_PL_EN = (os.getenv("DEEPL_GLOSSARY_ID_PL_EN") or "").strip()

USE_GLOSSARY = any([
    GLOSSARY_DE_EN,
    GLOSSARY_FR_EN,
    GLOSSARY_ES_EN,
    GLOSSARY_PL_EN,
])

# -----------------------------
# OCR cleanup helpers
# -----------------------------

_WEIRD_CHAR_MAP = {
    "\u00ad": "",   # soft hyphen
    "\ufb01": "fi",
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

def _normalize_chars(s: str) -> str:
    for a, b in _WEIRD_CHAR_MAP.items():
        s = s.replace(a, b)
    return s

def _fix_hyphenation(s: str) -> str:
    # Join words split across line breaks: Wort-\nbruch → Wortbruch
    s = re.sub(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])", r"\1\2", s)
    return s

def _remove_line_artifacts(s: str) -> str:
    s = s.replace("|", " ")
    s = re.sub(r"[_]{3,}", " ", s)
    s = re.sub(r"[=]{3,}", " ", s)

    lines = s.splitlines()
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        if len(stripped) == 1 and stripped in {"-", "_", "—", "–", "."}:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)

def _normalize_whitespace_poetry(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    out = []
    blank_run = 0
    for ln in s.split("\n"):
        ln2 = re.sub(r"[ \t]{2,}", " ", ln).rstrip()

        if ln2.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue

        blank_run = 0
        out.append(ln2)

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out)

def cleanup_ocr_for_translation(text: str) -> str:
    if not text:
        return ""

    s = text
    s = _normalize_chars(s)
    s = _fix_hyphenation(s)
    s = _remove_line_artifacts(s)
    s = _normalize_whitespace_poetry(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

# -----------------------------
# Glossary helpers
# -----------------------------

def get_glossary_id_for_pair(src: str, tgt: str) -> str | None:
    if src == "DE" and tgt == "EN":
        return GLOSSARY_DE_EN or None
    if src == "FR" and tgt == "EN":
        return GLOSSARY_FR_EN or None
    if src == "ES" and tgt == "EN":
        return GLOSSARY_ES_EN or None
    if src == "PL" and tgt == "EN":
        return GLOSSARY_PL_EN or None
    return None

# -----------------------------
# DeepL translation
# -----------------------------

def translate_to_english(text: str) -> tuple[str, str | None]:
    """
    Translates OCR text to English (lowercase).
    Automatically detects source language via DeepL.
    Supports DE, FR, ES, PL (and more).
    """
    _require_deepl_key()

    cleaned = cleanup_ocr_for_translation(text)
    corrected = apply_glossary(cleaned)

    translator = deepl.Translator(DEEPL_KEY)

    params = {
        "text": corrected,
        "source_lang": "auto",   # 👈 DeepL auto-detection
        "target_lang": "EN-GB",
    }

    # Attach glossary if DeepL tells us the detected language
    sig = inspect.signature(translator.translate_text)
    if USE_GLOSSARY and "glossary" in sig.parameters:
        # We only know detected language AFTER translation,
        # but DeepL allows glossary with auto-detection.
        # So we apply glossary conditionally after detection.
        pass

    result = translator.translate_text(**params)

    detected = getattr(result, "detected_source_lang", None)

    # Apply glossary only when we have a matching pair
    if detected and USE_GLOSSARY:
        glossary_id = get_glossary_id_for_pair(detected, "EN")
        if glossary_id:
            params["glossary"] = glossary_id
            result = translator.translate_text(**params)
            detected = getattr(result, "detected_source_lang", detected)

    translated = result.text.lower()  # 👈 enforce lowercase

    return translated, detected
