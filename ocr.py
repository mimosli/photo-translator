from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

from openai import OpenAI


# ---------------------------------
# Configuration
# ---------------------------------

@dataclass
class OcrSettings:
    """
    Settings kept mainly for compatibility & quality control.
    Preprocessing is now handled by the LLM internally.
    """
    min_lines: int = 1
    min_chars: int = 20
    min_alpha_ratio: float = 0.5   # letters / all non-space chars


DEFAULT = OcrSettings()


# ---------------------------------
# OpenAI client
# ---------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------
# Utilities
# ---------------------------------

def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _alpha_ratio(text: str) -> float:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if not cleaned:
        return 0.0
    letters = sum(ch.isalpha() for ch in cleaned)
    return letters / max(1, len(cleaned))


# ---------------------------------
# Core OCR (LLM-powered)
# ---------------------------------

def _ocr_with_openai(image_path: str) -> Dict:
    """
    Calls OpenAI Vision and returns parsed JSON.
    """

    image_b64 = _image_to_base64(image_path)

    prompt = """
You are a professional OCR engine.

TASK:
- Extract ALL visible text from the image.
- Detect text STRICTLY line by line.
- Preserve natural reading order (top to bottom).
- Do NOT merge lines.
- Do NOT infer or guess missing text.
- Ignore decorative graphics.

OUTPUT:
Return ONLY valid JSON in this exact format:

{
  "lines": [
    "first detected line",
    "second detected line",
    "third detected line"
  ]
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ],
            }
        ],
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(
            f"OpenAI OCR returned invalid JSON:\n{content}"
        ) from e


# ---------------------------------
# Public API (replacement for old)
# ---------------------------------

def extract_text(image_path: str, settings: OcrSettings = DEFAULT) -> str:
    """
    Simple OCR: returns text with newline-separated lines.
    """
    data = _ocr_with_openai(image_path)
    lines = [l.strip() for l in data.get("lines", []) if l.strip()]
    return "\n".join(lines)


def extract_text_with_layout(image_path: str, settings: OcrSettings = DEFAULT) -> str:
    """
    Kept for API compatibility.
    Now identical to extract_text because layout is natively preserved.
    """
    return extract_text(image_path, settings=settings)


def extract_text_regions(image_path: str, settings: OcrSettings = DEFAULT) -> Tuple[str, Dict]:
    """
    Region-based OCR is no longer needed.
    We simulate the old return structure for compatibility.
    """
    text = extract_text(image_path, settings=settings)

    alpha = _alpha_ratio(text)
    metrics = {
        "mode": "openai_vision",
        "lines": text.count("\n") + 1 if text else 0,
        "alpha_ratio": alpha,
        "char_count": len(text),
    }

    return text, metrics


def extract_text_best(image_path: str) -> str:
    """
    Best-quality mode.
    With LLM OCR, there is no need for multiple candidates.
    """
    return extract_text(image_path)
