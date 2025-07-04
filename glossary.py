import re

# Mapping für manuelle Korrekturen von OCR-Fehlern
GLOSSARY = {
    "crech": "credo",
    # Hier weitere Begriffe hinzufügen: "falsch": "richtig"
}


def apply_glossary(text: str) -> str:
    """
    Ersetzt falsch erkannte Wörter im OCR-Text anhand unserer GLOSSARY.
    Nur ganze Wörter werden angepasst.
    """
    # Baue Regex-Pattern für alle Keys in GLOSSARY
    pattern = r"\b(" + "|".join(map(re.escape, GLOSSARY.keys())) + r")\b"
    # Führe Ersetzungen durch (case-insensitive)
    return re.sub(pattern, lambda m: GLOSSARY[m.group(0).lower()], text, flags=re.IGNORECASE)
