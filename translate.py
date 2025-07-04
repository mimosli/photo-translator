import os
import deepl
from dotenv import load_dotenv
from glossary import apply_glossary

# Lade Umgebungsdaten aus .env
load_dotenv()
DEEPL_KEY   = os.getenv("DEEPL_API_KEY")
GLOSSARY_ID = os.getenv("DEEPL_GLOSSARY_ID")
USE_GLOSSARY = bool(GLOSSARY_ID)


def translate_with_deepl(text: str) -> str:
    """
    Übersetzt deutschen Text ins Englische (UK) via DeepL.
    - Zuerst werden OCR-Fehler per apply_glossary korrigiert.
    - Dann erfolgt der API-Aufruf, ggf. mit Glossary.
    """
    # 1) OCR-Ergebnis bereinigen
    corrected = apply_glossary(text)

    # 2) Deepl-Client initialisieren
    translator = deepl.Translator(DEEPL_KEY)

    # 3) Parameter für Übersetzung
    params = {
        "text": corrected,
        "source_lang": "DE",
        "target_lang": "EN-GB"
    }
    if USE_GLOSSARY:
        params["glossary"] = GLOSSARY_ID

    # 4) API-Aufruf
    result = translator.translate_text(**params)
    return result.text

# Optional: GPT-basierte Übersetzung (wenn benötigt)
# import openai
# def translate_with_gpt(text: str) -> str:
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful translation assistant."},
#             {"role": "user", "content": f"Übersetze ins Englische: {corrected}"}
#         ]
#     )
#     return response.choices[0].message.content.strip()
