import os
from dotenv import load_dotenv
import deepl

# Lade den API-Key aus .env
load_dotenv()
auth_key = os.getenv("DEEPL_API_KEY")

# DeepL-Client initialisieren (DeepLClient oder Translator)
try:
    client = deepl.DeepLClient(auth_key)
except AttributeError:
    client = deepl.Translator(auth_key)

# Glossar einmalig anlegen (nur ausführen, wenn es noch nicht existiert)
# Signature: create_glossary(glossary_name, source_lang, target_lang, entries)
entries = {
    "credo": "credo",
    # weitere Paare hier hinzufügen, z.B. "Haus": "house"
}
glossary = client.create_glossary(
    "PoetryGlossary",  # Name des Glossars
    "DE",              # Quellsprache
    "EN-GB",           # Zielsprache
    entries
)

# Glossary-ID ausgeben (kopiere sie in deine .env als DEEPL_GLOSSARY_ID)
print("Your glossary_id is:", glossary.glossary_id)
