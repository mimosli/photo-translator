import os
import openai
import deepl
from dotenv import load_dotenv

import ocr
from ocr import sample_text

load_dotenv()

def translate_with_deepl(text):
    translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
    return translator.translate_text(text, source_lang="DE", target_lang="EN-GB").text

def translate_with_gpt(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful translation assistant."},
            {"role": "user", "content": f"Übersetze folgenden Text ins Englische: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

sample_translated_dp = translate_with_deepl(sample_text)
#sample_translated_oai = translate_with_gpt(sample_text)
print(sample_translated_dp)
#print(sample_translated_oai)