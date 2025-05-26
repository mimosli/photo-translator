import os
import openai
import deepl
from dotenv import load_dotenv

load_dotenv()

def translate_with_deepl(text):
    translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
    return translator.translate_text(text, source_lang="DE", target_lang="EN").text

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
