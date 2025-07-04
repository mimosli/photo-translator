import os
import pytest
from ocr import extract_text_with_layout

# Ein kleines Testbild (du musst hier ein kurzes Beispielbild ablegen):
TEST_IMAGE = os.path.join("tests", "fixtures", "sample_poetry_01.png")

@pytest.fixture(scope="module")
def sample_text():
    # Lies den Text ein, damit mehrere Tests ihn verwenden können
    return extract_text_with_layout(TEST_IMAGE)

def test_lines_not_empty(sample_text):
    # Stelle sicher, dass wir überhaupt Text bekommen
    assert sample_text.strip(), "OCR lieferte keinen Text"

def test_line_breaks_preserved(sample_text):
    # Mindestens ein Zeilenumbruch enthalten?
    assert "\n" in sample_text, "Keine Zeilenumbrüche erkannt"

def test_paragraph_preserved(sample_text):
    # Mindestens eine Leerzeile (Absatz-Trenner)?
    assert "\n\n" in sample_text, "Keine Absätze (doppelte Zeilenumbrüche) erkannt"
