import os
import pytest
#from ocr import extract_text_with_layout
from translate import cleanup_ocr_for_translation

# Ein kleines Testbild (du musst hier ein kurzes Beispielbild ablegen):
TEST_IMAGE = os.path.join("tests", "fixtures", "sample_poetry_01.png")

@pytest.fixture()
def poem_with_stanza_break():
    # Has a stanza break (blank line) that must be preserved
    return "erste zeile\nzweite zeile\n\nneuer absatz\nnoch eine zeile\n"


@pytest.fixture()
def poem_without_stanza_break():
    # No blank line; paragraphs are optional in OCR output
    return "erste zeile\nzweite zeile\ndritte zeile\n"


def test_text_not_empty(poem_with_stanza_break):
    cleaned = cleanup_ocr_for_translation(poem_with_stanza_break)
    assert cleaned.strip(), "Text cleanup returned empty text"


def test_line_breaks_preserved(poem_with_stanza_break):
    cleaned = cleanup_ocr_for_translation(poem_with_stanza_break)
    assert "\n" in cleaned, "No line breaks preserved"


def test_stanza_break_preserved_when_present(poem_with_stanza_break):
    cleaned = cleanup_ocr_for_translation(poem_with_stanza_break)
    assert "\n\n" in cleaned, "Stanza/paragraph break (blank line) was not preserved"


def test_no_excessive_blank_lines_created():
    raw = "zeile 1\n\n\n\n\nzeile 2\n"
    cleaned = cleanup_ocr_for_translation(raw)
    assert "\n\n\n" not in cleaned, "Cleanup should cap consecutive blank lines"


def test_hyphenation_fixed_across_linebreak():
    raw = "un-\nendlich\n"
    cleaned = cleanup_ocr_for_translation(raw)
    assert "unendlich" in cleaned, "Hyphenation across newline not fixed"