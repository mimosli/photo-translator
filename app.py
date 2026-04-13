import os
import logging
import time
from uuid import uuid4
from io import BytesIO

from flask import Flask, request, jsonify, render_template, url_for, Response, g
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from PIL import Image, ImageOps

import psycopg
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from ocr import extract_text_best
from translate import translate_to_english


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    level=logging.INFO,
)
log = logging.getLogger("photo-translator")


# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MAX_BYTES = int(os.environ.get("MAX_BYTES", 3 * 1024 * 1024))      # 3 MB default
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", 2200))         # 2200 px default

# allow larger intake so we can recompress; cap to ~12 MB
app.config["MAX_CONTENT_LENGTH"] = max(MAX_BYTES * 6, 12 * 1024 * 1024)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
GEOIP_DB_PATH = os.environ.get("GEOIP_DB_PATH", "./GeoLite2-Country.mmdb")


# ──────────────────────────────────────────────────────────────────────────────
# GeoIP (optional)
# ──────────────────────────────────────────────────────────────────────────────
geoip_reader = None
try:
    from geoip2.database import Reader as GeoIP2Reader  # imported only if available
    if os.path.exists(GEOIP_DB_PATH):
        geoip_reader = GeoIP2Reader(GEOIP_DB_PATH)
    else:
        log.warning(f"GeoIP DB not found at {GEOIP_DB_PATH}; GeoIP disabled")
except Exception as e:
    log.warning(f"GeoIP disabled: {e}")


def get_country_from_ip(ip: str) -> str:
    if not geoip_reader:
        return "ZZ"
    try:
        return geoip_reader.country(ip).country.iso_code or "ZZ"
    except Exception:
        return "ZZ"


def get_client_ip() -> str:
    # Respect reverse proxy header if present
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip() or "0.0.0.0"
    return request.remote_addr or "0.0.0.0"


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────────────────────
TRANSLATE_COUNTER = Counter(
    "photo_translator_translate_requests_total",
    "Total number of /translate requests",
)
TRANSLATE_LATENCY = Histogram(
    "photo_translator_translate_request_seconds",
    "Histogram of /translate request latency",
)
UPLOAD_COUNTER = Counter(
    "photo_translator_upload_requests_total",
    "Total number of /api/upload requests",
)
COUNTRY_GAUGE = Gauge(
    "photo_translator_uploads_by_country",
    "Number of uploads per country (counter-like gauge)",
    ["country"],
)


_TRANSLATE_ENDPOINTS = {"translate_image", "api_translate"}


@app.before_request
def before_request():
    # time only translate endpoints
    if request.endpoint in _TRANSLATE_ENDPOINTS:
        g._t_start = time.perf_counter()


@app.after_request
def after_request(response):
    # metrics best-effort; never break response
    try:
        if request.endpoint in _TRANSLATE_ENDPOINTS:
            TRANSLATE_COUNTER.inc()
            t0 = getattr(g, "_t_start", None)
            if t0 is not None:
                TRANSLATE_LATENCY.observe(time.perf_counter() - t0)

        if request.endpoint == "api_upload":
            UPLOAD_COUNTER.inc()

        if request.endpoint in _TRANSLATE_ENDPOINTS | {"api_upload"}:
            ip = get_client_ip()
            country = get_country_from_ip(ip)
            COUNTRY_GAUGE.labels(country=country).inc()
    except Exception as e:
        log.warning(f"metrics_after_request_failed: {e}")
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return jsonify({"error": "File too large", "max": MAX_BYTES}), 413


# ──────────────────────────────────────────────────────────────────────────────
# DB helper (best-effort)
# ──────────────────────────────────────────────────────────────────────────────
def db_insert_upload(filename: str, client_ip: str, user_agent: str, country: str) -> None:
    """
    Best-effort insert. If DB unavailable or table missing, log and continue.
    Uses short-lived connections to avoid stale global connections in prod.
    """
    if not DATABASE_URL:
        return
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO uploads (filename, client_ip, user_agent, country)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (filename, client_ip, user_agent, country),
                )
    except Exception as e:
        log.warning(f"upload_db_insert_failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────
def _compress_to_jpeg_under_limit(img: Image.Image, max_bytes: int) -> bytes:
    """
    Compresses PIL image to JPEG under max_bytes by reducing quality.
    """
    quality = 88
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    while buf.tell() > max_bytes and quality > 70:
        quality -= 5
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _save_normalized_image(file_storage, out_path: str) -> int:
    """
    Load via PIL, auto-rotate (EXIF), convert to RGB, resize, and save as JPEG.
    Returns bytes written.
    """
    try:
        img = Image.open(file_storage.stream)
    except Exception:
        raise ValueError("Invalid image")

    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = img.convert("RGB")

    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

    jpeg_bytes = _compress_to_jpeg_under_limit(img, MAX_BYTES)
    if len(jpeg_bytes) > MAX_BYTES:
        raise ValueError("Image too large after compression")

    with open(out_path, "wb") as f:
        f.write(jpeg_bytes)

    return len(jpeg_bytes)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def upload_form():
    host = request.host.split(":")[0]
    if host in ("leilafrey.com", "www.leilafrey.com"):
        return render_template("welcome.html")
    return render_template("upload.html")


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({"maxBytes": MAX_BYTES, "maxDimension": MAX_DIMENSION}), 200


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(f.stream)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    # Auto-rotate and convert to RGB
    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = img.convert("RGB")

    # Fit to bounding box (no enlargement)
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

    jpeg_bytes = _compress_to_jpeg_under_limit(img, MAX_BYTES)
    if len(jpeg_bytes) > MAX_BYTES:
        return jsonify({"error": "Image too large after compression", "size": len(jpeg_bytes), "max": MAX_BYTES}), 413

    # Save with unique name
    orig = secure_filename(f.filename or "upload.jpg")
    base, _ = os.path.splitext(orig)
    unique = f"{int(time.time())}_{uuid4().hex[:8]}"
    safe_name = f"{base}_{unique}.jpg"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    with open(out_path, "wb") as out:
        out.write(jpeg_bytes)

    url = url_for("static", filename=f"uploads/{safe_name}", _external=False)

    ip = get_client_ip()
    country = get_country_from_ip(ip)
    ua = request.headers.get("User-Agent", "")

    log.info(f'upload_store: filename="{safe_name}" bytes={len(jpeg_bytes)} ip={ip} country={country}')
    db_insert_upload(filename=safe_name, client_ip=ip, user_agent=ua, country=country)

    return jsonify({"ok": True, "url": url, "bytes": len(jpeg_bytes)}), 200


def _run_translate_pipeline(f):
    """
    Shared logic for translate endpoints.
    Returns a dict with keys: original_image_url, extracted_text, translated_text,
    ocr_error, translation_error. Raises ValueError on bad input.
    """
    ip = get_client_ip()
    country = get_country_from_ip(ip)
    ua = request.headers.get("User-Agent", "")

    # Save input file
    original_name = secure_filename(f.filename or "photo.jpg")
    base, _ = os.path.splitext(original_name)
    filename = f"{base}_{int(time.time())}_{uuid4().hex[:6]}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    _save_normalized_image(f, filepath)

    # OCR
    extracted_text = ""
    ocr_error = None
    try:
        extracted_text = (extract_text_best(filepath) or "").strip()
        if not extracted_text:
            ocr_error = "No readable text detected. Try brighter light, less tilt, and fill the frame with the page."
    except Exception as e:
        log.exception("ocr_failed")
        ocr_error = str(e)[:200]

    # Translation
    translated_text = ""
    detected_lang = None
    translation_error = None
    try:
        if extracted_text:
            translated_text, detected_lang = translate_to_english(extracted_text)
            translated_text = (translated_text or "").strip()
    except Exception as e:
        log.exception("translation_failed")
        translation_error = str(e)[:200]

    log.info(
        f"upload_metadata: {{'client_ip': '{ip}', 'filename': '{filename}', 'user_agent': '{ua}', 'country': '{country}', 'detected_lang': '{detected_lang}'}}"
    )
    db_insert_upload(filename=filename, client_ip=ip, user_agent=ua, country=country)

    return {
        "original_image_url": url_for("static", filename=f"uploads/{filename}"),
        "extracted_text": extracted_text,
        "translated_text": translated_text,
        "ocr_error": ocr_error,
        "translation_error": translation_error,
    }


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """JSON endpoint used by the frontend JS."""
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        result = _run_translate_pipeline(f)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200


@app.route("/translate", methods=["POST"])
def translate_image():
    """Legacy HTML endpoint (kept for direct form submissions)."""
    if "image" not in request.files:
        return "No file part", 400
    f = request.files["image"]
    if not f or f.filename == "":
        return "No selected file", 400

    try:
        ctx = _run_translate_pipeline(f)
    except ValueError as e:
        return str(e), 400

    return render_template(
        "result.html",
        original_image=ctx["original_image_url"],
        extracted_text=ctx["extracted_text"],
        translated_text=ctx["translated_text"],
        ocr_error=ctx["ocr_error"],
        translation_error=ctx["translation_error"],
    )


@app.route("/result")
def result_page():
    """Serves the result.html shell; JS reads data from sessionStorage."""
    return render_template("result.html")


@app.route("/upload")
def upload_page():
    """Alias for the root upload form."""
    return render_template("upload.html")


@app.route("/de")
def landing_de():
    return render_template("landing_de.html")


@app.route("/en")
def landing_en():
    return render_template("landing_en.html")


@app.route("/fr")
def landing_fr():
    return render_template("landing_fr.html")


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", version="1.1"), 200


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # In prod you run gunicorn, so debug=False here
    app.run(host="0.0.0.0", port=5002, debug=False)
