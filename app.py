print("hello world")

import os
import logging
import time
from uuid import uuid4

from flask        import Flask, request, jsonify, render_template, url_for, Response, g
from ocr          import extract_text
from translate    import translate_with_deepl   # or translate_with_gpt
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from geoip2.database import Reader as GeoIP2Reader
import psycopg
from PIL import Image, ImageOps
from io import BytesIO
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge


# ─── basic JSON logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    level=logging.INFO
)

# ─── flask app + upload folder ─────────────────────────────────────────────────
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── upload limits (env-overridable) ───────────────────────────────────────────────────────────────
# e.g. export MAX_BYTES=3145728 and MAX_DIMENSION=2200 in docker-compose
MAX_BYTES = int(os.environ.get("MAX_BYTES", 3 * 1024 * 1024))   # default 3 MB
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", 2200))      # default 2200 px box

# allow larger intake so we can recompress; cap to ~12 MB
app.config["MAX_CONTENT_LENGTH"] = max(MAX_BYTES * 6, 12 * 1024 * 1024)

# ─── postgres DB ───────────────────────────────────────────────────────────────
# expects DATABASE_URL in your .env, e.g.:
#   export DATABASE_URL="postgresql://photo:secretpassword@db:5432/photodb"
conn = None
def get_db_conn():
    global conn
    if conn is not None:
        try:
            if not conn.closed:
                return conn
        except Exception:
            conn = None
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logging.warning("DATABASE_URL not set; DB disabled")
        return None
    for attempt in range(1, 8):  # retry ~7s total
        try:
            c = psycopg.connect(dsn)
            c.autocommit = True
            conn = c
            return conn
        except Exception as e:
            logging.warning(f"DB connect failed (attempt {attempt}): {e}")
            time.sleep(1)
    logging.error("DB unavailable; proceeding without DB")
    return None


# ─── GeoIP ─────────────────────────────────────────────────────────────────────
GEOIP_DB_PATH = os.environ.get("GEOIP_DB_PATH", "./GeoLite2-Country.mmdb")
geoip_reader = None
try:
    from geoip2.database import Reader as GeoIP2Reader
    geoip_reader = GeoIP2Reader(GEOIP_DB_PATH)
except Exception as e:
    logging.warning(f"GeoIP disabled (path={GEOIP_DB_PATH}): {e}")

def get_country_from_ip(ip: str) -> str:
    if not geoip_reader:
        return "ZZ"
    try:
        return geoip_reader.country(ip).country.iso_code or "ZZ"
    except Exception:
        return "ZZ"


# ─── Prometheus metrics ────────────────────────────────────────────────────────
TRANSLATE_COUNTER = Counter(
    'photo_translator_translate_requests_total',
    'Total number of /translate requests'
)
TRANSLATE_LATENCY = Histogram(
    'photo_translator_translate_request_seconds',
    'Histogram of /translate request latency'
)
COUNTRY_GAUGE = Gauge(
    'photo_translator_uploads_by_country',
    'Number of uploads per country',
    ['country']
)


@app.before_request
def _before_request():
    # Only time the translate endpoint
    if request.endpoint == "translate_image":
        g._t_start = time.perf_counter()

@app.after_request
def _after_request(response):
    try:
        if request.endpoint == "translate_image":
            # 1) Prometheus counter
            TRANSLATE_COUNTER.inc()

            # 2) Prometheus latency
            t0 = getattr(g, "_t_start", None)
            if t0 is not None:
                TRANSLATE_LATENCY.observe(time.perf_counter() - t0)

            # 3) GeoIP gauge (safe)
            ip = request.remote_addr or "0.0.0.0"
            try:
                if "get_country_from_ip" in globals():
                    country = get_country_from_ip(ip)
                else:
                    country = geoip_reader.country(ip).country.iso_code if geoip_reader else "ZZ"
            except Exception:
                country = "ZZ"
            COUNTRY_GAUGE.labels(country=country or "ZZ").inc()
    except Exception as e:
        logging.warning(f"metrics_after_request_failed: {e}")
    return response


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({"error": "File too large", "max": MAX_BYTES}), 413

def get_country_from_ip(ip: str) -> str:
    try:
        return geoip_reader.country(ip).country.iso_code or "Hint: GeoAnalytics" if geoip_reader else "Check: GeoAnalytics"
    except Exception:
        return "ZZ"

# ─── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def upload_form():
    #return render_template("upload.html")
    return render_template("welcome.html")

@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "maxBytes": MAX_BYTES,
        "maxDimension": MAX_DIMENSION
    }), 200

@app.route("/api/upload", methods=["POST"])
def api_upload():
    # validate
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # load & auto-rotate based on EXIF
    try:
        img = Image.open(f.stream)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to RGB (JPG target), autorotate using EXIF
    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = img.convert("RGB")

    # Fit to bounding box (preserves aspect), no enlargement
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

    # Iteratively compress to target MAX_BYTES
    quality = 88
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    while buf.tell() > MAX_BYTES and quality > 60:
        quality -= 6
        buf.seek(0); buf.truncate(0)
        img.save(buf, format="JPEG", quality=quality, optimize=True)

    if buf.tell() > MAX_BYTES:
        return jsonify({
            "error": "Image too large after compression",
            "size": buf.tell(),
            "max": MAX_BYTES
        }), 413

    # Persist with safe unique name
    orig = secure_filename(f.filename or "upload.jpg")
    base, _ = os.path.splitext(orig)
    unique = f"{int(time.time())}_{uuid4().hex[:8]}"
    safe_name = f"{base}_{unique}.jpg"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    with open(out_path, "wb") as out:
        out.write(buf.getvalue())

    # Build relative URL (served from /static/uploads)
    url = url_for("static", filename=f"uploads/{safe_name}", _external=False)

    # Optional: minimal metadata log (keeps your existing pattern)
    logging.info(f'upload_store: filename="{safe_name}" bytes={buf.tell()} ip={request.remote_addr}')

    # Optional: DB insert for the upload (mirrors your /translate logging)
    try:
        with conn.cursor() as cur:
            country = "ZZ"
            try:
                ip = request.remote_addr or "0.0.0.0"
                country = geoip_reader.country(ip).country.iso_code or "ZZ"
            except Exception:
                pass
            cur.execute(
                """
                INSERT INTO uploads (filename, client_ip, user_agent, country)
                VALUES (%s, %s, %s, %s)
                """,
                ( safe_name, request.remote_addr, request.headers.get("User-Agent"), country )
            )
    except Exception as e:
        logging.warning(f"upload_db_insert_failed: {e}")

    return jsonify({
        "ok": True,
        "url": url,
        "bytes": buf.tell()
    }), 200


@app.route("/translate", methods=["POST"])
def translate_image():
    try:
        # 1) Upload validieren
        if "image" not in request.files:
            return "No file part", 400
        f = request.files["image"]
        if not f or f.filename == "":
            return "No selected file", 400

        # 2) Sicherer, eindeutiger Dateiname
        original_name = secure_filename(f.filename or "photo.jpg")
        if "." not in original_name:
            original_name += ".jpg"
        base, ext = os.path.splitext(original_name)
        filename = f"{base}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        # 3) OCR
        german_text = ""
        try:
            german_text = (extract_text(filepath) or "").strip()
        except Exception as e:
            logging.exception("ocr_failed")

        # 4) Übersetzung
        translated_text = ""
        translation_error = None
        try:
            if german_text:
                translated_text = (translate_with_deepl(german_text) or "").strip()
        except Exception as e:
            logging.exception("translation_failed")
            translation_error = str(e)[:200]  # kurze Fehlermeldung für UI

        # 5) Metadaten & DB (best effort)
        metadata = {
            "client_ip":  request.remote_addr,
            "filename":   filename,
            "user_agent": request.headers.get("User-Agent"),
            "country":    get_country_from_ip(request.remote_addr or "0.0.0.0"),
        }
        logging.info(f"upload_metadata: {metadata}")

        db = get_db_conn() if 'get_db_conn' in globals() else None
        if db:
            try:
                with db.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO uploads (filename, client_ip, user_agent, country)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (metadata["filename"], metadata["client_ip"], metadata["user_agent"], metadata["country"])
                    )
            except Exception as e:
                logging.warning(f"upload_db_insert_failed: {e}")

        # 6) Ergebnis HTML rendern (zeigt ggf. translation_error an)
        return render_template(
            "result.html",
            original_image  = url_for("static", filename=f"uploads/{filename}"),
            extracted_text  = german_text,
            translated_text = translated_text,
            translation_error = translation_error
        )
    except Exception as e:
        logging.exception("translate_route_failed")
        # Fallback: sehr einfache Fehlermeldung, aber kein 500 für den User-Flow
        return render_template(
            "result.html",
            original_image=None,
            extracted_text="",
            translated_text="",
            translation_error="Unerwarteter Serverfehler. Bitte später erneut versuchen."
        ), 200


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", version="1.0"), 200


@app.route("/metrics")
def metrics():
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


# ─── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
