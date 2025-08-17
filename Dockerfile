# Use a slim Python image
FROM python:3.12-slim

# Install Tesseract OCR + German language support
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    tesseract-ocr \
    tesseract-ocr-deu \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

 
# ---- MaxMind GeoLite2 Country (requires license key) ----
ARG MAXMIND_LICENSE_KEY
RUN if [ -n "$MAXMIND_LICENSE_KEY" ]; then \
      echo "Downloading GeoLite2-Country with license key..." && \
      curl -fsSL "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz" \
        -o /tmp/GeoLite2-Country.tar.gz && \
      mkdir -p /tmp/geolite && \
      tar -xzf /tmp/GeoLite2-Country.tar.gz -C /tmp/geolite --strip-components=1 && \
      mv /tmp/geolite/GeoLite2-Country.mmdb /app/GeoLite2-Country.mmdb && \
      rm -rf /tmp/* ; \
    else \
      echo "WARNING: MAXMIND_LICENSE_KEY not set; skipping GeoLite2 DB download"; \
    fi

# Copy all code
COPY . ./

# Expose port your Flask/Gunicorn binds to
EXPOSE 5001

# Run Gunicorn on container start
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5001", "app:app"]
