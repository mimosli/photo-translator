# Use a slim Python image
FROM python:3.12-slim

# Install Tesseract OCR + German language support
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

 # install geolite2 database and reader
RUN pip install geoip2
# fetch the free MaxMind GeoLite2-City database
ADD https://geolite.maxmind.com/download/geoip/database/GeoLite2-City.tar.gz /tmp/
RUN cd /tmp && tar xzf GeoLite2-City.tar.gz \
    && mv GeoLite2-City_*/GeoLite2-City.mmdb /app/GeoLite2-City.mmdb \
    && rm -rf /tmp/*

# Set working dir
WORKDIR /app

# Copy and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . ./

# Expose port your Flask/Gunicorn binds to
EXPOSE 5001

# Run Gunicorn on container start
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5001", "app:app"]
