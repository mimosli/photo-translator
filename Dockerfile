# Use a slim Python image
FROM python:3.12-slim

# Install Tesseract OCR + German language support
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

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
