# docker/ui/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Context 'root' olacağı için ui/requirements.txt yolunu veriyoruz
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm ui klasörünü içeri al
COPY ui /app

# Flask Sunucusunu Başlat
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]