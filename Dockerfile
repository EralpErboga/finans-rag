# 1. Hafif ve güncel Python tabanı
FROM python:3.11-slim

# 2. Çalışma dizini belirle
WORKDIR /app

# 3. Sistem bağımlılıklarını güncelle ve gereksizleri temizle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Proje dosyalarını kopyala
COPY . .

# 6. Streamlit için portu dışa aç
EXPOSE 8501

# 7. Ağ ve arayüz ayarlarıyla uygulamayı başlat
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]