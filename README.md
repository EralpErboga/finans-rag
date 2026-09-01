# ⚡ Finans-RAG: Kurum İçi EPDK Mevzuatı & Mali Tablo Asistanı

Bu proje, elektrik dağıtım sektörü finans ve regülasyon ekipleri için geliştirilmiş; EPDK mevzuat dokümanları ve Excel mali tabloları (Mizan, Bilanço, Gelir Tablosu) üzerinden hibrit analiz yapabilen, bağlam ve bellek yönetimli **%100 yerel** bir RAG (Retrieval-Augmented Generation) asistanıdır.

---

## 🎯 Temel Yetenekler
* **Veri Güvenliği:** Tüm süreç yerel donanımda (On-Premise) çalışır; kurum içi veriler dış servislerle paylaşılmaz.
* **Akıllı Yönlendirici (Router):** Türkçe karakter ve kök normalizasyonu ile sorguları otomatik olarak Mevzuat, Finans, Hibrit veya Sohbet Belleği modüllerine yönlendirir.
* **Hibrit Çapraz Denetim:** Mevzuat kuralları (DVT kriterleri, Kayıp-Kaçak tavan indirimi) ile şirket bilançosunu eşleştirip etki analizi üretir.
* **Dinamik Bellek & İndeksleme:** Kısa takip sorularını ("şartları", "b ve c") önceki bağlama oturtur; sıra ve indeks bazlı geçmiş sorgularına ("ondan 2 sonrakinde ne sordum") yanıt verir.

---

## 🛠️ Kullanılan Teknolojiler
* **Arayüz:** Streamlit
* **Yerel LLM:** Qwen 2.5 (7B) via Ollama
* **Yerel Embedding:** BAAI/bge-m3 via Ollama
* **Veri Motoru:** Pandas, OpenPyXL
* **Vektör Deposu:** Cosine Similarity Vektör Arama (JSON Tabanlı)

---

## 🚀 Kurulum ve Çalıştırma

1. **Bağımlılıkları Kurun:**

pip install -r requirements.txt

2. **Ollama Modellerini İndirin:**

ollama pull qwen2.5:7b
ollama pull bge-m3

3. **Mevzuat Dokümanlarını Vektörleştirin:**

python ingest.py

4. **Arayüzü Başlatın:**

streamlit run app.py

**Uygulamayı konteyner ortamında ayağa kaldırmak için:**

docker-compose up --build