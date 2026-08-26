# ⚡ Finans-RAG: Kurum İçi EPDK Mevzuatı & Mizan Asistanı

Bu proje, elektrik dağıtım sektörü finans ve regülasyon ekipleri için geliştirilmiş; EPDK mevzuat dokümanları ve Excel mizan tabloları üzerinden soru-cevap yapabilen **%100 yerel (on-premise)** bir RAG (Retrieval-Augmented Generation) asistanıdır.

---

## 🎯 Projenin Amacı ve Problem Tanımı
* **Veri Güvenliği:** Finansal tablolar ve kurum içi regülasyon verileri harici bulut servislerine gönderilmez; tüm süreç yerel donanımda çalışır.
* **Sıfır Halüsinasyon (Zero-Hallucination):** 
  * Mevzuat sorgularında anlamsal arama ile doğrudan ilgili yönetmelik maddeleri referans gösterilir.
  * Mali tablolarda (Mizan, Bilanço, Gelir Tablosu) matematiksel işlemler LLM'e bırakılmaz; bellek içi SQLite ve Pandas motoru üzerinden kesin olarak hesaplanır.

---

## 🏗️ Sistem Mimarisi

```
[ Kullanıcı Sorusu ]
         │
         ▼
[ LLM Sınıflandırıcı (Router) ]
   ├── (Mevzuat Sorusu) ──► [ Ollama BGE-M3 Vektör Arama ] ──► [ İlgili Madde Alıntısı ] ──► [ Qwen 2.5 LLM ]
   └── (Finansal / Mizan)  ──► [ SQLite & Pandas Motoru ]   ──► [ Kesin Matematiksel Veri ] ──► [ Qwen 2.5 LLM ]
                                                                                                    │
                                                                                                    ▼
                                                                                           [ Streamlit Arayüzü ]
```

---

## 🛠️ Kullanılan Teknolojiler

* **Arayüz:** Streamlit
* **Yerel LLM:** Qwen 2.5 (7B) via Ollama
* **Yerel Embedding:** BAAI/bge-m3 via Ollama
* **Veri Motoru:** Pandas, SQLite, OpenPyXL
* **Vektör Deposu:** Cosine Similarity Vektör Arama

---

## 🚀 Kurulum ve Çalıştırma

1. **Bağımlılıkları Kurun:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Modellerini İndirin:**
   ```bash
   ollama pull qwen2.5:7b
   ollama pull bge-m3
   ```

3. **Mevzuat Dokümanlarını Vektörleştirin:**
   ```bash
   python ingest.py
   ```

4. **Arayüzü Başlatın:**
   ```bash
   streamlit run app.py
   ```