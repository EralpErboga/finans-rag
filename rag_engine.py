import os
import json
import math
import sqlite3
import pandas as pd
import ollama

VECTOR_DB_PATH = "db/mevzuat_vektorleri.json"
EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"


# 1. Benzerlik Fonksiyonu
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0


# 2. Mevzuat RAG Hattı
def query_mevzuat(soru: str):
    if not os.path.exists(VECTOR_DB_PATH):
        return "Vektör veritabanı bulunamadı. Lütfen önce ingest.py çalıştırın.", "Hata"

    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    soru_vektoru = ollama.embeddings(model="bge-m3", prompt=soru)["embedding"]

    skorlar = []
    for item in db:
        skor = cosine_similarity(soru_vektoru, item["embedding"])
        skorlar.append((skor, item))

    skorlar.sort(key=lambda x: x[0], reverse=True)
    en_iyi = skorlar[:2]
    baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for _, item in en_iyi])
    kaynaklar = ", ".join(list(set([item["source"] for _, item in en_iyi])))

    prompt = f"""Sen şirket finans ve mevzuat ekibine yardımcı olan uzman bir asistansın.
Sana verilen mevzuat bağlamını kullanarak soruyu doğrudan Türkçe olarak cevapla.
Metinde olmayan bilgileri ekleme.
Cevabının altına mutlaka referans aldığın Belge adını ve Madde numarasını ekle.

Bağlam:
{baglam}

Soru: {soru}
Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    return cevap["message"]["content"], f"EPDK Mevzuatı ({kaynaklar})"


# 3. SQLite Veritabanı
def get_db_connection():
    conn = sqlite3.connect(":memory:")
    xls = pd.ExcelFile(EXCEL_PATH)
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4).dropna(subset=['Hesap Kodu'])
    df_mizan['Hesap Kodu'] = df_mizan['Hesap Kodu'].astype(int)

    # Kolon isimlerini küçük harf ve alt tireli yap
    df_mizan.columns = ['hesap_kodu', 'hesap_adi', 'hesap_grubu', 'borc', 'alacak', 'borc_bakiye', 'alacak_bakiye']
    df_mizan.to_sql("mizan", conn, index=False, if_exists="replace")
    return conn


# 4. Mali Tablo Hattı
def query_financial(soru: str):
    conn = get_db_connection()

    schema_info = """Tablo: mizan
Sütunlar:
- hesap_kodu (INT)
- hesap_adi (TEXT)
- borc (FLOAT)
- alacak (FLOAT)
- borc_bakiye (FLOAT)
- alacak_bakiye (FLOAT)"""

    sql_prompt = f"""Sen bir SQLite uzmanısın. Kullanıcı sorusunu yanıtlayacak tek satırlık geçerli SQL sorgusu yaz.

{schema_info}

KURALLAR:
- Kasa veya bankalardaki nakit varlık sorulduğunda: SELECT hesap_kodu, hesap_adi, borc_bakiye FROM mizan WHERE hesap_kodu IN (100, 102)
- Genel yönetim gideri sorulduğunda: SELECT hesap_kodu, hesap_adi, borc_bakiye FROM mizan WHERE hesap_kodu = 632

Soru: {soru}
Sadece SELECT ile başlayan SQL kodunu yaz:"""

    res_sql = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": sql_prompt}]
    )["message"]["content"].strip().replace("```sql", "").replace("```", "").strip()

    try:
        df_res = pd.read_sql(res_sql, conn)
    except Exception:
        df_res = pd.read_sql(
            "SELECT hesap_kodu, hesap_adi, borc_bakiye, alacak_bakiye FROM mizan WHERE hesap_kodu IN (100, 102)", conn)

    # Toplam tutarı doğrudan Python hesaplar
    sayisal_sutunlar = [c for c in df_res.columns if 'bakiye' in c or 'borc' in c or 'alacak' in c]
    toplam_tutar = df_res[sayisal_sutunlar[0]].sum() if sayisal_sutunlar else 0.0

    final_prompt = f"""Aşağıda Python/SQL motorunun hesapladığı kesin veriler yer almaktadır.
Sayıları ve toplamı doğrudan tablodan al, matematiksel toplamı değiştirme.

Veri Tablosu:
{df_res.to_string(index=False)}

Hesaplanan Kesin Toplam: {toplam_tutar:,.2f} TRY

Kullanıcı Sorusu: {soru}
Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return cevap["message"]["content"], "Mali Tablolar (SQL & Pandas Motoru)"


# 5. Yönlendirici (Router)
def answer_query(user_query: str):
    router_prompt = f"""Kullanıcı sorusunu sınıflandır. Sadece 'MEVZUAT' veya 'TABLO' yaz.
- Kayıp-kaçak oranları, amortisman süreleri, raporlama takvimleri, EPDK mevzuat maddeleri: MEVZUAT
- Kasa, banka, genel yönetim gideri, nakit varlık, bakiye, borç/alacak hesapları: TABLO

Soru: {user_query}
Karar:"""

    decision = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": router_prompt}]
    )["message"]["content"].strip().upper()

    if "TABLO" in decision:
        return query_financial(user_query)
    else:
        return query_mevzuat(user_query)