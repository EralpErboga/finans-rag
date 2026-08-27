import os
import json
import math
import sqlite3
import pandas as pd
import ollama
import re

VECTOR_DB_PATH = "db/mevzuat_vektorleri.json"
EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"


# Çince/Bozuk Karakter Temizleyici
def clean_text(text: str) -> str:
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    return text.strip()


# Güvenli Para Formatlayıcı (Örn: 24.300.000,00 TL)
def format_try(val) -> str:
    try:
        val_float = float(val)
        if math.isnan(val_float):
            return ""
        return f"{val_float:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(val)


# Ortak Mali Tablo Yükleyici & Formatlayıcı
def load_formatted_financial_tables():
    xls = pd.ExcelFile(EXCEL_PATH)

    # 1. Gelir Tablosu
    df_gelir = pd.read_excel(xls, sheet_name="Gelir Tablosu").dropna(how='all')
    gelir_lines = []
    for _, row in df_gelir.iterrows():
        kalem = str(row.iloc[0]).strip()
        tutar = format_try(row.iloc[1]) if len(row) > 1 else ""
        gelir_lines.append(f"{kalem}: {tutar}")
    gelir_metni = "\n".join(gelir_lines)

    # 2. Bilanço
    df_bilanco = pd.read_excel(xls, sheet_name="Bilanço").dropna(how='all')
    bilanco_lines = []
    for _, row in df_bilanco.iterrows():
        kalem = str(row.iloc[0]).strip()
        tutar = format_try(row.iloc[1]) if len(row) > 1 else ""
        bilanco_lines.append(f"{kalem}: {tutar}")
    bilanco_metni = "\n".join(bilanco_lines)

    # 3. Mizan Tablosu
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4).dropna(subset=['Hesap Kodu'])
    mizan_lines = []
    for _, row in df_mizan.iterrows():
        kod = int(row['Hesap Kodu'])
        ad = str(row['Hesap Adı']).strip()
        b_bak = float(row['Borç Bakiye']) if pd.notna(row['Borç Bakiye']) else 0.0
        a_bak = float(row['Alacak Bakiye']) if pd.notna(row['Alacak Bakiye']) else 0.0

        tutar_str = f"Borç: {format_try(b_bak)}" if b_bak > 0 else f"Alacak: {format_try(a_bak)}"
        mizan_lines.append(f"Hesap {kod} - {ad}: {tutar_str}")
    mizan_metni = "\n".join(mizan_lines)

    # Önceden hesaplanmış kritik hazır değerler
    kasa_val = df_mizan.loc[df_mizan['Hesap Kodu'] == 100, 'Borç Bakiye'].sum()
    banka_val = df_mizan.loc[df_mizan['Hesap Kodu'] == 102, 'Borç Bakiye'].sum()
    hazir_degerler_toplam = kasa_val + banka_val

    on_hesaplar = (
        f"- Kasa Hesabı (100): {format_try(kasa_val)}\n"
        f"- Bankalar Hesabı (102): {format_try(banka_val)}\n"
        f"- Toplam Hazır Değerler (Kasa + Banka): {format_try(hazir_degerler_toplam)}"
    )

    return gelir_metni, bilanco_metni, mizan_metni, on_hesaplar


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

    baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for item in db])
    kaynaklar = ", ".join(list(set([item["source"] for item in db])))

    prompt = f"""Sen enerji şirketi finans ve mevzuat biriminde görevli bir danışmansın.
Aşağıda şirketin tabi olduğu EPDK mevzuat belgeleri yer almaktadır:

=== EPDK MEVZUAT METİNLERİ ===
{baglam}

Kullanıcı Sorusu: {soru}

KURALLAR:
1. Sadece yukarıdaki bağlamda yer alan maddelere ve kurallara dayanarak cevap ver.
2. Bağlam dışından kurum, kurul veya kural uydurma.
3. Cevabın sonuna referans aldığın belge adını ve madde başlığını ekle.

Cevap (Türkçe):"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    return clean_text(cevap["message"]["content"]), f"EPDK Mevzuatı ({kaynaklar})"


# 3. SQLite Bağlantısı
def get_db_connection():
    conn = sqlite3.connect(":memory:")
    xls = pd.ExcelFile(EXCEL_PATH)
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4).dropna(subset=['Hesap Kodu'])
    df_mizan['Hesap Kodu'] = df_mizan['Hesap Kodu'].astype(int)
    df_mizan.columns = ['hesap_kodu', 'hesap_adi', 'hesap_grubu', 'borc', 'alacak', 'borc_bakiye', 'alacak_bakiye']
    df_mizan.to_sql("mizan", conn, index=False, if_exists="replace")
    return conn


# 4. Mali Tablo Hattı
def query_financial(soru: str):
    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    prompt = f"""Sen üst düzey bir mali müşavir ve denetim uzmanısın.
Aşağıda şirketin 2024 yılı resmi mali tabloları yer almaktadır:

=== GELİR TABLOSU ===
{gelir_metni}

=== BİLANÇO ===
{bilanco_metni}

=== MİZAN HESAPLARI ===
{mizan_metni}

=== ÖN HESAPLANMIŞ KESİN DEĞERLER ===
{on_hesaplar}

Kullanıcı Sorusu: {soru}

GÖREVİN VE KESİN KURALLAR:
1. İstenen tutarları tablolardaki ve ön hesaplanmış resmi değerlerdeki net haliyle (noktası, virgülüyle) yaz.
2. Kafandan ek aritmetik veya tahmini hesaplama yapma.
3. Maddeler halinde net ve anlaşılır açıkla.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    return clean_text(cevap["message"]["content"]), "Mali Tablolar (Bilanço & Gelir Tablosu & Mizan)"
# 5. Birleşik (Hibrit) Hat: Görev Ayrımı Netleştirilmiş Mimari
def query_combined(soru: str):
    if not os.path.exists(VECTOR_DB_PATH):
        return "Vektör veritabanı bulunamadı.", "Hata"

    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    # --- 1. AŞAMA: Sadece Mali/Sayısal Verileri Çek ---
    mali_prompt = f"""Sen bir mali müşavirsin. Aşağıdaki mali tablolara bakarak kullanıcının sorusundaki SADECE finansal/parasal/gider tutarlarını cevapla.

=== GELİR TABLOSU ===
{gelir_metni}

=== BİLANÇO ===
{bilanco_metni}

=== MİZAN HESAPLARI ===
{mizan_metni}

Soru: {soru}

GÖREV: 
- SADECE soruda istenen parasal toplamları veya hesap tutarlarını yaz.
- Soru içindeki mevzuat/kural kısımlarına kesinlikle girme, mizanı gereksiz yere baştan sona listeleme.
- Başlık: **Mali Veriler & Gider/Gelir Analizi**"""

    mali_yanit = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": mali_prompt}],
        options={"temperature": 0.0}
    )["message"]["content"].strip()

    # --- 2. AŞAMA: Sadece Mevzuat/Yönetmelik Kuralını Çek ---
    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)
    mevzuat_baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for item in db])

    mevzuat_prompt = f"""Sen bir EPDK mevzuat danışmanısın.
Aşağıda EPDK mevzuat belgeleri yer almaktadır:

{mevzuat_baglam}

Soru: {soru}

GÖREV:
- Soruda geçen mevzuat/yönetmelik kuralını (Örn: Düzenlenmiş Varlık Tabanı koşulları, faydalı ömür, hedefler vb.) ilgili maddeye göre açıkla.
- En sona ilgili mevzuat belgesi adını ve Madde numarasını referans olarak ekle.
- Başlık: **İlgili EPDK Mevzuat Hükmü**"""

    mevzuat_yanit = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": mevzuat_prompt}],
        options={"temperature": 0.0}
    )["message"]["content"].strip()

    birlestirilmis_cevap = f"{mali_yanit}\n\n---\n\n{mevzuat_yanit}"
    return clean_text(birlestirilmis_cevap), "Hibrit Motor (Mali Tablolar & Mevzuat RAG)"
# 6. Yönlendirici (Router)
def answer_query(user_query: str):
    router_prompt = f"""Sen bir sorgu sınıflandırıcısısın. Kullanıcı sorusunu dikkatle oku ve sadece 3 kategoriden birini seç:

KATEGORİ TANIMLARI:
1. MEVZUAT:
   - Soru amortisman süreleri, faydalı ömür (trafo, scada, sayaç vb. kaç yıl), yasal oranlar, EPDK yönetmelikleri, tebliğler, raporlama takvimi, son bildirim tarihleri veya yasal kurallar hakkındaysa.

2. TABLO:
   - Soru şirketin mevcut bilançosundaki, gelir tablosundaki veya mizanındaki TL tutarları, hesap bakiyeleri, kâr, nakit, borç veya varlık toplamları hakkındaysa (Mevzuat/yönetmelik kuralı sormuyorsa).

3. BIRLESIK:
   - Soru HEM yasal kuralı/faydalı ömrü/tebliği HEM DE şirketin tablosundaki TL tutarını AYNI ANDA soruyorsa.

Kullanıcı Sorusu: {user_query}
Sınıflandırma (Sadece tek bir kelime: MEVZUAT, TABLO veya BIRLESIK):"""

    decision = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": router_prompt}],
        options={"temperature": 0.0}
    )["message"]["content"].strip().upper()

    # Yönlendirme Kontrolü
    if "BIRLES" in decision or "BİRLEŞ" in decision or "BIRLEŞ" in decision:
        return query_combined(user_query)
    elif "MEVZUAT" in decision or "YONETMELIK" in decision or "TEBLIG" in decision:
        return query_mevzuat(user_query)
    elif "TABLO" in decision or "MALI" in decision:
        return query_financial(user_query)
    else:
        # Kararsız kalırsa varsayılan olarak mevzuata baksın
        return query_mevzuat(user_query)