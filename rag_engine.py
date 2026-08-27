import os
import json
import math
import re
import pandas as pd
import ollama

VECTOR_DB_PATH = "db/mevzuat_vektorleri.json"
EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"


def clean_text(text: str) -> str:
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    return text.strip()


def format_try(val) -> str:
    try:
        val_float = float(val)
        if math.isnan(val_float) or pd.isna(val_float):
            return "0,00 TL"
        return f"{val_float:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(val)


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0


def retrieve_mevzuat(soru: str, top_k: int = 2):
    if not os.path.exists(VECTOR_DB_PATH):
        return "", []

    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    # Soruyu bge-m3 ile vektörleştir
    res = ollama.embeddings(model="bge-m3", prompt=soru)
    soru_vektoru = res["embedding"]

    # Benzerlik sıralaması
    skorlar = []
    for item in db:
        skor = cosine_similarity(soru_vektoru, item["embedding"])
        skorlar.append((skor, item))

    skorlar.sort(key=lambda x: x[0], reverse=True)
    en_iyi = [item for _, item in skorlar[:top_k]]

    baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for item in en_iyi])
    kaynaklar = list(set([item["source"] for item in en_iyi]))
    return baglam, kaynaklar


def load_formatted_financial_tables():
    if not os.path.exists(EXCEL_PATH):
        return "", "", "", ""

    xls = pd.ExcelFile(EXCEL_PATH)

    # 1. Gelir Tablosu
    df_gelir = pd.read_excel(xls, sheet_name="Gelir Tablosu").dropna(how='all')
    gelir_lines = []
    toplam_gelir = 0.0
    for _, row in df_gelir.iterrows():
        if len(row) > 1:
            kalem = str(row.iloc[0]).strip()
            tutar_ham = row.iloc[1]
            gelir_lines.append(f"{kalem}: {format_try(tutar_ham)}")
            try:
                val_f = float(tutar_ham)
                if not math.isnan(val_f) and val_f > 0:
                    toplam_gelir += val_f
            except (ValueError, TypeError):
                pass
    gelir_metni = "\n".join(gelir_lines)

    # 2. Bilanço
    df_bilanco = pd.read_excel(xls, sheet_name="Bilanço").dropna(how='all')
    bilanco_lines = []
    for _, row in df_bilanco.iterrows():
        if len(row) > 1:
            kalem = str(row.iloc[0]).strip()
            bilanco_lines.append(f"{kalem}: {format_try(row.iloc[1])}")
    bilanco_metni = "\n".join(bilanco_lines)

    # 3. Mizan Tablosu
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4).dropna(subset=['Hesap Kodu'])
    kasa_toplam = 0.0
    banka_toplam = 0.0
    yonetim_gideri_toplam = 0.0
    mizan_lines = []

    for _, row in df_mizan.iterrows():
        kod_str = str(row['Hesap Kodu']).strip().replace(".0", "")
        ana_kod = kod_str.split(".")[0]
        ad = str(row['Hesap Adı']).strip()
        b_bak = float(row['Borç Bakiye']) if pd.notna(row['Borç Bakiye']) else 0.0
        a_bak = float(row['Alacak Bakiye']) if pd.notna(row['Alacak Bakiye']) else 0.0

        tutar_str = f"Borç: {format_try(b_bak)}" if b_bak > 0 else f"Alacak: {format_try(a_bak)}"
        mizan_lines.append(f"Hesap {kod_str} - {ad}: {tutar_str}")

        if ana_kod == "100" or kod_str.startswith("100"):
            kasa_toplam += b_bak
        elif ana_kod == "102" or kod_str.startswith("102"):
            banka_toplam += b_bak
        # 632 veya 770 kodları ile 'Genel Yönetim' başlıklarını kapsar
        elif ana_kod in ["632", "770"] or kod_str.startswith(("632", "770")) or "genel yönetim" in ad.lower():
            yonetim_gideri_toplam += b_bak

    mizan_metni = "\n".join(mizan_lines)
    hazir_degerler_toplam = kasa_toplam + banka_toplam

    on_hesaplar = (
        f"- Kasa Hesabı Toplamı (100): {format_try(kasa_toplam)}\n"
        f"- Bankalar Hesabı Toplamı (102): {format_try(banka_toplam)}\n"
        f"- Toplam Hazır Değerler (Kasa + Banka): {format_try(hazir_degerler_toplam)}\n"
        f"- 2024 Yılı Toplam Gelir: {format_try(toplam_gelir)}\n"
        f"- 2024 Yılı Toplam Genel Yönetim Gideri (632 / 770 Grubu): {format_try(yonetim_gideri_toplam)}"
    )

    return gelir_metni, bilanco_metni, mizan_metni, on_hesaplar


def query_financial(soru: str):
    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    prompt = f"""Sen resmi mali tablolar denetçisisin.
Aşağıda şirketin 2024 yılı mali tabloları ve önceden hesaplanmış resmi değerleri yer almaktadır:

=== ÖN HESAPLANMIŞ RESMİ DEĞERLER ===
{on_hesaplar}

=== GELİR TABLOSU ===
{gelir_metni}

=== BİLANÇO ===
{bilanco_metni}

=== MİZAN HESAPLARI ===
{mizan_metni}

Kullanıcı Sorusu: {soru}

GÖREVİN:
1. Sorulan tutarı ön hesaplanmış resmi değerlerden veya ilgili tablodan bularak net olarak yaz.
2. Bilgi tablolarda yoksa "Belgelerde bu bilgi bulunmamaktadır." de.
3. Kafandan tahmin veya ek aritmetik yürütme.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    return clean_text(cevap["message"]["content"]), "Mali Tablolar (Mizan & Bilanço & Gelir Tablosu)"


def query_mevzuat(soru: str):
    baglam, kaynaklar = retrieve_mevzuat(soru, top_k=2)
    if not baglam:
        return "İlgili mevzuat belgesi bulunamadı. Lütfen ingest.py scriptini çalıştırın.", "Hata"

    prompt = f"""Sen enerji sektörü mevzuat danışmanısın.
Aşağıda EPDK mevzuatından ilgili maddeler yer almaktadır:

=== İLGİLİ MEVZUAT METİNLERİ ===
{baglam}

Kullanıcı Sorusu: {soru}

KURALLAR:
1. Sadece yukarıdaki mevzuat maddelerine dayanarak net cevap ver.
2. Bilgi metinde yoksa "Belgelerde bu bilgi bulunmamaktadır." de.
3. Cevabın altına referans aldığın belge adını ve madde başlığını ekle.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    kaynak_str = ", ".join(kaynaklar) if kaynaklar else "EPDK Mevzuatı"
    return clean_text(cevap["message"]["content"]), f"EPDK Mevzuatı ({kaynak_str})"


def answer_query(user_query: str, chat_history: list = None):
    # 1. Sohbet Geçmişi / Hafıza Tespiti (Deterministik Python Katmanı)
    q_lower = user_query.lower()
    gecmis_tetikleyicileri = [
        "mesaj", "soru", "yaz", "sord", "konus", "konuş", "neydi",
        "ilk", "son", "onceki", "önceki", "az önce", "azonce", "bastan", "baştan",
        "ondan", "sonrakinde", "peki ya"
    ]

    user_messages = [msg["content"] for msg in chat_history if msg["role"] == "user"] if chat_history else []

    if any(t in q_lower for t in gecmis_tetikleyicileri) and user_messages:
        sayilar = re.findall(r'\d+', user_query)
        if sayilar:
            sira = int(sayilar[0])
            if 1 <= sira <= len(user_messages):
                return f"Baştan {sira}. mesajınızda şunu yazmıştınız: \"{user_messages[sira - 1]}\"", "Sohbet Belleği"
            return f"Toplam {len(user_messages)} mesajınız var. {sira}. mesaj bulunamadı.", "Sohbet Belleği"

        if "ilk" in q_lower:
            return f"İlk mesajınızda şunu yazmıştınız: \"{user_messages[0]}\"", "Sohbet Belleği"

        if any(w in q_lower for w in ["önceki", "az önce", "ondan", "sonrakinde"]):
            hedef = user_messages[-2] if len(user_messages) >= 2 else user_messages[0]
            return f"Önceki mesajınızda şunu yazmıştınız: \"{hedef}\"", "Sohbet Belleği"

        return f"Son mesajınızda şunu yazmıştınız: \"{user_messages[-1]}\"", "Sohbet Belleği"

    # 2. Yönlendirme (Router)
    finans_kelimeleri = ["gelir", "gider", "yönetim", "kasa", "banka", "mizan", "bilanço", "tutar", "hesap", "tl",
                         "nakit"]
    if any(k in q_lower for k in finans_kelimeleri):
        return query_financial(user_query)

    return query_mevzuat(user_query)