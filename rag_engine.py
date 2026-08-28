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

    res = ollama.embeddings(model="bge-m3", prompt=soru)
    soru_vektoru = res["embedding"]

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
        elif ana_kod in ["632", "770", "630"] or "genel yönetim" in ad.lower():
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
3. Kafandan tahmin yürütme.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    return clean_text(cevap["message"]["content"]), "Mali Tablolar (Mizan & Bilanço & Gelir Tablosu)"
def query_hybrid(soru: str):
    baglam_mevzuat, kaynaklar = retrieve_mevzuat(soru, top_k=2)
    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    prompt = f"""Sen enerji sektörü finans ve mevzuat danışmanısın.
Aşağıda şirketin mali verileri ve EPDK mevzuatı yer almaktadır:

=== MEVZUAT METİNLERİ ===
{baglam_mevzuat}

=== MALİ TABLOLAR & HESAPLAMALAR ===
{on_hesaplar}
{gelir_metni}

Kullanıcı Sorusu: {soru}

GÖREVİN:
1. Mevzuat hedeflerini (varsa oranları ve gelir tavanı indirim kurallarını) özetle.
2. Şirketin mali tablolarındaki ilgili gelir/gider kalemini belirt.
3. Bu durumun finansal etki analizini tek bir sonuç paragrafında net ve tekrara düşmeden açıkla.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "repeat_penalty": 1.2
        }
    )
    kaynak_str = ", ".join(kaynaklar) if kaynaklar else "EPDK Mevzuatı"
    return clean_text(cevap["message"]["content"]), f"Karma (Mali Tablolar & {kaynak_str})"

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


def normalize_text(text: str) -> str:
    mapping = {'İ': 'i', 'I': 'ı', 'Ş': 'ş', 'Ğ': 'ğ', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç'}
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text.lower().strip()


def answer_query(user_query: str, chat_history: list = None, last_focused_index: int = None):
    q_norm = normalize_text(user_query)
    user_messages = [msg["content"] for msg in chat_history if msg["role"] == "user"] if chat_history else []
    total_questions = len(user_messages)

    # 1. TÜM SOHBETİN ÖZETİ / LİSTESİ
    if any(k in q_norm for k in ["neler konustuk", "neler konuştuk", "özetle", "ozetle", "özet", "ozet", "bütün sorular", "tum sorular", "tüm mesajlar"]):
        if not user_messages:
            return "Henüz bir mesaj geçmişi bulunmamaktadır.", "Sohbet Belleği", last_focused_index
        liste = "\n".join([f"{i + 1}. {m}" for i, m in enumerate(user_messages)])
        return f"Şu ana kadar sorduğunuz {total_questions} soru:\n\n{liste}", "Sohbet Belleği", last_focused_index

    # 2. MESAJ İÇİ KELİME SORGULARI (Örn: "ilk mesajın 3. kelimesi ne")
    if ("kelime" in q_norm or "kelimesi" in q_norm) and user_messages:
        sayilar = [int(s) for s in re.findall(r'\d+', user_query)]
        target_idx = 0
        if "ilk" in q_norm or "1." in q_norm:
            target_idx = 0
        elif "son" in q_norm:
            target_idx = total_questions - 1
        elif sayilar:
            target_idx = sayilar[0] - 1

        if 0 <= target_idx < total_questions:
            target_msg = user_messages[target_idx]
            kelimeler = [w for w in re.sub(r'[^\w\s]', '', target_msg).split() if w]
            k_sira = sayilar[-1] if len(sayilar) > 1 else 1
            if 1 <= k_sira <= len(kelimeler):
                return f"İlgili sorudaki {k_sira}. kelime: **\"{kelimeler[k_sira - 1]}\"**", "Sohbet Belleği", target_idx + 1
            return f"Soruda toplam {len(kelimeler)} kelime var. {k_sira}. kelime bulunamadı.", "Sohbet Belleği", target_idx + 1

    # 3. İÇERİK BAZLI GEÇMİŞ ARAMASI ("hangi soruda / nerede sordum")
    if any(k in q_norm for k in ["hangi", "nerede", "ne zaman"]):
        stop_words = {
            "hangi", "mesajda", "mesaj", "soruda", "soru", "sordumu", "sordum", "sordugumu", "sorduğumu",
            "nerede", "ne", "zaman", "diye", "veya", "gecen", "geçen", "içeren", "hakkinda", "hakkında",
            "dedim", "dedigimi", "dediğimi", "yazdim", "yazdım", "soyledim", "söyledim", "bahsettim"
        }
        raw_words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{2,}\b', q_norm)
        arananlar = [w for w in raw_words if w not in stop_words]

        if arananlar and user_messages:
            for i, m in enumerate(user_messages):
                m_norm = normalize_text(m)
                if any(w in m_norm.split() or w in m_norm for w in arananlar):
                    return f"Bu konuyu baştan **{i + 1}. sorunuzda** sormuştunuz: \"{m}\"", "Sohbet Belleği", i + 1
            for i, m in enumerate(user_messages):
                m_norm = normalize_text(m)
                if any(w[:4] in m_norm for w in arananlar if len(w) >= 4):
                    return f"Bu konuyu baştan **{i + 1}. sorunuzda** sormuştunuz: \"{m}\"", "Sohbet Belleği", i + 1
            return "Sohbet geçmişinde bu içerikle eşleşen bir soru bulunamadı.", "Sohbet Belleği", last_focused_index

    # 4. DİNAMİK / BAĞIL GEÇMİŞ VE SIRALI GEÇMİŞ SORGULARI
    gecmis_tetikleyicileri = [
        "mesaj", "soru", "yaz", "sord", "neydi", "ilk", "son",
        "onceki", "önceki", "az önce", "bastan", "baştan", "ondan", "sonraki"
    ]

    if any(t in q_norm for t in gecmis_tetikleyicileri) and user_messages:
        sayilar = [int(s) for s in re.findall(r'\d+', user_query)]
        delta = sayilar[0] if sayilar else 1

        if "ondan" in q_norm or "sonraki" in q_norm or "onceki" in q_norm or "önceki" in q_norm:
            base_idx = last_focused_index if last_focused_index is not None else total_questions
            if "sonra" in q_norm or "sonraki" in q_norm:
                target_idx = base_idx + delta
            else:
                target_idx = base_idx - delta

            if 1 <= target_idx <= total_questions:
                return f"Baştan {target_idx}. sorunuzda şunu sormuştunuz: \"{user_messages[target_idx - 1]}\"", "Sohbet Belleği", target_idx
            return f"Toplam {total_questions} sorunuz var. Hesaplanmak istenen {target_idx}. soru mevcut değil.", "Sohbet Belleği", last_focused_index

        if sayilar:
            sira = sayilar[0]
            if 1 <= sira <= total_questions:
                return f"Baştan {sira}. sorunuzda şunu sormuştunuz: \"{user_messages[sira - 1]}\"", "Sohbet Belleği", sira
            return f"Toplam {total_questions} sorunuz var. {sira}. soru bulunamadı.", "Sohbet Belleği", last_focused_index

        if "ilk" in q_norm or "1." in q_norm:
            return f"İlk sorunuzda şunu sormuştunuz: \"{user_messages[0]}\"", "Sohbet Belleği", 1

        return f"Son sorunuzda şunu sormuştunuz: \"{user_messages[-1]}\"", "Sohbet Belleği", total_questions

    # 5. KARMA / HİBRİT SORGULAR (Mevzuat + Mali Etki)
    mevzuat_kelimeleri = ["hedef", "kayıp", "kaçak", "tebliğ", "yönetmelik", "dvt", "amortisman süresi"]
    finans_kelimeleri = ["gelir", "gider", "yönetim", "kasa", "banka", "mizan", "bilanço", "tutar", "hesap", "tl", "nakit", "alacak", "borç", "ticari", "şüpheli", "aktif", "pasif", "özkaynak", "kâr", "zarar"]

    has_mevzuat = any(k in q_norm for k in mevzuat_kelimeleri)
    has_finans = any(k in q_norm for k in finans_kelimeleri)

    if has_mevzuat and has_finans and len(q_norm.split()) > 5:
        ans, src = query_hybrid(user_query)
        return ans, src, last_focused_index

    # 6. FİNANSAL TABLOLAR
    if has_finans:
        ans, src = query_financial(user_query)
        return ans, src, last_focused_index

    # 7. MEVZUAT RAG (Kısa takip sorularında bağlam zenginleştirme)
    arama_metni = user_query
    if len(q_norm.split()) <= 3 and user_messages:
        arama_metni = f"{user_messages[-1]} {user_query}"

    ans, src = query_mevzuat(arama_metni)
    return ans, src, last_focused_index