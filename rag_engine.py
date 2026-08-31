import json
import math
import os
import re
import pandas as pd
import ollama

VECTOR_DB_PATH = "db/mevzuat_vektorleri.json"
EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"


def clean_text(text: str) -> str:
    """Metindeki istenmeyen unicode karakterleri ve citation kalıplarını temizler."""
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    text = re.sub(r'\]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def format_try(val) -> str:
    """Sayısal tutarları Türk Lirası para birimi formatına dönüştürür."""
    try:
        val_float = float(val)
        if math.isnan(val_float) or pd.isna(val_float):
            return "0,00 TL"
        return f"{val_float:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(val)


def cosine_similarity(vec_a, vec_b):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0


def retrieve_mevzuat(soru: str, top_k: int = 3):
    """Vektör veritabanından en alakalı mevzuat metinlerini getirir."""
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
            if "Net Dağıtım Geliri" in kalem or "Satışlar" in kalem:
                try:
                    val_f = float(tutar_ham)
                    if not math.isnan(val_f):
                        toplam_gelir = val_f
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

    # 3. Mizan
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4)
    df_mizan.columns = [str(c).strip() for c in df_mizan.columns]
    df_mizan = df_mizan.dropna(subset=['Hesap Kodu'])

    kasa_toplam = 0.0
    banka_toplam = 0.0
    mizan_lines = []

    for _, row in df_mizan.iterrows():
        kod_str = str(row['Hesap Kodu']).strip().replace(".0", "")
        ana_kod = kod_str.split(".")[0]
        ad = str(row['Hesap Adı']).strip()
        b_bak = float(row['Borç Bakiye']) if ('Borç Bakiye' in row and pd.notna(row['Borç Bakiye'])) else 0.0
        a_bak = float(row['Alacak Bakiye']) if ('Alacak Bakiye' in row and pd.notna(row['Alacak Bakiye'])) else 0.0

        tutar_str = f"Borç Bakiyesi: {format_try(b_bak)}" if b_bak > 0 else f"Alacak Bakiyesi: {format_try(a_bak)}"
        mizan_lines.append(f"Hesap {kod_str} - {ad}: {tutar_str}")

        if ana_kod == "100":
            kasa_toplam += b_bak
        elif ana_kod == "102":
            banka_toplam += b_bak

    mizan_metni = "\n".join(mizan_lines)
    hazir_degerler_toplam = kasa_toplam + banka_toplam

    on_hesaplar = (
        f"- Kasa Hesabı Toplamı (100): {format_try(kasa_toplam)}\n"
        f"- Bankalar Hesabı Toplamı (102): {format_try(banka_toplam)}\n"
        f"- Toplam Hazır Değerler (Kasa + Banka): {format_try(hazir_degerler_toplam)}\n"
        f"- Toplam Düzenlenmiş Satış / Dağıtım Geliri: {format_try(toplam_gelir)}"
    )

    return gelir_metni, bilanco_metni, mizan_metni, on_hesaplar


def query_financial(soru: str):
    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    prompt = f"""Sen resmi şirket verilerini inceleyen bir mali müşavir ve finans analistisin.
Aşağıda şirketin 2024 yılı resmi mali tabloları yer almaktadır:

=== ÖZET DEĞERLER ===
{on_hesaplar}

=== GELİR TABLOSU ===
{gelir_metni}

=== BİLANÇO ===
{bilanco_metni}

=== MİZAN ===
{mizan_metni}

Kullanıcı Sorusu: {soru}

GÖREVİN:
1. Kullanıcı sorusuna tablolardaki gerçek değerlere göre net ve doğrudan yanıt ver.
2. Eğer genel bir grup kalemi (örneğin 'Ticari Alacaklar') sorulmuşsa hem bilançodaki ana grup toplamını hem de mizandaki alt hesapları net tutarlarıyla belirt.
3. Hesap kodu veya kalem sorulmuşsa ilgili satırın tutarını ve borç/alacak bakiyesini açıkça yaz.
4. İstenen veri tablolarda yoksa sadece 'Belgelerde bu bilgi bulunmamaktadır.' de.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    return clean_text(cevap["message"]["content"]), "Mali Tablolar (Mizan & Bilanço & Gelir Tablosu)"


def query_hybrid(soru: str):
    baglam_mevzuat, kaynaklar = retrieve_mevzuat(soru, top_k=3)
    gelir_metni, bilanco_metni, mizan_metni, on_hesaplar = load_formatted_financial_tables()

    prompt = f"""Sen enerji sektörü finans ve EPDK regülasyonu uzmanısın.
Aşağıda mevzuat metinleri ve şirketin resmi mali tabloları yer almaktadır:

=== MEVZUAT METİNLERİ ===
{baglam_mevzuat}

=== MALİ TABLOLAR VE ÖZET VERİLER ===
{on_hesaplar}
{gelir_metni}
{bilanco_metni}
{mizan_metni}

Kullanıcı Sorusu: {soru}

GÖREVİN VE ÇIKTI FORMATI:
1. **Mevzuat Açıklaması**: Sadece soruyla doğrudan ilgili mevzuat maddelerini ve kurallarını listele.
2. **Mali Tablo Tespiti**: Sadece soruda geçen hesap kalemlerinin (örn. 253, 257 veya 602) tablodaki bakiye tutarlarını yaz. Alakasız kasa/banka rakamlarını buraya ekleme.
3. **Finansal Değerlendirme**:
   - DVT sorularında: (253 - 257) farkını "Net Defter Değeri (Net Varlık Tabanı)" olarak belirt. Duran varlıkları gelir tablosu kârından veya hasılatından ASLA çıkarma ya da ekleme.
   - Kayıp-kaçak sorularında: Fiili teknik kayıp-kaçak oranının operasyonel bir veri olup mali tablolarda yer almadığını, bu nedenle hedefin aşılıp aşılmadığının tablodan bilinemeyeceğini belirt. Olası azami yaptırımın ise Toplam Dağıtım Geliri (209.500.000 TL) üzerinden azami %2 tavan indirimi (4.190.000 TL) olabileceğini açıkla. Uydurma matematik formülleri üretme.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    kaynak_str = ", ".join(kaynaklar) if kaynaklar else "EPDK Mevzuatı"
    return clean_text(cevap["message"]["content"]), f"Karma (Mali Tablolar & {kaynak_str})"


def query_mevzuat(soru: str):
    baglam, kaynaklar = retrieve_mevzuat(soru, top_k=3)
    if not baglam:
        return "Belgelerde bu bilgi bulunmamaktadır.", "Hata"

    prompt = f"""Sen resmi bir mevzuat uzmanısın.
Aşağıdaki mevzuat metinlerini kullanarak soruyu yanıtla.

=== MEVZUAT METNİ ===
{baglam}

Kullanıcı Sorusu: {soru}

GÖREVİN VE ÇIKTI FORMATI:
- Soruda istenen varlık veya maddeyi metinden bularak doğrudan yanıtla.
- Kullanıcı sadece yeni bir konuyu sorduysa (örn. bilgi işlem veya sayaçlar), önceki konuları (trafo merkezleri vb.) gereksiz yere cevaba dahil etme.
- Yanıtının altına mutlaka 'Referans: [Belge Adı] MADDE X' şeklinde kaynak maddesini ekle.
- Asla 'Satır 1:', 'Cevap:' gibi başlıklar kullanma.
- Bilgi metinde yoksa sadece 'Belgelerde bu bilgi bulunmamaktadır.' yaz.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    raw_content = clean_text(cevap["message"]["content"])

    ref_match = re.findall(r'\[(EPDK_[^\]]+\.txt)\]', raw_content)
    if ref_match:
        kaynak_str = ", ".join(sorted(list(set(ref_match))))
    else:
        kaynak_str = ", ".join(kaynaklar) if kaynaklar else "EPDK Mevzuatı"

    return raw_content, f"EPDK Mevzuatı ({kaynak_str})"


def rewrite_query_with_context(user_query: str, user_messages: list) -> str:
    """Kullanıcının kısa veya takip sorularını önceki ana konuya bağlayarak tamamlar."""
    if not user_messages:
        return user_query

    temiz_q = user_query.strip().lower()

    if len(temiz_q.split()) >= 6 and not any(k in temiz_q for k in ["peki", "ya", "ise", "bunun", "ondan"]):
        return user_query

    # Geçmişteki en son tam soru cümlesini bul
    soru_ekleri = ["nedir", "kaçtır", "ne kadar", "kaç", "nasıl", "kimdir", "nelerdir",
                   "mi", "mı", "mu", "mü", "yıldır", "sınırı"]
    ana_soru = user_messages[-1]
    for msg in reversed(user_messages):
        m_temiz = msg.strip().lower()
        if len(m_temiz.split()) >= 4 and any(k in m_temiz for k in soru_ekleri):
            ana_soru = msg
            break

    prompt = f"""Kullanıcının önceki sorusu: "{ana_soru}"
Kullanıcının yeni kısa girdisi: "{user_query}"

GÖREV:
Yeni girdi önceki sorunun bir devamıdır. Önceki sorunun genel soru fiilini/kalıbını al, ancak önceki sorudaki ESKİ ÖZNEYİ TAMAMEN AT.
Yeni girdideki konuyu sorunun tek öznesi yap.

Örnekler:
- Önceki: "Trafo merkezleri için amortisman süresi kaç yıldır?" | Yeni: "bilgi işlem" -> "Bilgi işlem ve SCADA sistemleri için amortisman süresi kaç yıldır?"
- Önceki: "Trafo merkezleri için amortisman süresi kaç yıldır?" | Yeni: "sayaçlar" -> "Sayaçlar ve ölçüm sistemleri için amortisman süresi kaç yıldır?"
- Önceki: "A grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?" | Yeni: "b ve c" -> "B ve C grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?"

Sadece yeni Türkçe soruyu tek satır olarak yaz:"""

    res = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    yeni_soru = clean_text(res["message"]["content"]).strip('"\n ')
    return yeni_soru if yeni_soru else user_query


def normalize_text(text: str) -> str:
    """Türkçe karakterleri normalize ederek küçük harfe çevirir."""
    mapping = {'İ': 'i', 'I': 'ı', 'Ş': 'ş', 'Ğ': 'ğ', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç'}
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text.lower().strip()


def answer_query(user_query: str, chat_history: list = None, last_focused_index: int = None):
    q_norm = normalize_text(user_query)

    # Gerçek kullanıcı mesajlarını ayıkla
    meta_kaliplar = ["ne sordum", "neydi", "özetle", "ozetle", "neler konustuk",
                     "kelimesi", "hangi soruda", "sonrakinde", "oncekinde"]
    raw_user_msgs = [msg["content"] for msg in chat_history if msg["role"] == "user"] if chat_history else []
    user_messages = [m for m in raw_user_msgs if not any(k in normalize_text(m) for k in meta_kaliplar)]
    total_questions = len(user_messages)

    # 1. SOHBET ÖZETİ
    ozet_kaliplari = ["neler konustuk", "neler konuştuk", "özetle", "ozetle",
                      "özet", "ozet", "bütün sorular", "tum sorular", "tüm mesajlar"]
    if any(k in q_norm for k in ozet_kaliplari):
        if not user_messages:
            return "Henüz konuyla ilgili bir soru geçmişi bulunmamaktadır.", "Sohbet Belleği", last_focused_index
        liste = "\n".join([f"{i + 1}. {m}" for i, m in enumerate(user_messages)])
        return f"Şu ana kadar sorduğunuz {total_questions} asıl soru:\n\n{liste}", "Sohbet Belleği", last_focused_index

    # 2. MESAJ İÇİ KELİME SORGULARI
    kelime_sorgu_kaliplari = ["kelimesi neydi", "kelime nedir", "kelimeyi söyle", "kelime hangisi"]
    sayilar = [int(s) for s in re.findall(r'\d+', user_query)]

    if (any(k in q_norm for k in kelime_sorgu_kaliplari) or ("kelime" in q_norm and sayilar)) and user_messages:
        target_idx = sayilar[0] - 1 if sayilar else 0
        if 0 <= target_idx < total_questions:
            target_msg = user_messages[target_idx]
            kelimeler = [w for w in re.sub(r'[^\w\s]', '', target_msg).split() if w]
            k_sira = sayilar[-1] if len(sayilar) > 1 else 1
            if 1 <= k_sira <= len(kelimeler):
                return (
                    f"İlgili sorudaki {k_sira}. kelime: **\"{kelimeler[k_sira - 1]}\"**",
                    "Sohbet Belleği",
                    target_idx + 1
                )
            return (
                f"Soruda toplam {len(kelimeler)} kelime var. {k_sira}. kelime bulunamadı.",
                "Sohbet Belleği",
                target_idx + 1
            )

    # 3. İÇERİK ARAMASI
    if any(k in q_norm for k in ["hangi", "nerede", "ne zaman"]):
        stop_words = {
            "hangi", "mesajda", "mesaj", "soruda", "soru", "sordumu", "sordum",
            "sordugumu", "sorduğumu", "nerede", "ne", "zaman", "diye", "veya",
            "gecen", "geçen", "içeren", "hakkinda", "hakkında", "dedim",
            "dedigimi", "dediğimi", "yazdim", "yazdım", "soyledim", "söyledim",
            "bahsettim"
        }
        raw_words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{2,}\b', q_norm)
        arananlar = [w for w in raw_words if w not in stop_words]

        if arananlar and user_messages:
            for i, m in enumerate(user_messages):
                m_norm = normalize_text(m)
                if any(w in m_norm.split() or w in m_norm for w in arananlar):
                    return f"Bu konuyu baştan **{i + 1}. sorunuzda** sormuştunuz: \"{m}\"", "Sohbet Belleği", i + 1
            return "Sohbet geçmişinde bu içerikle eşleşen bir soru bulunamadı.", "Sohbet Belleği", last_focused_index

    # 4. SOHBET BELLEĞİ (Bağıl ve Mutlak İndeksler)
    gecmis_kaliplari = ["mesaj", "soru", "sord", "neydi", "ilk", "son",
                        "onceki", "önceki", "sonraki", "sonrakinde", "ondan"]
    if any(k in q_norm for k in gecmis_kaliplari) and user_messages and not any(
            k in q_norm for k in ["hedef", "amortisman", "gelir", "gider"]):
        delta = sayilar[0] if sayilar else 1

        if any(k in q_norm for k in ["ondan", "sonraki", "sonrakinde", "onceki", "önceki"]):
            base_idx = last_focused_index if last_focused_index is not None else total_questions
            if any(k in q_norm for k in ["sonra", "sonraki", "sonrakinde"]):
                target_idx = base_idx + delta
            else:
                target_idx = base_idx - delta

            if 1 <= target_idx <= total_questions:
                return (
                    f"Baştan {target_idx}. sorunuzda şunu sormuştunuz: \"{user_messages[target_idx - 1]}\"",
                    "Sohbet Belleği",
                    target_idx
                )
            return (
                f"Toplam {total_questions} asıl sorunuz var. Hesaplanmak istenen {target_idx}. soru mevcut değil.",
                "Sohbet Belleği",
                last_focused_index
            )

        if sayilar:
            sira = sayilar[0]
            if 1 <= sira <= total_questions:
                return (
                    f"Baştan {sira}. sorunuzda şunu sormuştunuz: \"{user_messages[sira - 1]}\"",
                    "Sohbet Belleği",
                    sira
                )
            return (
                f"Toplam {total_questions} asıl sorunuz var. {sira}. soru bulunamadı.",
                "Sohbet Belleği",
                last_focused_index
            )

        if "ilk" in q_norm or "1." in q_norm:
            return f"İlk sorunuzda şunu sormuştunuz: \"{user_messages[0]}\"", "Sohbet Belleği", 1
        if "son" in q_norm:
            return f"Son sorunuzda şunu sormuştunuz: \"{user_messages[-1]}\"", "Sohbet Belleği", total_questions

    # 5. KARMA / HİBRİT SORGULAR
    mevzuat_kelimeleri = [r"\bhedef\b", r"\bkayıp\b", r"\bkaçak\b", r"\btebliğ\b",
                          r"\byönetmelik\b", r"\bdvt\b", r"\bamortisman\b", r"\bsüre\b", r"\bsüresi\b"]
    finans_kelimeleri = [r"\bgelir\b", r"\bgider\b", r"\byönetim\b", r"\bkasa\b",
                         r"\bbanka\b", r"\bmizan\b", r"\bbilanço\b", r"\btutar\b",
                         r"\bhesap\b", r"\btl\b", r"\btry\b", r"\bnakit\b",
                         r"\balacak\b", r"\bborç\b", r"\bticari\b", r"\bşüpheli\b",
                         r"\baktif\b", r"\bpasif\b", r"\bözkaynak\b", r"\bkâr\b",
                         r"\bzarar\b", r"\btopla\b", r"\btoplam\b", r"\bbakiye\b"]

    has_mevzuat = any(re.search(k, q_norm) for k in mevzuat_kelimeleri)
    has_finans = any(re.search(k, q_norm) for k in finans_kelimeleri)
    has_hesap_kodu = bool(re.search(r'\b[1-7]\d{2}\b', user_query))
    has_dvt = "dvt" in q_norm or "düzenlenmiş varlık" in q_norm

    if has_dvt or (has_mevzuat and has_finans) or ("tablo" in q_norm and has_mevzuat):
        ans, src = query_hybrid(user_query)
        return ans, src, last_focused_index

    # 6. FİNANSAL TABLOLAR
    if has_hesap_kodu or has_finans:
        ans, src = query_financial(user_query)
        return ans, src, last_focused_index

    # 7. MEVZUAT RAG (Dinamik Tamamlanmış Soru ile)
    aranacak_soru = rewrite_query_with_context(user_query, user_messages)
    ans, src = query_mevzuat(aranacak_soru)
    return ans, src, last_focused_index