import json
import math
import os
import re
import pandas as pd
import ollama

VECTOR_DB_PATH = "db/mevzuat_vektorleri.json"
EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"
def clean_text(text: str) -> str:
    """Metindeki istenmeyen unicode karakterleri, noktalama ve dosya adı kalıplarını temizler."""
    # Çince / Japonca / Asya karakterleri ve noktalama işaretlerini temizle (örn: 。)
    text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', text)
    # [EPDK_...txt] gibi köşeli parantezli dosya adlarını ve önündeki iki noktayı temizle
    text = re.sub(r':?\s*\[EPDK_[^\]]+\.txt\]', '', text, flags=re.IGNORECASE)
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
from collections import Counter

def retrieve_mevzuat(soru: str, top_k: int = 4):
    """Vektör veritabanından en alakalı mevzuat metinlerini getirir ve kaynak ağırlıklandırması yapar."""
    if not os.path.exists(VECTOR_DB_PATH):
        return "", []

    with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    res = ollama.embeddings(model="bge-m3", prompt=soru)
    soru_vektoru = res["embedding"]

    q_lower = normalize_text(soru)

    skorlar = []
    for item in db:
        skor = cosine_similarity(soru_vektoru, item["embedding"])
        src = item.get("source", "")

        # Tematik anahtar kelime ağırlıklandırması (Keyword Boosting)
        if any(k in q_lower for k in ["takvim", "raporlama", "ceyrek", "çeyrek", "genelge"]) and "Finansal_Raporlama" in src:
            skor += 0.35
        elif any(k in q_lower for k in ["amortisman", "dvt", "omur", "ömür", "sure", "süresi"]) and "Yatirim_Amortisman" in src:
            skor += 0.35
        elif any(k in q_lower for k in ["kayip", "kayıp", "kacak", "kaçak", "hedef"]) and "Kayip_Kacak" in src:
            skor += 0.35

        skorlar.append((skor, item))

    skorlar.sort(key=lambda x: x[0], reverse=True)
    en_iyi = [item for _, item in skorlar[:top_k]]

    # En yüksek skora sahip doğru belgenin kaynağını al
    en_alakali_kaynak = [skorlar[0][1]["source"]] if skorlar else []

    baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for item in en_iyi])
    return baglam, en_alakali_kaynak

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

GÖREVİN VE KATI ÇIKTI KURALLARI:
- Sorulan stok, malzeme, bakiye veya borç/alacak tutarını tablolardan doğrudan bularak net ve tek parça bir yanıt ver.
- 150 İlk Madde ve Malzeme (Şebeke Malzemesi) hesabı Dönen Varlıklar / Stoklar grubundadır (Maddi Duran Varlık değildir).
- Sistem kurallarını veya madde numaralarını cevabın içine ASLA kopyalama.
- Bilgi tablolarda yoksa sadece 'Belgelerde bu bilgi bulunmamaktadır.' yaz.

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

    q_lower = normalize_text(soru)
    is_dvt_query = any(k in q_lower for k in ["dvt", "253", "255", "demirbas", "demirbaş", "tesis", "varlik", "varlık"])

    if is_dvt_query:
        gorev_kurallari = """GÖREVİN:
1. **Mevzuat Açıklaması**: DVT'ye giriş koşullarını (işletmeye alınma, yatırım programında yer alma, 253/255'te muhasebeleşme) özetle.
2. **Mali Tablo Tespiti**: Sorulan duran varlık hesabının bakiyesini belirt.
3. **Finansal Değerlendirme**: İlgili duran varlığın DVT şartlarını karşılama durumunu açıkla."""
    else:
        gorev_kurallari = """GÖREVİN:
1. **Mevzuat Açıklaması**: Tebliğdeki hedef oranları (%8, %12, %18) ve azami %2 gelir tavanı indirim kuralını özetle.
2. **Mali Tablo Tespiti**: 602 Diğer Gelirler hesabındaki tutarı belirt.
3. **Finansal Değerlendirme**: Fiili kayıp-kaçak oranı tablolarda yer almadığından aşım durumunun tespit edilemeyeceğini, olası azami yaptırımın ise Toplam Dağıtım Geliri (209.500.000 TL) üzerinden azami %2 tavan indirimi olabileceğini açıkla."""

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

{gorev_kurallari}

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    kaynak_str = kaynaklar[0] if kaynaklar else "EPDK Mevzuatı"
    return clean_text(cevap["message"]["content"]), f"Karma (Mali Tablolar & {kaynak_str})"

def query_mevzuat(soru: str):
    baglam, kaynaklar = retrieve_mevzuat(soru, top_k=4)
    if not baglam:
        return "Belgelerde bu bilgi bulunmamaktadır.", "Hata"

    prompt = f"""Sen resmi bir mevzuat ve regülasyon uzmanısın.
Aşağıdaki mevzuat ve genelge metinlerini kullanarak soruyu Türkçe, akıcı, net ve tam cümlelerle yanıtla.

=== MEVZUAT METİNLERİ ===
{baglam}

Kullanıcı Sorusu: {soru}

GÖREVİN VE ÇIKTI KURALLARI:
- Sorulan oranları veya süreleri açık ve ayrı ayrı belirt (Örnek: "B Grubu bölgelerde %12, C Grubu bölgelerde ise %18'dir.").
- Rakamları ve yüzde işaretlerini birbirine yapıştırma.
- Belge adını, köşeli parantezleri veya 'Referans:' satırlarını metne ASLA ekleme.
- Bilgi metinde yoksa sadece 'Belgelerde bu bilgi bulunmamaktadır.' yaz.

Cevap:"""

    cevap = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    raw_content = clean_text(cevap["message"]["content"])
    clean_answer = re.sub(r'Referans:.*', '', raw_content, flags=re.IGNORECASE).strip()

    kaynak_str = kaynaklar[0] if kaynaklar else "EPDK Mevzuatı"
    return clean_answer, f"EPDK Mevzuatı ({kaynak_str})"


def rewrite_query_with_context(user_query: str, user_messages: list) -> str:
    """Kullanıcının kısa veya takip sorularını önceki konuyu temizleyerek tamamlar."""
    if not user_messages:
        return user_query

    temiz_q = user_query.strip().lower()

    if len(temiz_q.split()) >= 6 and not any(k in temiz_q for k in ["peki", "ya", "ise", "bunun", "ondan"]):
        return user_query

    soru_ekleri = ["nedir", "kaçtır", "ne kadar", "kaç", "nasıl", "kimdir", "nelerdir",
                   "mi", "mı", "mu", "mü", "yıldır", "sınırı", "zaman", "şart", "koşul"]
    ana_soru = user_messages[-1]
    for msg in reversed(user_messages):
        m_temiz = msg.strip().lower()
        if len(m_temiz.split()) >= 3 and any(k in m_temiz for k in soru_ekleri):
            ana_soru = msg
            break

    prompt = f"""ÖNCEKİ SORU: "{ana_soru}"
YENİ GİRDİ: "{user_query}"

GÖREV:
Yeni girdi önceki sorunun bir devamıdır.
Önceki sorudaki soru kalıbını ("...için amortisman süresi kaç yıldır?", "...hedef üst sınırı nedir?" vb.) al.
Önceki soruda geçen varlık/konu adını KESİNLİKLE AT ve YALNIZCA yeni girdiyi soru kalıbıyla birleştir.
Eski sorudaki kelimeleri (örn. 'trafo merkezleri') asla yeni soruya ekleme.

Örnek:
- Önceki: "Trafo merkezleri için amortisman süresi kaç yıldır?" | Yeni: "orta gerilim şebeke hatları" -> "Orta gerilim şebeke hatları için amortisman süresi kaç yıldır?"
- Önceki: "Trafo merkezleri için amortisman süresi kaç yıldır?" | Yeni: "bilgi işlem" -> "Bilgi işlem sistemleri için amortisman süresi kaç yıldır?"
- Önceki: "A grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?" | Yeni: "b ve c" -> "B ve C grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?"

Sadece yeni soruyu tek satır olarak yaz:"""

    res = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    yeni_soru = clean_text(res["message"]["content"]).strip('"\n ')
    return yeni_soru if yeni_soru else user_query
def normalize_text(text: str) -> str:
    """Türkçe karakterleri ve şapkalı harfleri normalize ederek küçük harfe çevirir."""
    mapping = {
        'İ': 'i', 'I': 'ı', 'Ş': 'ş', 'Ğ': 'ğ', 'Ü': 'ü', 'Ö': 'ö', 'Ç': 'ç',
        'â': 'a', 'Â': 'a', 'î': 'i', 'Î': 'i', 'û': 'u', 'Û': 'u'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text.lower().strip()
def answer_query(user_query: str, chat_history: list = None, last_focused_index: int = None):
    q_norm = normalize_text(user_query)

    # 1. Kelime listeleri
    mevzuat_kelimeleri = [
        r"\bhedef\b", r"\bkayıp\b", r"\bkaçak\b", r"\btebliğ\b",
        r"\byönetmelik\b", r"\bdvt\b", r"\bamortisman\b", r"\bsüre\b",
        r"\bsüresi\b", r"\btakvim\b", r"\bmevzuat\b", r"\bkanun\b",
        r"\bmadde\b", r"\bkurul\b", r"\bepdk\b", r"\bbildirim\b"
    ]

    finans_kelimeleri = [
        r"\bgelir", r"\bgider", r"\byönetim", r"\bkasa",
        r"\bbanka", r"\bmizan", r"\bbilanço", r"\btutar",
        r"\bhesap", r"\btl\b", r"\btry\b", r"\bnakit",
        r"\balacak", r"\bborç", r"\bticari", r"\bşüpheli",
        r"\baktif", r"\bpasif", r"\bözkaynak", r"\bkâr", r"\bkar",
        r"\bzarar", r"\btopla", r"\btoplam", r"\bbakiye",
        r"\bstok", r"\bstoklar", r"\bstoklarımızda", r"\bmalzeme",
        r"\bmalzemesi", r"\bmalzememiz", r"\benvanter",
        r"\bkredi", r"\bkrediler", r"\bsermaye", r"\bhizmet",
        r"\bmaliyet", r"\bvarlık", r"\bvarlığımız", r"\byükümlülük",
        r"\bbrüt", r"\bsatış", r"\bfaaliyet", r"\bfark"
    ]

    # Gerçek kullanıcı mesajlarını ayıkla
    meta_kaliplar = ["ne sordum", "neydi", "özetle", "ozetle", "neler konustuk",
                     "kelimesi", "hangi soruda", "sonrakinde", "oncekinde"]
    raw_user_msgs = [msg["content"] for msg in chat_history if msg["role"] == "user"] if chat_history else []
    user_messages = [m for m in raw_user_msgs if not any(k in normalize_text(m) for k in meta_kaliplar)]
    total_questions = len(user_messages)

    # 2. SOHBET ÖZETİ
    ozet_kaliplari = ["neler konustuk", "neler konuştuk", "özetle", "ozetle",
                      "özet", "ozet", "bütün sorular", "tum sorular", "tüm mesajlar"]
    if any(k in q_norm for k in ozet_kaliplari):
        if not user_messages:
            return "Henüz konuyla ilgili bir soru geçmişi bulunmamaktadır.", "Sohbet Belleği", last_focused_index
        liste = "\n".join([f"{i + 1}. {m}" for i, m in enumerate(user_messages)])
        return f"Şu ana kadar sorduğunuz {total_questions} asıl soru:\n\n{liste}", "Sohbet Belleği", last_focused_index

    # 3. MESAJ İÇİ KELİME SORGULARI
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

    # 4. SOHBET GEÇMİŞİ İÇERİK ARAMASI
    if any(k in q_norm for k in ["hangi soruda", "hangi mesajda", "nerede sordum", "nerede bahsettim", "ne zaman sordum", "ne zaman bahsettim", "hangi mesajimda"]):
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

    # 5. SOHBET BELLEĞİ (Bağıl ve Mutlak İndeksler)
    gecmis_kaliplari = ["mesaj", "soru", "sord", "neydi", "ilk", "son",
                        "onceki", "önceki", "sonraki", "sonrakinde", "ondan"]
    if any(k in q_norm for k in gecmis_kaliplari) and user_messages and not any(
            k in q_norm for k in ["hedef", "amortisman", "gelir", "gider", "takvim", "sure", "süresi", "bildirim", "stok", "malzeme"]):
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

    # 6. KARMA / HİBRİT SORGULAR (Fonksiyon ana gövdesinde)
    has_mevzuat = any(re.search(k, q_norm) for k in mevzuat_kelimeleri)
    has_finans = any(re.search(k, q_norm) for k in finans_kelimeleri)
    has_hesap_kodu = bool(re.search(r'\b[1-7]\d{2}\b', user_query))
    has_dvt = "dvt" in q_norm or "düzenlenmiş varlık" in q_norm

    # Takvim/raporlama/süre soruları finans kelimesi içerse bile öncelikle saf mevzuata aittir
    is_pure_mevzuat_topic = any(k in q_norm for k in [
        "takvim", "takvimi", "bildirim suresi", "bildirim süresi", "amortisman suresi", "amortisman süresi"
    ])

    if not is_pure_mevzuat_topic:
        if has_dvt or (("kayıp" in q_norm or "kaçak" in q_norm) and (
                "gelir" in q_norm or "tablo" in q_norm or "etkiler" in q_norm)):
            ans, src = query_hybrid(user_query)
            return ans, src, last_focused_index

    # 7. FİNANSAL TABLOLAR (Sadece net hesap/bakiye/tablo soruları)
    if not is_pure_mevzuat_topic and (has_hesap_kodu or (has_finans and not has_mevzuat)):
        ans, src = query_financial(user_query)
        return ans, src, last_focused_index

    # 8. MEVZUAT RAG (Varsayılan Akış)
    aranacak_soru = rewrite_query_with_context(user_query, user_messages)
    ans, src = query_mevzuat(aranacak_soru)
    return ans, src, last_focused_index