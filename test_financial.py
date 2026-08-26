import pandas as pd
import ollama

EXCEL_PATH = "data/mizan_bilanco_dummy_2024.xlsx"


def load_excel_data():
    xls = pd.ExcelFile(EXCEL_PATH)

    # 1. Mizan Sayfası (Başlıklar 5. satırda olduğu için header=4)
    df_mizan = pd.read_excel(xls, sheet_name="Mizan", header=4)
    df_mizan.columns = [str(c).strip() for c in df_mizan.columns]

    # Boş olmayan hesapları filtrele
    df_mizan_clean = df_mizan.dropna(subset=['Hesap Kodu'])

    # 2. Gelir Tablosu ve Bilanço Sayfaları
    df_gelir = pd.read_excel(xls, sheet_name="Gelir Tablosu", header=2).dropna()

    preview = f"""=== MİZAN TABLOSU ===\n{df_mizan_clean.to_string(index=False)}\n\n=== GELİR TABLOSU ===\n{df_gelir.to_string(index=False)}"""
    return preview


def ask_financial(soru: str):
    preview_text = load_excel_data()

    prompt = f"""Sen uzman bir mali müşavir ve finans analistisin.
Aşağıda şirketin 2024 yılı Mizan ve Gelir Tablosu yer almaktadır.

{preview_text}

Kullanıcının Sorusu: {soru}

GÖREVİN:
1. Tablodan ilgili hesap kodunu veya kalemini bul.
2. Sayısal tutarı (Borç / Alacak veya Bakiye) net olarak Türkçe para birimi (TRY) ile belirt.
3. Cevabının altına mutlaka (Hesap Kodu - Hesap Adı) referansını ekle.

Cevap:"""

    print("Mali veriler taranıyor ve LLM yanıtı üretiliyor...")
    res = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]


if __name__ == "__main__":
    test_sorusu = "2024 yılı toplam genel yönetim gideri (632 hesabı) ne kadardır?"
    print(f"Soru: {test_sorusu}\n")
    cevap = ask_financial(test_sorusu)
    print("\n--- FİNANSAL YANIT ---")
    print(cevap)