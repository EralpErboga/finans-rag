import os
import glob
import json
import ollama

DATA_PATH = "data/mevzuat"
OUTPUT_FILE = "db/mevzuat_vektorleri.json"


def build_vector_db():
    os.makedirs("db", exist_ok=True)
    print("1. Mevzuat dokümanları okunuyor...")

    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    chunks = []

    for file_path in txt_files:
        dosya_adi = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Metinleri paragraflara göre mantıklı parçalara böl
            paragraflar = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
            for p in paragraflar:
                chunks.append({"source": dosya_adi, "content": p})

    print(f"-> Toplam {len(chunks)} parça oluşturuldu.")
    print("2. Parçalar Ollama (bge-m3) ile vektörleştiriliyor...")

    vektor_veritabani = []
    for i, item in enumerate(chunks):
        print(f"  -> Parça {i + 1}/{len(chunks)} işleniyor...", end="\r")
        res = ollama.embeddings(model="bge-m3", prompt=item["content"])
        vektor_veritabani.append({
            "source": item["source"],
            "content": item["content"],
            "embedding": res["embedding"]
        })
    print()

    # Vektörleri JSON olarak kaydet
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vektor_veritabani, f, ensure_ascii=False, indent=2)

    print(f"-> Başarılı! Vektör veritabanı '{OUTPUT_FILE}' dosyasına kaydedildi.")


if __name__ == "__main__":
    build_vector_db()