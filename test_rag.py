import json
import math
import ollama

# Kosinüs Benzerliği (Cosine Similarity) Hesaplama Fonksiyonu
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

# 1. Vektör Veritabanını Yükle
with open("db/mevzuat_vektorleri.json", "r", encoding="utf-8") as f:
    db = json.load(f)

soru = "A grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?"
print(f"Soru: {soru}")

# 2. Soruyu Vektörleştir
print("\n1. Soru vektörleştiriliyor...")
soru_vektoru = ollama.embeddings(model="bge-m3", prompt=soru)["embedding"]

# 3. En Yakın Mevzuat Parçalarını Bul (Vektör Arama)
print("2. İlgili mevzuat maddeleri aranıyor...")
skorlar = []
for item in db:
    skor = cosine_similarity(soru_vektoru, item["embedding"])
    skorlar.append((skor, item))

# En yüksek benzerlik skoruna sahip 2 parçayı seç
skorlar.sort(key=lambda x: x[0], reverse=True)
en_iyi_parcalar = skorlar[:2]

baglam = "\n\n".join([f"[{item['source']}]\n{item['content']}" for _, item in en_iyi_parcalar])

print("\n--- BULUNAN EN ALAKALI MEVZUAT ---")
print(baglam)
print("----------------------------------\n")

# 4. LLM Yanıtı Üret
prompt = f"""Sen şirket finans ve mevzuat ekibine yardımcı olan uzman bir asistansın.
Sana verilen mevzuat bağlamını kullanarak soruyu doğrudan Türkçe olarak cevapla.
Metinde olmayan hiçbir bilgiyi uydurma.
Cevabının altına mutlaka referans aldığın Belge adını ve Madde numarasını ekle.

Bağlam:
{baglam}

Soru: {soru}
Cevap:"""

print("3. LLM (Qwen 2.5) yanıtı üretiyor...")
cevap = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- LLM YANITI ---")
print(cevap["message"]["content"])