import ollama

print("1. Qwen 2.5 modeli test ediliyor...")
res = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': 'Merhaba, 2+2 kaç eder? Tek kelimeyle cevap ver.'}])
print("Qwen Yanıtı:", res['message']['content'].strip())

print("\n2. bge-m3 embedding modeli test ediliyor...")
emb = ollama.embeddings(model='bge-m3', prompt='örnek mevzuat metni')
print(f"Embedding boyutu: {len(emb['embedding'])} (Başarılı!)")