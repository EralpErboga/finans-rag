import streamlit as st
from rag_engine import answer_query

st.set_page_config(
    page_title="Finans-RAG Asistanı",
    page_icon="⚡",
    layout="wide"
)

# Yan Menü (Sidebar)
with st.sidebar:
    st.header("⚙️ Sistem Bilgisi")
    st.info("**Çalışma Modu:** Kurum İçi (On-Premise)\n\n**LLM:** Qwen 2.5 (7B)\n\n**Embedding:** BGE-M3\n\n**Veri:** EPDK Mevzuatı & 2024 Mizan Tablosu")
    st.markdown("---")
    st.markdown("### 💡 Örnek Sorular:")
    st.markdown("- *A grubu bölgelerde kayıp-kaçak hedef üst sınırı nedir?*")
    st.markdown("- *Trafo merkezleri için amortisman süresi kaç yıldır?*")
    st.markdown("- *2024 yılı toplam genel yönetim gideri ne kadar?*")
    st.markdown("- *Kasa ve bankalardaki toplam nakit varlığımız nedir?*")

st.title("⚡ Finans-RAG: EPDK Mevzuatı & Mizan Asistanı")
st.caption("Veri güvenliği garantili, kurum içi yapay zeka soru-cevap asistanı")

# Mesaj Geçmişini Başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski Mesajları Listele
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "source" in msg:
            st.caption(f"📌 **Kaynak:** {msg['source']}")

# Yeni Soru Girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Yanıt Üret
    with st.chat_message("assistant"):
        with st.spinner("İlgili kaynak taranıyor ve yanıt üretiliyor..."):
            try:
                answer, source = answer_query(prompt)
                st.markdown(answer)
                st.caption(f"📌 **Kaynak:** {source}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source": source
                })
            except Exception as e:
                st.error(f"Hata oluştu: {e}")