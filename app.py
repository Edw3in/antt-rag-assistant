import streamlit as st
from query import qa

st.title("🛣️ ANTT RAG Assistant")
question = st.text_input("Digite sua pergunta:")

if st.button("Enviar") and question:
    st.write("Buscando resposta...")
    result = qa.invoke({"query": question})
    st.markdown(f"**Resposta:** {result['result']}")
    st.markdown("### Fontes:")
    for doc in result["source_documents"]:
        st.write(f"- {doc.metadata.get('source', '')}")
