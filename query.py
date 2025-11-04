from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq # <--- NOVA IMPORTAÇÃO
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3-8b-8192" # Modelo Llama 3 da Groq
llm = ChatGroq(model=LLM_MODEL, temperature=0.0)
print("🔄 Carregando sistema RAG...\n")

print("📂 Conectando ao banco de dados...")
emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

# Retriever com MMR
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.5}
)

# Prompt especializado
template = """
Você é um assistente técnico-jurídico especializado em concessões rodoviárias,
regulamentos da ANTT (como RCR-2, RCR-3, RCR-4, RCR-5) e resoluções relacionadas.

Use APENAS as informações presentes nos trechos de "Contexto" para responder.

Regras:
- Responda sempre em português do Brasil, de forma objetiva e técnica.
- Quando possível, cite explicitamente o número da resolução, artigo, parágrafo
  ou cláusula contratual (por exemplo: "art. 50 da RCR-3", "Resolução 6.053/2024").
- Se a pergunta for muito genérica (por exemplo: apenas "seguro"), explique isso
  ao usuário e peça que detalhe melhor (ex: "seguro de risco de engenharia",
  "seguro de responsabilidade civil - RC-OPER", etc.).
- Se o contexto não tiver informação suficiente, diga claramente:
  "Com base apenas nos documentos carregados, não encontrei informação suficiente
  para responder com segurança."

Contexto:
{context}

Pergunta do usuário:
{question}

Resposta (em português, organizada em tópicos quando fizer sentido):
"""

prompt = PromptTemplate.from_template(template)

qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model=LLM_MODEL, temperature=0.1),
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

# MMR para reduzir redundância e ampliar cobertura
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.4}
)

print(f"🤖 Conectando ao modelo {LLM_MODEL} (Ollama)...")
llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\n✅ Sistema pronto! Faça suas perguntas.\n" + "=" * 60)

while True:
    try:
        q = input("\n💬 Sua pergunta (ou ENTER para sair): ").strip()
        if not q:
            print("\n👋 Até logo!")
            break

        print("\n🔍 Buscando resposta...\n")
        result = qa.invoke({"query": q})

        print("=" * 60)
        print("📝 RESPOSTA:\n" + result["result"])

        print("\n" + "=" * 60)
        print("📚 FONTES CONSULTADAS:")
        for i, d in enumerate(result["source_documents"], 1):
            src = d.metadata.get("source", "desconhecido")
            page = d.metadata.get("page", "?")
            print(f"{i}. {src} (página {page})")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n👋 Até logo!")
        break
    except Exception as e:
        print(f"\n❌ Erro: {e}")
