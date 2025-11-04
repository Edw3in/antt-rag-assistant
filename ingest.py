import sys, pathlib
from typing import List

# Docling (carrega múltiplos formatos)
from langchain_docling import DoclingLoader

# Split por tokens para respeitar limites do embedding
from langchain_text_splitters import TokenTextSplitter

# Vector store
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Embeddings (pacote novo)
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "BAAI/bge-m3"
ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg"}

def collect_paths(args: List[str]) -> list[str]:
    files: list[str] = []
    for a in args:
        p = pathlib.Path(a).resolve()
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                    files.append(str(f))
        elif p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            files.append(str(p))
        else:
            print(f"[AVISO] Ignorado (não existe/sem suporte): {p}")
    return sorted(set(files))

if __name__ == "__main__":
    inputs = sys.argv[1:]
    if not inputs:
        print("❌ Você precisa informar arquivos/pastas. Ex.:")
        print("   python ingest.py .\\docs")
        sys.exit(1)

    print("🔄 Iniciando processamento...\n")

    file_paths = collect_paths(inputs)
    if not file_paths:
        print("❌ Nenhum arquivo suportado encontrado.")
        sys.exit(1)

    print("📄 Lendo e convertendo documentos...")

    docs = []
    for fp in file_paths:
        print(f"   → Convertendo: {fp}")
        loader = DoclingLoader(file_path=fp)
        docs.extend(loader.load())

    print(f"✅ {len(docs)} documento(s) carregado(s)\n")

    print("✂️  Dividindo em pedaços (por tokens)...")
    splitter = TokenTextSplitter(chunk_size=350, chunk_overlap=40)  # evita ultrapassar 512 tokens
    splits = splitter.split_documents(docs)
    splits = filter_complex_metadata(splits)
    print(f"✅ {len(splits)} pedaço(s) criado(s)\n")

    print("🧠 Criando embeddings (GPU se disponível)...")
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        # normalização é recomendada no bge-m3
        encode_kwargs={"normalize_embeddings": True},
    )

    print("💾 Salvando no banco vetorial (Chroma)...")
    db = Chroma.from_documents(splits, emb, persist_directory=PERSIST_DIR)
    # Chroma >0.4 persiste automaticamente; não precisa db.persist()
    print(f"\n✅ CONCLUÍDO! Base salva em: {PERSIST_DIR}")
