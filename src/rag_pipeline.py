# ── Imports ───────────────────────────────────────────────────────────
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("Tudo funcionando 🚀")

# ── Carregar dataset ──────────────────────────────────────────────────
# amnesty_qa tem perguntas + contextos reais — ideal para RAG
print("A carregar dataset...")
ds = load_dataset("explodinggradients/amnesty_qa", "english_v1", split="eval")

print(f"✅ {len(ds)} exemplos carregados")
print(f"Colunas: {ds.column_names}")

# ── Extrair contextos ─────────────────────────────────────────────────
# Cada item tem uma lista de contextos — extraímos todos como documentos
documentos = []
for item in ds:
    for ctx in item["contexts"]:
        documentos.append(Document(
            page_content=ctx,
            metadata={"question": item["question"]}
        ))

print(f"✅ {len(documentos)} documentos extraídos")
print(f"\nExemplo de documento:")
print(documentos[0].page_content[:200])

# ── Chunking ──────────────────────────────────────────────────────────
# Divide os documentos em pedaços menores para recuperação mais precisa
# overlap garante que o contexto não se perde entre chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documentos)

print(f"✅ {len(chunks)} chunks criados")
print(f"Média de palavras por chunk: {sum(len(c.page_content.split()) for c in chunks) // len(chunks)}")

# ── Embeddings ────────────────────────────────────────────────────────
# Transforma texto em números para comparar por similaridade
# all-MiniLM-L6-v2 é leve, rápido e funciona bem para inglês

print("A criar embeddings (pode demorar 1-2 min)...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("✅ Modelo de embeddings carregado")

# ── Vector Store ──────────────────────────────────────────────────────
# FAISS armazena os embeddings e permite busca por similaridade

print("A criar vector store...")

vectorstore = FAISS.from_documents(chunks, embeddings)

print("✅ Vector store criado")
print(f"Total de vectores: {vectorstore.index.ntotal}")

# ── Retriever ─────────────────────────────────────────────────────────
# k=3 = recupera os 3 chunks mais relevantes para cada pergunta

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Teste com 3 perguntas reais do dataset

print("TESTE DO RETRIEVER")
print("=" * 50)

perguntas_teste = [item["question"] for item in list(ds)[:3]]

for pergunta in perguntas_teste:
    docs = retriever.invoke(pergunta)
    print(f"\n📌 Pergunta: {pergunta}")
    print(f"Contextos recuperados: {len(docs)}")
    print(f"Preview: {docs[0].page_content[:150]}...")

print("\n✅ Pipeline RAG funcionando!")