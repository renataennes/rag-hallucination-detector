# ============================================================
# PROJECT 1 — RAG Hallucination Detector
# Author: Renata Araújo | AI Model Evaluation Portfolio
# ============================================================
# SETUP:
#   pip install langchain langchain-openai langchain-community
#              faiss-cpu ragas deepeval streamlit pandas python-dotenv
#
# Create a .env file with:
#   OPENAI_API_KEY=
# ============================================================

import os
import json
from dotenv import load_dotenv
from pathlib import Path

# LangChain
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

load_dotenv()

# ─────────────────────────────────────────────
# 1. DOCUMENT INGESTION
# ─────────────────────────────────────────────

def load_documents(docs_path: str = "data/documents"):
    """Load all .txt documents from the documents folder."""
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# 2. VECTOR STORE (FAISS)
# ─────────────────────────────────────────────

def build_vector_store(chunks, save_path: str = "data/faiss_index"):
    """Create FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ Vector store saved to {save_path}")
    return vectorstore


def load_vector_store(index_path: str = "data/faiss_index"):
    """Load an existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"✅ Vector store loaded from {index_path}")
    return vectorstore


# ─────────────────────────────────────────────
# 3. RAG PIPELINE
# ─────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say: "I don't have enough information to answer this."
Do NOT add information that is not in the context.

Context:
{context}

Question: {question}

Answer:
"""

def build_rag_chain(vectorstore, model="gpt-4o-mini"):
    """Build the RAG chain with a strict grounding prompt."""
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


def run_rag(chain, question: str) -> dict:
    """Run a single question through the RAG pipeline."""
    result = chain.invoke({"query": question})
    return {
        "question": question,
        "answer": result["result"],
        "source_chunks": [doc.page_content for doc in result["source_documents"]]
    }


# ─────────────────────────────────────────────
# 4. LLM-AS-JUDGE HALLUCINATION EVALUATOR
# ─────────────────────────────────────────────

JUDGE_PROMPT = """
You are an expert evaluator assessing whether an AI response is grounded in the provided context.

Context:
{context}

Question: {question}

AI Response: {answer}

Task: Evaluate the response and return a JSON with this exact format:
{{
  "label": "faithful" | "partially_grounded" | "hallucinated",
  "faithfulness_score": <float 0.0 to 1.0>,
  "reasoning": "<brief explanation>",
  "unsupported_claims": ["<claim1>", "<claim2>"]
}}

Rules:
- "faithful": every claim in the response is directly supported by the context
- "partially_grounded": most claims are supported, but 1-2 are not in the context
- "hallucinated": the response contains significant fabricated or unsupported information

Return ONLY the JSON, no extra text.
"""

def evaluate_with_judge(question: str, context: str, answer: str, model="gpt-4o") -> dict:
    """Use an LLM judge to evaluate hallucination."""
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = JUDGE_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )
    response = llm.invoke(prompt)
    try:
        evaluation = json.loads(response.content)
    except json.JSONDecodeError:
        evaluation = {
            "label": "error",
            "faithfulness_score": 0.0,
            "reasoning": "Failed to parse judge response",
            "unsupported_claims": []
        }
    return evaluation


# ─────────────────────────────────────────────
# 5. BATCH EVALUATION ON GOLDEN DATASET
# ─────────────────────────────────────────────

def load_golden_dataset(path: str) -> list[dict]:
    """Load golden dataset JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_full_evaluation(golden_dataset_path: str, chain, output_path: str = "results/eval_results.json"):
    """Run evaluation on all Q&A pairs in the golden dataset."""
    dataset = load_golden_dataset(golden_dataset_path)
    results = []

    for i, item in enumerate(dataset):
        print(f"  Evaluating {i+1}/{len(dataset)}: {item['question'][:60]}...")

        # Get RAG answer
        rag_output = run_rag(chain, item["question"])
        context_str = "\n\n".join(rag_output["source_chunks"])

        # Judge evaluation
        eval_result = evaluate_with_judge(
            question=item["question"],
            context=context_str,
            answer=rag_output["answer"]
        )

        results.append({
            "id": item.get("id", f"item_{i}"),
            "language": item.get("language", "unknown"),
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "rag_answer": rag_output["answer"],
            "context_used": context_str,
            "evaluation": eval_result,
            "difficulty": item.get("difficulty", "unknown")
        })

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation complete. Results saved to {output_path}")
    return results


# ─────────────────────────────────────────────
# 6. SUMMARY METRICS
# ─────────────────────────────────────────────

def compute_summary_metrics(results: list[dict]) -> dict:
    """Compute aggregate evaluation metrics."""
    total = len(results)
    labels = [r["evaluation"].get("label") for r in results]
    scores = [r["evaluation"].get("faithfulness_score", 0) for r in results]

    hallucinated = labels.count("hallucinated")
    partial = labels.count("partially_grounded")
    faithful = labels.count("faithful")

    en_results = [r for r in results if r.get("language") == "en"]
    pt_results = [r for r in results if r.get("language") == "pt"]

    summary = {
        "total_evaluated": total,
        "faithful_count": faithful,
        "partially_grounded_count": partial,
        "hallucinated_count": hallucinated,
        "hallucination_rate": round(hallucinated / total, 3) if total > 0 else 0,
        "avg_faithfulness_score": round(sum(scores) / total, 3) if total > 0 else 0,
        "by_language": {
            "en": {
                "total": len(en_results),
                "hallucination_rate": round(
                    sum(1 for r in en_results if r["evaluation"].get("label") == "hallucinated") / len(en_results), 3
                ) if en_results else 0
            },
            "pt": {
                "total": len(pt_results),
                "hallucination_rate": round(
                    sum(1 for r in pt_results if r["evaluation"].get("label") == "hallucinated") / len(pt_results), 3
                ) if pt_results else 0
            }
        }
    }

    print("\n📊 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"  Total evaluated:      {summary['total_evaluated']}")
    print(f"  ✅ Faithful:          {faithful} ({faithful/total:.0%})")
    print(f"  ⚠️  Partially grounded: {partial} ({partial/total:.0%})")
    print(f"  ❌ Hallucinated:      {hallucinated} ({hallucinated/total:.0%})")
    print(f"  Avg Faithfulness:    {summary['avg_faithfulness_score']}")
    print(f"  EN hallucination:    {summary['by_language']['en']['hallucination_rate']:.0%}")
    print(f"  PT hallucination:    {summary['by_language']['pt']['hallucination_rate']:.0%}")

    return summary


# ─────────────────────────────────────────────
# 7. MAIN — FULL PIPELINE RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 RAG Hallucination Detector — Starting pipeline\n")

    # Step 1: Ingest documents
    docs = load_documents("data/documents")
    chunks = chunk_documents(docs)

    # Step 2: Build or load vector store
    index_path = "data/faiss_index"
    if Path(index_path).exists():
        vectorstore = load_vector_store(index_path)
    else:
        vectorstore = build_vector_store(chunks, index_path)

    # Step 3: Build RAG chain
    chain = build_rag_chain(vectorstore)

    # Step 4: Run evaluation on golden dataset (EN)
    results_en = run_full_evaluation(
        golden_dataset_path="data/golden_dataset_en.json",
        chain=chain,
        output_path="results/eval_results_en.json"
    )

    # Step 5: Run evaluation on golden dataset (PT)
    results_pt = run_full_evaluation(
        golden_dataset_path="data/golden_dataset_pt.json",
        chain=chain,
        output_path="results/eval_results_pt.json"
    )

    # Step 6: Summary metrics
    all_results = results_en + results_pt
    summary = compute_summary_metrics(all_results)

    print("\n✅ Done! Run: streamlit run dashboard/app.py")