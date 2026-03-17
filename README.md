> 🔍 End-to-end RAG pipeline with automated hallucination detection 
> using LLM-as-Judge evaluation. Bilingual EN/PT golden dataset. 
> Built as part of a 40-day AI Evaluator portfolio project.

**Stack:** Python · LangChain · FAISS · OpenAI API · RAGAS · DeepEval  
**Status:** 🚧 In progress — Day 6/40

---

## 📌 Overview

This project builds a **Retrieval-Augmented Generation (RAG) pipeline** with automated **hallucination detection**, evaluated against a curated bilingual (English–Portuguese) golden dataset.

The goal is to measure how faithfully an LLM answers questions based *only* on retrieved context — and flag responses that contain fabricated or unsupported information.

---

## 🎯 Objectives

- Build a minimal RAG pipeline using LangChain + FAISS + OpenAI
- Create a **golden dataset** of 60+ verified Q&A pairs (EN + PT)
- Implement an LLM-as-judge evaluator to score each response
- Report hallucination rates, faithfulness scores, and failure patterns
- Visualize results in an interactive dashboard (Streamlit)

---

## 🗂️ Project Structure

```
rag-hallucination-detector/
│
├── data/
│   ├── documents/          # Source documents (PDFs, TXTs)
│   ├── golden_dataset_en.json   # Ground truth Q&A pairs (English)
│   └── golden_dataset_pt.json   # Ground truth Q&A pairs (Portuguese)
│
├── src/
│   ├── ingestion.py        # Document loading and chunking
│   ├── retriever.py        # FAISS vector store setup
│   ├── rag_pipeline.py     # Full RAG chain (retrieve + generate)
│   ├── evaluator.py        # LLM-as-judge hallucination scorer
│   └── metrics.py          # Faithfulness, precision, recall metrics
│
├── notebooks/
│   └── 01_exploratory.ipynb    # EDA on golden dataset & outputs
│
├── dashboard/
│   └── app.py              # Streamlit dashboard for results
│
├── results/
│   └── eval_results.json   # Saved evaluation outputs
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Evaluation Rubric

Each LLM response is labeled across 3 dimensions:

| Label | Description |
|---|---|
| `faithful` | Response is fully supported by retrieved context |
| `partially_grounded` | Response uses context but adds unsupported claims |
| `hallucinated` | Response contains facts not present in context |

**Scoring dimensions (0–1):**
- **Faithfulness** — Is every claim grounded in the retrieved documents?
- **Context Precision** — Did retrieval return relevant chunks?
- **Answer Relevance** — Does the answer address the question?

---

## 📊 Key Metrics Tracked

- Overall Hallucination Rate (%)
- Faithfulness Score (avg per model)
- Failure breakdown by language (EN vs PT)
- Failure breakdown by question type (factual / reasoning / ambiguous)

---

## 🔧 Tech Stack

| Tool | Role |
|---|---|
| `LangChain` | RAG pipeline orchestration |
| `FAISS` | Local vector store |
| `OpenAI API` | Embedding + generation |
| `RAGAS` | Automated RAG evaluation framework |
| `DeepEval` | LLM evaluation assertions |
| `Streamlit` | Results dashboard |
| `Pandas` | Dataset management |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/renataennes/rag-hallucination-detector
cd rag-hallucination-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 4. Run the full evaluation pipeline
python src/rag_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---

## 📁 Golden Dataset Format

```json
{
  "id": "en_001",
  "language": "en",
  "question": "What is the definition of hallucination in LLMs?",
  "ground_truth": "Hallucination in LLMs refers to the generation of content that is factually incorrect or unsupported by the provided context.",
  "source_document": "llm_fundamentals.pdf",
  "difficulty": "factual",
  "tags": ["hallucination", "LLM basics"]
}
```

---

## 📈 Results Summary *(example)*

| Model | Faithfulness | Hallucination Rate | Language |
|---|---|---|---|
| gpt-4o | 0.87 | 13% | EN |
| gpt-4o | 0.79 | 21% | PT |
| gpt-3.5-turbo | 0.71 | 29% | EN |

> Full results available in `results/eval_results.json`

---

## 🔗 Related Projects

- [Project 2 — Bilingual LLM Annotation Test Set](../project2-annotation-testset/)
- [Project 3 — LLM Eval Dashboard](../project3-eval-dashboard/)

---

*Built as part of an AI Model Evaluation portfolio. Author: Renata Araújo — [LinkedIn](https://www.linkedin.com/in/renata-araujo-en/)*

