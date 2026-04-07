# RAG Hallucination Detector

Pipeline for detecting and classifying hallucinations in RAG systems,
combining automatic metrics, LLM-as-a-Judge, and red teaming.

**Stack:** Python · LangChain · FAISS · RAGAS · Groq API  
**Status:** ✅ Complete

---

## Results at a Glance

| Evaluation | Result |
|------------|--------|
| RAGAS faithfulness | 0.117 |
| RAGAS context precision | 0.400 |
| Human vs. LLM agreement | 100% (κ = 1.0) |
| Red team SAFE rate | 4/10 |
| Red team UNSAFE rate | 0/10 |

---

## Overview

This project answers one question: **how do you know if a RAG
pipeline is hallucinating?**

Three complementary approaches are used:

1. **RAGAS** — automatic metrics measuring faithfulness, relevancy,
   recall and precision
2. **LLM-as-a-Judge** — classifies each response as CORRECT,
   FACTUAL_ERROR, FABRICATION or CONTRADICTION
3. **Red Teaming** — structured safety evaluation across 5 categories

---

## Project Structure
rag-hallucination-detector/
├── src/
│   └── rag_pipeline.py          # RAG pipeline (FAISS + LangChain)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_pipeline.ipynb
│   ├── 03_ragas_evaluation.ipynb
│   ├── 04_llm_judge.ipynb
│   ├── 05_human_vs_llm.ipynb
│   └── 06_red_teaming.ipynb
├── results/
│   ├── ragas_baseline.csv
│   ├── llm_judge_results.csv
│   ├── human_vs_llm_agreement.csv
│   └── red_team_results.csv
└── docs/
├── methodology.md
└── red_team_cases.md

---

## Key Findings

**1. Format consistency is a real production problem**
LLM-as-a-Judge returned inconsistent output formats — a finding
with direct implications for pipeline reliability.

**2. Low RAGAS scores are diagnostic, not failure**
Scores reflect a minimal pipeline under context constraints.
They provide a baseline for improvement, not a final verdict.

**3. Bilingual signal matters**
Developed alongside a bilingual EN/PT annotation testset (κ = 0.92)
— cultural context affects hallucination detection in ways
monolingual pipelines miss.

---

## Getting Started
```bash
git clone https://github.com/renataennes/rag-hallucination-detector
cd rag-hallucination-detector
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🔗 Related Projects

- [Project 2 — Bilingual LLM Annotation Test Set](../project2-annotation-testset/)
- [Project 3 — LLM Eval Dashboard](../project3-eval-dashboard/)

---

*Built as part of an AI Model Evaluation portfolio. Author: Renata Araújo — [LinkedIn](https://www.linkedin.com/in/renata-araujo-en/)*

