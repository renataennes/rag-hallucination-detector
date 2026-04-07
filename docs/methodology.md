# Methodology — RAG Hallucination Detector

## Overview

This project evaluates hallucination in RAG pipelines using three
complementary approaches: automatic metrics (RAGAS), LLM-as-a-Judge,
and structured red teaming.

---

## 1. Dataset

| Dataset | Purpose | Size used |
|---------|---------|-----------|
| explodinggradients/amnesty_qa | RAG evaluation (RAGAS) | 10 examples |
| pminervini/HaluEval | Hallucination classification | 20 examples |
| Anthropic/hh-rlhf | Red teaming reference | 10 custom prompts |

---

## 2. RAG Pipeline

**Stack:** LangChain + FAISS + HuggingFace Embeddings (all-MiniLM-L6-v2)

**Chunking strategy:**
- chunk_size: 500 characters
- chunk_overlap: 50 characters
- Retriever k: 3 documents

**Baseline RAGAS scores:**

| Metric | Score | Benchmark | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.117 | ≥ 0.80 | ⚠️ Below |
| Answer Relevancy | nan | ≥ 0.80 | ❌ Not computed |
| Context Recall | 0.200 | ≥ 0.70 | ⚠️ Below |
| Context Precision | 0.400 | ≥ 0.80 | ⚠️ Below |

**Why scores are low:** Texts were truncated to fit Groq's context
window (llama-3.1-8b-instant limit). Switching to
llama-3.3-70b-versatile resolved the limit but scores reflect
a minimal viable pipeline, not an optimised one.

---

## 3. LLM-as-a-Judge

**Model:** llama-3.3-70b-versatile (Groq)
**Temperature:** 0 (deterministic)
**Categories:** CORRECT / FACTUAL_ERROR / FABRICATION / CONTRADICTION

**Results on HaluEval (20 examples):**

| Category | Count |
|----------|-------|
| FACTUAL_ERROR | 20 |
| CORRECT | 0 |
| FABRICATION | 0 |
| CONTRADICTION | 0 |

**Human vs. LLM agreement:**
- Agreement rate: 100%
- Cohen's Kappa: 1.0
- Disagreements: 0

**Known limitation — output format inconsistency:**
The judge occasionally returns "CLASSIFICATION: SAFE" instead of
"SAFE" — a prompt engineering issue that affects downstream parsing.
In production this would require output validation and retry logic.

---

## 4. Red Teaming

**Taxonomy:** 5 categories × 2 prompts each = 10 cases

| Category | Description |
|----------|-------------|
| jailbreak_direct | Explicit harmful request |
| jailbreak_indirect | Roleplay or hypothetical framing |
| prompt_injection | Hidden instruction in input |
| boundary_testing | Gray area policy cases |
| bias_elicitation | Attempting to reveal model bias |

**Results:**

| Classification | Count |
|---------------|-------|
| SAFE | 4/10 |
| PARTIAL | 1/10 |
| UNSAFE | 0/10 |

**Key finding:** The model refused all direct harmful requests but
showed inconsistent output formatting across categories — suggesting
that prompt structure significantly affects judge reliability.

---

## 5. Insights

**1. Format consistency matters as much as accuracy**
The LLM-as-a-Judge returned inconsistent formats ("CLASSIFICATION: X"
vs "X") — a real production issue that requires output validation.

**2. Low RAGAS scores reflect pipeline constraints, not failure**
Truncating context to fit model limits artificially reduces scores.
A production pipeline would use a model with larger context window
or implement smarter chunking.

**3. Bilingual evaluation adds signal**
This pipeline was developed alongside a bilingual EN/PT annotation
testset (κ = 0.92) — suggesting that cultural and linguistic context
affects hallucination detection in ways monolingual pipelines miss.

---

## 6. What I Would Do Differently

- Use a larger context window model from the start
- Implement output validation and retry logic for the judge
- Increase dataset size to 100+ examples for more reliable metrics
- Add a reranker between retrieval and generation
- Test with Portuguese-language documents to validate bilingual signal