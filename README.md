# Beyond Semantic Search: Prioritizing Trust in Clinical Retrieval-Augmented Generation

> Traditional RAG architectures prioritize semantic "closeness" but often ignore the hierarchical nature of clinical authority. This repository introduces **Traceability-First Retrieval (TFR)**, a modular architecture that elevates provenance from a post-hoc annotation to a first-class ranking signal. By integrating the **Oxford Centre for Evidence-Based Medicine (OCEBM)** hierarchy directly into the retrieval math, TFR ensures that high-quality evidence (RCTs, Systematic Reviews) is prioritized over lower-tier anecdotal data.

---

## The TFR Architecture

The **Traceability-First Retrieval (TFR)** framework is a 5-stage pipeline designed for high-precision, resource-constrained clinical environments.

### Stage 1: Self-Querying (Medical Triage)
*   An LLM extracts structured metadata filters (e.g., `{"domain": "cardiology"}`) from raw natural language queries.
*   This step constrains the downstream search space, reducing noise and improving retrieval efficiency.

### Stage 2: Hybrid Retrieval
*   **Dense Search:** Utilizes a **TurboQuant-compressed FAISS** index for semantic retrieval with near-zero re-indexing time.
*   **Sparse Search:** Employs **BM25Okapi** for lexical keyword matching.
*   This parallel approach ensures the system captures both high-level concepts and specific medical terminology.

### Stage 3: Trust-Weighted Ranking (TWR)
*   This stage introduces a novel fusion mechanism: **Trust-Weighted Ranking (TWR)**.
*   Unlike standard Reciprocal Rank Fusion (RRF), TWR uses a trust-scaled variant: $$\sum{\frac{trust(source)}{k + rank}}$$
*   Fixed weights are mapped to the **OCEBM hierarchy**, accounting for journal tier, evidence level (e.g., RCT > Case Study), and recency decay.

### Stage 4: Provenance Enrichment
*   Final chunks are delivered as structured objects containing first-class provenance metadata: `{source, journal_tier, evidence_level, publication_year, domain, chunk_id}`.

---

## Evaluation

The framework includes a built-in benchmark harness to measure the **Cost-Quality Pareto Frontier** across 16 possible pipeline configurations.

*   **Primary Metrics:** Precision, Recall, MRR, and **NDCG@10**.
*   **Efficiency Metrics:** Token usage, latency (ms), and accumulated API cost.
*   **Stratified Analysis:** Results are logged by query type (Factual, Treatment, or Definition) to identify domain-specific strengths and weaknesses.

---

## Quick Start

### Prerequisites
*   Python 3.11
*   download csv from https://www.scimagojr.com/journalrank.php

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Traceability-First-Retrieval.git
cd Traceability First Retrieval

pip install -r "requirements.txt"
python app.py

```

---

## Research & Documentation
This project is part of a **100-day technical sprint** focused on radical authenticity and the scientific method.
*   **Lead Engineer:** khalid Iqnaibi
*   **Methodology:** Hypothesis-driven ablation.
