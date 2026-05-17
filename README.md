# Traceability-First Retrieval (TFR): Mathematical Evaluation and Ablation of Trust-Weighted Fusion in Clinical RAG

Traditional Retrieval-Augmented Generation (RAG) architectures rely heavily on semantic "closeness" (vector distance), exposing them to the **Semantic Trap**: low-evidence sources (e.g., case reports, expert opinions) can rank higher than high-evidence sources (e.g., systematic reviews, RCTs) simply due to verbose text alignment or keyword density.

TFR resolves this by introducing **Trust-Weighted Ranking (TWR)**, a modular mathematical transformation layer that maps objective clinical authority signals directly onto the retrieval fusion operator.

---

## 1. Core Mathematical Framework

This study evaluates the performance delta between traditional **Reciprocal Rank Fusion (RRF)** and **Trust-Weighted Ranking (TWR)** over an identical candidate pool.

### Baseline: Reciprocal Rank Fusion (RRF)

Standard RRF computes a score based solely on a document's ordinal position within a set of rankers $R$ (e.g., BM25 and FAISS dense embeddings):

$$RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where $r(d)$ is the rank of document $d$ in ranker $r$, and $k$ is a smoothing constant (default: $60$). RRF is blind to document provenance, structural metadata, or truth hierarchies.

### Proposed: Trust-Weighted Ranking (TWR)

TWR optimizes the fusion score by scaling positional consensus with a multi-factor structural authority vector $\Gamma(d)$:

$$TWR(d) = RRF(d) \cdot \Gamma(d)$$

The trust scalar $\Gamma(d)$ maps real-world clinical credentials to a bounded scalar interval $[0.0, 1.0]$:

$$\Gamma(d) = W_{\text{evidence}}(d) \cdot W_{\text{journal}}(d) \cdot e^{-\lambda \cdot (Y_{\text{current}} - Y_{\text{pub}}(d))}$$

Where:

* $W_{\text{evidence}}(d)$: Mapping function for the **Oxford Centre for Evidence-Based Medicine (OCEBM)** hierarchy (e.g., Meta-Analysis = 1.0; Expert Opinion = 0.1).
* $W_{\text{journal}}(d)$: Normalized **SCImago Journal Rank (SJR)** metric grouping (Q1 = 1.0, Q2 = 0.85, Q3 = 0.7, Q4 = 0.5, Unranked = 0.2).
* $e^{-\lambda \cdot \Delta Y}$: Exponential decay function penalizing outdated clinical data given decay parameter $\lambda$.

---

## 2. Experimental Design & Ablation Protocol

To prove the generalizability and mathematical soundness of TWR for high-tier venues (Q1), the evaluation engine isolates the fusion mechanism across a curated corpus of 300 specialized PubMed documents spanning 10 distinct biomedical domains.

```
                              ┌──────────────────┐
                              │  Raw User Query  │
                              └────────┬─────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              ┌───────────────────┐        ┌───────────────────┐
              │ Lexical Retrieval │        │  Dense Retrieval  │
              │    (BM25Okapi)    │        │   (FAISS Index)   │
              └─────────┬─────────┘        └─────────┬─────────┘
                        │                            │
                        └─────────────┬──────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Shared Candidate Pool│
                          └───────────┬──────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
            ┌───────────────────────┐   ┌───────────────────────┐
            │   Control Arm (RRF)   │   │ Experimental Arm (TWR)│
            │ Flat Rank Reciprocal  │   │  Multi-Factor Trust   │
            └───────────────────────┘   └───────────────────────┘

```

### The CTE-50 Benchmark Dataset

Evaluation runs utilize the **Clinical Trust Evaluation (CTE-50)** benchmark suite (`queries.json`). This dataset features 50 professional-grade clinical queries stratified across four distinct validation dimensions:

1. `evidence_level`: Verifies the system's ability to prioritize high-level clinical evidence when low-tier documents exhibit high keyword match density.
2. `journal_tier`: Validates the impact of indexing peer-reviewed literature (Q1/Q2) above unranked medical commentary.
3. `multi_factor`: Tests the intersection and compound penalty behavior of multiple overlapping metadata tags.
4. `adversarial`: Specifically targets boundary/corner cases (e.g., rare diseases lacking RCTs) to chart the exact limits of the trust weight bounds (`expected_twr_advantage: false`).

---

## 3. Directory Structure

```pattern
├── app.py                      # Flask API Gateway & Batch Evaluation Routes
├── audit.py                    # CSV-based PipelineAudit Engine (Logging Architecture)
├── retrieval.py                # TFR Architecture Pipeline (TWR + Metadata Processing)
├── standard_pipeline.py        # Baseline Production RAG Pipeline (Standard RRF)
├── queries.json                # CTE-50 Benchmarking Suite (Stratified Queries)
├── requirements.txt            # System Dependencies Matrix
└── data/
    ├── domain.json             # Core Domain Classification Metadata
    └── scimagojr_2023.csv      # SCImago Journal Rank Metric Mapping Data

```

---

## 4. Reproducibility Guide

### Prerequisites
*   Python 3.11

Set up your environmental variables in a local `.env` file within the repository root to look like `.env.example`

```

### Step 1: Data Preparation
``` bash
# Fetch the official 2025 SCImago Journal Rank global collection via cURL
curl -L "https://www.scimagojr.com/journalrank.php?year=2025&out=xls" -o data/scimagojr_2025.csv
```

### Step 2: Requirements Installation

```bash
pip install -r "requirements.txt"
```
### Step 3: Database Ingestion & Seeding

To build a reproducible baseline database, execute seed requests to ingest domain-stratified documents directly into the document database:
```bash
# Seed db for repreduseable ablation studies

# CARDIOVASCULAR
# L1 – Systematic reviews & meta-analyses
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "systematic review[pt] AND cardiovascular diseases[mh]", "max_results": 50}'

curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "randomized controlled trial[pt] AND (antiplatelet OR anticoagulant OR coronary)", "max_results": 50}'

# L2 – Cohort / prospective
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "cohort studies[mh] AND cardiovascular diseases[mh]", "max_results": 50}'

# L3 – Case-control / cross-sectional
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case-control studies[mh] AND cardiovascular diseases[mh]", "max_results": 50}'

# L4-L5 – Case series + expert review
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case reports[pt] AND (thrombosis OR arrhythmia OR heart failure)", "max_results": 50}'


# ONCOLOGY
# L1
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "meta-analysis[pt] AND neoplasms[mh]", "max_results": 50}'

curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "randomized controlled trial[pt] AND (chemotherapy OR immunotherapy OR targeted therapy)", "max_results": 50}'

# L2
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "cohort studies[mh] AND neoplasms[mh]", "max_results": 50}'

# L3
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case-control studies[mh] AND neoplasms[mh]", "max_results": 50}'

# L4-L5
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case reports[pt] AND (lymphoma OR carcinoma OR melanoma)", "max_results": 50}'


# NEUROLOGY
# L1
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "systematic review[pt] AND nervous system diseases[mh]", "max_results": 50}'

curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "randomized controlled trial[pt] AND (stroke OR epilepsy OR parkinson OR multiple sclerosis)", "max_results": 50}'

# L2
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "cohort studies[mh] AND nervous system diseases[mh]", "max_results": 50}'

# L3
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case-control studies[mh] AND nervous system diseases[mh]", "max_results": 50}'

# L4-L5
curl -X POST http://localhost:8000/ablation/seed \
     -H "Content-Type: application/json" \
     -d '{"query": "case reports[pt] AND (neuropathy OR dementia OR migraine OR encephalitis)", "max_results": 50}'

```
### Step 4: Running the Ablation Study
```bash
# Run Batch Ablation Study
curl -X POST http://localhost:8000/ablation/batch \
     -H "Content-Type: application/json" \
     -d '{"queries_path": "./data/queries.json"}'

```

---
## 4. Evaluation Logging & Audit Trail

The audit file `pipeline_audit_log.csv` captures detailed logs for each evaluation query, including the original query, the retrieved document indices from both BM25 and FAISS, the computed TWR scores, and the final ranked outputs for both the TFR and standard RRF pipelines . You can use this log to reconstruct the exact retrieval and fusion behavior for each query.

Every evaluation query produces two records within the log matrix:

* **`pipeline: "TFR"`**: Demonstrates indices generated via the trust-weighted formula.
* **`pipeline: "Standard_RRF"`**: Establishes the traditional flat fusion rank behavior.

Every batch run appends to the same log file to build a comprehensive dataset.

---

## Research & Documentation
*   **Lead Engineer:** khalid Iqnaibi
*   **Methodology:** Hypothesis-driven ablation.




