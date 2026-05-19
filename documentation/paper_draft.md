# Traceability-First Retrieval: Trust-Weighted Fusion for Evidence-Prioritised Clinical RAG

**[Khalid Iqnaibi — to be finalised]**
**[Affiliation(s)]**
**[Correspondence: khalidiqnaibi@gmail.com]**

---
## Abstract

Retrieval-Augmented Generation (RAG) models are increasingly utilized to support complex decision-making in high-stakes professional domains. However, standard architectures remain vulnerable to the **Semantic Trap**: the tendency to rank documents based purely on lexical or semantic textual alignment. This vulnerabilities allows verbose, low-evidence text (e.g., non-peer-reviewed commentaries or anecdotal case reports) to bypass retrieval layers and displace concise, high-evidence sources (e.g., systematic reviews and randomized controlled trials) due to keyword density or embedding proximity.

To address this limitation, we present **Traceability-First Retrieval (TFR)**, an architecture that treats document authority and structural provenance as primary ranking metrics rather than post-hoc evaluation filters. We introduce **Trust-Weighted Ranking (TWR)**, a fusion framework that scales position-based consensus rankers with a domain-configurable structural authority vector ($\Gamma$).

To evaluate TFR, we established the **Clinical Trust Evaluation (CTE-180)** benchmark, comprising 180 expert-validated clinical queries across Oncology, Neurology, and Dermatology designed to test semantic boundary conditions.

Our empirical results show that TFR significantly outperforms traditional Reciprocal Rank Fusion (RRF). T06WR achieved a significantly higher mean nDCG@3 for evidence-level recovery ($p = 1.69 \times 10^{-17}$, Cliff's $\delta = +0.4580$) and journal-tier prioritization ($p = 1.81 \times 10^{-27}$, Cliff's $\delta = +0.8253$). Mean Reciprocal Rank (MRR) increased substantially, ensuring high-trust sources consistently captured the primary ranking slot ($p = 2.45 \times 10^{-21}$, Cliff's $\delta = +0.7140$).

Finally, we provide a generalized mathematical framework proving that TWR can be mapped to any domain governed by strict, hierarchical source authority, such as jurisprudence or corporate finance.

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has significantly enhanced the utility of Large Language Models (LLMs) by grounding text generation in external knowledge bases. By decoupling static parametric memory from dynamic document repositories, RAG systems reduce hallucinations, enhance verifiability, and enable real-time data updates.

Despite these practical benefits, the fundamental retrieval mechanisms powering modern RAG architectures remain brittle when applied to specialized professional fields. Traditional pipelines rely on lexical algorithms (e.g., BM25) or dense embedding vectors (e.g., cosine similarity over latent spaces). These methods share a common operational assumption: **topical overlap or semantic similarity correlates with informational utility**.

In high-risk domains like medicine, law, or financial auditing, this assumption can fail. These fields are governed by explicit hierarchical truth models, where a source's structural authority is independent of its phrasing or length.

Traditional retrieval models are vulnerable to the **Semantic Trap**: a situation where an informal or unverified source closely mirrors the conversational tone, verbosity, or vocabulary of a user query, causing it to outrank a highly dense, rigorous, peer-reviewed study.

```
[User Query: Conversational, specific clinical phrasing]
       │
       ├─► [Low-Evidence Source]  (High lexical density, verbose matching) ──► Rank 1 (Standard RAG)
       └─► [High-Evidence Source] (Concise, structured data)              ──► Rank 4 (Burying Bias)

```

Failing to prioritize authoritative documentation can compromise the safety of downstream text generation.

To resolve this issue, we introduce **Traceability-First Retrieval (TFR)**, an architecture that incorporates data provenance into the core retrieval equation. Rather than treating metadata as a secondary filtering field or a post-hoc display token, TFR introduces **Trust-Weighted Ranking (TWR)**. This method adjusts ordinal rank consensus using an objective structural authority vector, protecting downstream language generation from semantic biases without requiring aggressive database pruning.

---

## 2. Core Mathematical Framework

The TFR framework evaluates retrieval safety by measuring performance differences between baseline consensus mechanisms and our proposed trust-weighted fusion layer over a unified candidate store.

### 2.1 Baseline Consensus: Reciprocal Rank Fusion (RRF)

Consider a document store $D$ and a set of independent retrieval rankers $R$. For an incoming query $q$, each ranker $r \in R$ outputs a permutation of $D$, where $r(d)$ represents the unique ordinal position of document $d \in D$. The baseline standard consensus score is calculated via Reciprocal Rank Fusion (RRF):

$$RRF(d \in D) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where $k \in \mathbb{N}^{+}$ is a smoothing parameter (traditionally set to $60$) that mitigates the impact of low-ranking outliers. While RRF effectively merges disparate rankers (such as lexical keyword matching and dense vector distances), it is structurally unable to account for internal document metadata or quality variations.

### 2.2 Proposed Architecture: Trust-Weighted Ranking (TWR)

TWR restructures the consensus rank by applying a domain-specific structural authority vector, denoted as $\Gamma(d)$, directly to the consensus output:

$$TWR(d) = RRF(d) \cdot \Gamma(d)$$

The trust vector $\Gamma(d)$ maps qualitative real-world professional certifications to a bounded scalar interval $[0.0, 1.0]$. To instantiate this framework within clinical informatics, we model $\Gamma(d)$ using three distinct structural criteria: evidence tiers, publication impacts, and information age:

$$\Gamma(d) = W_{\text{evidence}}(d) \cdot W_{\text{journal}}(d) \cdot e^{-\lambda \cdot (Y_{\text{current}} - Y_{\rm pub}(d))}$$

Where:

* $W_{\text{evidence}}(d) \in [0.1, 1.0]$ represents an ordinal mapping function derived from the **Oxford Centre for Evidence-Based Medicine (OCEBM)** hierarchy, formalizing clinical study design quality:
$$\begin{cases}
1.0, & d \in \text{Meta-Analysis / Systematic Review (Level 1)} \
0.9, & d \in \text{Randomized Controlled Trial (Level 2)} \
0.7, & d \in \text{Cohort Study (Level 3)} \
0.5, & d \in \text{Case-Control Study (Level 4)} \
0.1, & d \in \text{Case Report / Expert Opinion / Narrative Blog (Level 5)}
\end{cases}$$
* $W_{\text{journal}}(d) \in [0.2, 1.0]$ represents the normalized **SCImago Journal Rank (SJR)** metric classification, mapping peer-reviewed impact quartiles to specific weights:
$$\begin{cases}
1.0, & d \in \text{Quartile 1 (Q1)} \
0.85, & d \in \text{Quartile 2 (Q2)} \
0.7, & d \in \text{Quartile 3 (Q3)} \
0.5, & d \in \text{Quartile 4 (Q4)} \
0.2, & d \in \text{Unranked / Non-indexed Medium}
\end{cases}$$
* $e^{-\lambda \cdot \Delta Y}$ introduces a continuous exponential decay penalty for aging information, where $Y_{\text{current}}$ denotes the execution year, $Y_{\rm pub}(d)$ denotes the document's publication year, and $\lambda \in \mathbb{R}^{+}$ represents a domain-specific volatility constant (set to $0.05$ for clinical medical rollouts).

---

## 3. Architectural Design Rationale

### 3.1 Downstream Synthesis and Rank-1 Rationalization

The primary structural objective of TFR is to optimize the top retrieval slots, specifically focusing on Rank-1 and Rank-3 recovery. This design choice is supported by long-context language model attention dynamics.

Recent work on position bias identifies structural challenges in how generative networks process in-context information. Crucially, the "Lost in the Middle" phenomenon (Liu et al., 2023) demonstrates that LLMs demonstrate high retrieval accuracy when relevant informational contexts reside at the absolute boundaries of the prompt context window.

```
                       In-Context Attention Curve
             High                                    High
              │  \                                  /  │
              │   \                                /   │
  Attention   │    \                              /    │
  Intensity   │     \                            /     │
              │      \                          /      │
             Low      \────────────────────────/      Low
              └───────┴────────────────────────┴───────┘
               Rank 1 (Top)     Middle Context   Rank N (Bottom)

```

When high-evidence documentation is pushed to middle ranks (e.g., Rank 4 or Rank 5) by verbose, low-trust content, generative performance degrades significantly.

Furthermore, downstream generators are vulnerable to context distraction. Incorporating lower-tier information into an evaluation prompt can propagate logical errors or hallucinations, even when higher-evidence documents are present elsewhere in the context window. Ensuring that Rank-1 represents an authoritative, high-trust source is therefore critical for downstream safety.

### 3.2 Dual-Index Pipeline Architecture

The implementation relies on an integrated storage and retrieval pipeline:

1. **Lexical Indexing:** Configured via a customized `BM25Okapi` distribution. It uses regex-based string tokenization (`r'\w+'`) to prevent token-splitting bugs on fractional medical terms, decimals, and alphanumeric clinical designations.
2. **Dense Indexing:** Constructed using an `all-MiniLM-L6-v2` SentenceTransformer pipeline to map textual chunks into a uniform 384-dimensional dense vector space. This space is queried using a standardized non-metric FAISS database.
3. **Relational Meta-Mapping:** Implemented as a fast SQLite engine mapping source document hashes directly to external SJR journal indices and OCEBM categories inside a structured database (`documents.db`).

---

## 4. Experimental Setup & The CTE-180 Benchmark

To evaluate TWR's ability to resist semantic bias, we developed the **Clinical Trust Evaluation (CTE-180)** benchmark. This evaluation framework contains 180 expert-formulated clinical queries distributed evenly across three operational medical categories (Oncology, Neurology, and Dermatology) and stratified across four analytical ablation dimensions:

* **`evidence_level`:** Measures rank-order stability when low-tier documents contain high keyword matching density but lower-level clinical authority.
* **`journal_tier`:** Evaluates the system's ability to prioritize peer-reviewed literature indexed in top quartiles (Q1/Q2) over unranked clinical text.
* **`multi_factor`:** Evaluates the compounding penalties of overlapping trust factors (e.g., old, lower-tier documents from unranked journals).
* **`adversarial`:** Tests boundary conditions where high-trust primary documentation is sparse or absent, evaluating the system's mathematical fallback behavior.

### Table 1: Stratified Benchmark Query Distribution Matrix

| Evaluation Track / Ablation Focus | Neoplasms (Oncology) | Nervous System (Neurology) | Skin Diseases (Dermatology) | Track Total |
| --- | --- | --- | --- | --- |
| `evidence_level` | 15 | 15 | 15 | **45** |
| `journal_tier` | 15 | 15 | 15 | **45** |
| `multi_factor` | 15 | 15 | 15 | **45** |
| `adversarial` | 15 | 15 | 15 | **45** |
| **Total Query Pool** | **60** | **60** | **60** | **180** |

---

## 5. Empirical Results & Statistical Evaluation

### 5.1 Macro-Performance Ablation Analysis

Evaluating the full 180-query benchmark demonstrated that TFR consistently outperformed standard RRF across all metrics.

### Table 2: Complete Performance Summary Across Ablation Tracks

| Track | Strategy | Mean Top-1 Evidence Level ($\downarrow$)* | Mean Top-1 Journal Tier (Ordinal) | Mean nDCG@3 (Evidence) | Mean nDCG@3 (Tier) | Mean MRR |
| --- | --- | --- | --- | --- | --- | --- |
| **Evidence Level** | Standard | 2.022 | 1.733 (Q3) | 0.8694 | 0.6199 | 0.5737 |
|  | **TFR** | **1.178** | **0.133 (Q1)** | **0.9613** | **0.9575** | **0.9407** |
| **Journal Tier** | Standard | 1.844 | 1.778 (Q3) | 0.8844 | 0.6014 | 0.4963 |
|  | **TFR** | **1.111** | **0.067 (Q1)** | **0.9765** | **0.9691** | **0.9630** |
| **Multi-Factor** | Standard | 2.044 | 1.778 (Q3) | 0.8711 | 0.6033 | 0.5481 |
|  | **TFR** | **1.156** | **0.111 (Q1)** | **0.9681** | **0.9566** | **0.9481** |
| **Adversarial** | Standard | 2.555 | 2.222 (Q3) | 0.8200 | 0.5974 | 0.3652 |
|  | **TFR** | **1.289** | **0.200 (Q1)** | **0.9562** | **0.9418** | **0.9074** |

**Note: Lower scores for Mean Top-1 Evidence Level are superior, tracking closer to absolute Level-1 Systematic Reviews.*

### 5.2 Statistical Significance Testing

Because clinical information retrieval distributions often exhibit non-normal distributions, performance deltas were validated using a two-tailed **Wilcoxon Signed-Rank test** with a Holm-Bonferroni correction applied for multiple comparisons. Effect sizes were measured using **Cliff's Delta ($\delta$)**.

* **Evidence Level Recovery Metrics (nDCG@3):** The TFR framework demonstrated a statistically significant improvement over baseline standard retrieval ($p = 1.69 \times 10^{-17}$), yielding a solid structural effect size ($\delta = +0.4580$, Medium).
* **Journal Tier Prioritization Metrics (nDCG@3):** The promotion of peer-reviewed high-impact literature over unranked documents achieved high significance ($p = 1.812 \times 10^{-27}$, adjusted Holm-Bonferroni $p = 5.437 \times 10^{-27}$), establishing a dominant effect size ($\delta = +0.8253$, Large).
* **Mean Reciprocal Rank (MRR):** TFR consistently surfaced authoritative documentation to Rank-1, significantly outperforming standard RRF across all 180 paired queries ($p = 2.454 \times 10^{-21}$, $\delta = +0.7140$, Large).

---

## 6. Discussion, Limitations, and Future Work

### 6.1 Generalized Mathematical Abstraction

While this study implements TWR using clinical metadata, the underlying architecture is domain-agnostic. The Trust Vector $\Gamma$ can be formalized mathematically as a product of an arbitrary set of configurable structural weights $W$:

$$TWR(d) = RRF(d) \cdot \prod_{i=1}^{n} W_i(d)$$

Using this formulation, $W$ can represent any domain-specific hierarchical truth structure or institutional ranking system.

```
                  ┌─────────────────────────────────────────┐
                  │    Generalized Framework: TWR Vector    │
                  └────────────────────┬────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  Clinical Medicine               Jurisprudence               Corporate Finance
  ───► OCEBM Rank (W1)            ───► Court Level (W1)       ───► Audit Level (W1)
  ───► SCImago SJR (W2)           ───► Precedent Age (W2)     ───► Filing Type (W2)

```

### Table 3: Cross-Domain Structural Mapping Examples

| Target Domain | Weight Factor ($W_1$) | Weight Factor ($W_2$) |
| --- | --- | --- |
| **Jurisprudence (Law)** | **Court Hierarchy**<br>

<br>(Supreme Court > Circuit Court > District) | **Precedent Status**<br>

<br>(Active/Affirmed > Stale/Overruled) |
| **Corporate Finance** | **Audit Verification Level**<br>

<br>(Big 4 Audited > Mid-Tier > Unaudited) | **Regulatory Filing Type**<br>

<br>(Form 10-K > Form 10-Q > Press Release) |

> "To empirically validate the TWR architecture, this study instantiates the generalized framework within the highly rigorous domain of Clinical Medicine, defining the Trust Vector ($\Gamma$) using OCEBM Evidence Levels and SCImago Journal Ranks."

### 6.2 Limitations and Future Work

A primary limitation of the current TWR formulation is its reliance on statically defined, rule-based structural coefficients. While these weights provide predictable, auditable performance, they require manual adjustment by domain experts when adapting the system to new fields.

To improve generalizability, future research will explore integrating **Learning to Rank (LTR)** architectures into the trust fusion layer. We hypothesize that these multi-factor structural coefficients can be learned and optimized automatically for each target domain. Transitioning from static multipliers to a learned optimization function should preserve the system's mathematical guardrails while making it easier to scale across diverse professional fields.

---

## 7. Conclusion

The Traceability-First Retrieval architecture addresses the Semantic Trap in RAG systems by embedding objective truth and authority hierarchies directly into the retrieval equations. Evaluating the framework on the CTE-180 benchmark demonstrates that scaling reciprocal ranking with structural authority values yields statistically significant improvements in document ranking quality ($p < 1.0 \times 10^{-16}$). By preventing low-evidence documents from dominating the top ranking slots, TFR provides an auditable and mathematically sound foundation for high-precision deployment in professional decision support systems.

---

## References

* Liu, N. F., Lin, K., Chen, J., Chuang, S., Cho, K., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. *arXiv preprint arXiv:2307.03172*.
* Oxford Centre for Evidence-Based Medicine (OCEBM). (2011). Levels of Evidence. *CEBM*.
* SCImago Journal & Country Rank. (2025). Global Journal Classification Datasets. *SCImago*.

---
## Roles and Responsibilities
**Khalid Iqnaibi**: First author,Conceptualization, Methodology, Software Development, Data Curation, Formal Analysis, Writing - Original Draft, Visualization ,lead engineer, designed the TFR architecture, wrote the code, ran the quantitative statistical ablation (nDCG, MRR, Wilcoxon tests).

**[Additional Authors]**: Co-authors, clinical domain experts, designed the clinical evaluation rubric, performed the blinded qualitative validation of the Top-1 retrieval outputs, and reviewed the manuscript for clinical accuracy and relevance.