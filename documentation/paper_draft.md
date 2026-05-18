# Traceability-First Retrieval: Trust-Weighted Fusion for Evidence-Prioritised Clinical RAG

**[Khalid Iqnaibi — to be finalised]**
**[Affiliation(s)]**
**[Correspondence: khalid.iqnaibi@[institution]]**

---

# Traceability-First Retrieval: Mathematical Evaluation and Ablation of Trust-Weighted Fusion in Clinical RAG

## Abstract

Retrieval-Augmented Generation (RAG) models are increasingly deployed in high-stakes domains like clinical decision support. However, standard architectures suffer from the **Semantic Trap**: they rank documents based purely on lexical or semantic text alignment, leaving them vulnerable to prioritizing low-evidence information (e.g., anecdotal case reports, unverified commentary) over high-evidence literature (e.g., systematic reviews, randomized controlled trials) simply due to text verbosity or keyword density. This paper presents **Traceability-First Retrieval**, an architecture that introduces **Trust-Weighted Ranking (TWR)** to fuse structural provenance and objective authority signals directly into the retrieval layer.

Evaluating this framework using the **Clinical Trust Evaluation (CTE-180)** benchmark—a dataset comprising 180 professional-grade clinical queries across Oncology, Neurology, and Dermatology—revealed substantial performance improvements. TWR achieved a significantly higher mean nDCG@3 for evidence-level recovery ($p = 1.69 \times 10^{-17}$, Cliff's $\delta = +0.4580$) and journal-tier prioritization ($p = 1.81 \times 10^{-27}$, Cliff's $\delta = +0.8253$) compared to standard Reciprocal Rank Fusion (RRF). These findings confirm that integrating structured authority metrics directly into retrieval mechanics protects downstream generation from semantic bias without requiring post-hoc filters.

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has transformed how large language models (LLMs) access domain-specific information. By fetching context from an external vector space or document store, RAG reduces hallucinations and grounds responses in factual data. Despite these advances, traditional RAG systems remain exposed to a fundamental structural vulnerability: the **Semantic Trap**.

In professional domains governed by objective evidence hierarchies, semantic proximity does not imply truth or clinical authority. For instance, an anecdotal medical blog post or an uncontrolled case report might closely match a query's phrasing or tone. Standard semantic search metrics (such as cosine similarity over dense embeddings) or lexical algorithms (such as BM25) often score these lower-tier documents higher than concise, highly authoritative systematic reviews or randomized controlled trials (RCTs).

To address this challenge, we introduce **Traceability-First Retrieval**. This architecture treats data provenance and document authority as foundational constraints within the primary retrieval equation rather than as post-hoc filtering layers. By implementing a novel fusion algorithm called **Trust-Weighted Ranking (TWR)**, TFR combines multi-ranker lists with a structural authority vector mapped directly to established professional standards.

---

## 2. Core Mathematical Framework

TFR measures performance differences by applying traditional Reciprocal Rank Fusion (RRF) and Trust-Weighted Ranking (TWR) to identical source documents.

### 2.1 Baseline: Reciprocal Rank Fusion (RRF)

Standard RRF scores documents based on their ordinal position within an arbitrary set of rankers $R$ (such as lexical BM25 and dense FAISS vector embeddings):

$$RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where $r(d)$ is the specific rank assigned to document $d$ within ranker $r$, and $k$ represents a smoothing constant traditionally set to 60. While RRF effectively aggregates disparate ranking strategies, it is inherently blind to metadata quality, source provenance, or structural authority hierarchies.

### 2.2 Proposed: Trust-Weighted Ranking (TWR)

TWR alters the consensus rank by scaling the RRF score with a multi-factor structural authority vector, denoted as $\Gamma(d)$:

$$TWR(d) = RRF(d) \cdot \Gamma(d)$$

The trust scalar $\Gamma(d)$ maps real-world professional credentials to a bounded scalar interval $[0.0, 1.0]$. In a clinical setting, this structural vector evaluates source documentation, peer-review indexing status, and data currency:

$$\Gamma(d) = W_{\text{evidence}}(d) \cdot W_{\text{journal}}(d) \cdot e^{-\lambda \cdot (Y_{\text{current}} - Y_{\text{pub}}(d))}$$

Where:

* $W_{\text{evidence}}(d)$ is a mapping function derived from the **Oxford Centre for Evidence-Based Medicine (OCEBM)** hierarchy (Meta-Analysis / Systematic Review = 1.0; Randomized Controlled Trial = 0.9; Cohort Study = 0.7; Case-Control Study = 0.5; Case Report / Expert Opinion = 0.1).
* $W_{\text{journal}}(d)$ is the normalized **SCImago Journal Rank (SJR)** metric classification, mapping peer-reviewed impact quartiles to specific weights (Q1 = 1.0; Q2 = 0.85; Q3 = 0.7; Q4 = 0.5; Unranked = 0.2).
* $e^{-\lambda \cdot \Delta Y}$ introduces an exponential decay penalty for aging data based on a temporal parameter $\lambda$, ensuring that outdated information is penalized relative to the current calendar year ($Y_{\text{current}}$).

---

## 3. Architectural Design & Methodology

### 3.1 Downstream Impact and Rank-1 Rationalization

The architectural design of TFR prioritizes optimizing the top-ranked slots (Top-1 and Top-3). This design choice is supported by established long-context LLM research.

Empirical studies on position bias demonstrate that generative models exhibit a strong recency and primacy bias when processing in-context information. Notably, the "Lost in the Middle" phenomenon discovered by Liu et al. (2023) shows that LLMs identify and synthesize evidence most effectively when it is placed at the absolute beginning or end of the context window. If authoritative documents are buried down at rank 4 or rank 5, downstream text generation performance degrades rapidly.

Furthermore, LLMs are vulnerable to distraction; adding low-evidence text into the prompt context can induce reasoning errors or hallucinations, even when high-evidence text is also present. Consequently, ensuring that Rank-1 represents an authoritative, high-trust source is vital for maintaining downstream safety.

### 3.2 Dual-Index Pipeline Architecture

The implementation relies on an integrated storage and retrieval pipeline:

1. **Lexical Indexing:** Implemented via a modified `BM25Okapi` framework using regex-based string tokenization to prevent integer truncation errors on numeric medical terms.
2. **Dense Indexing:** Constructed using an `all-MiniLM-L6-v2` SentenceTransformer model to generate 384-dimensional dense vectors, indexed inside a standardized FAISS execution environment.
3. **Database Layer:** Source documents are standardized via an ingestion engine that references an external SCImago global dataset and maps them to an underlying SQLite `documents.db` store.

---

## 4. Experimental Setup & The CTE-180 Benchmark

To evaluate the mathematical generalizability of TWR, we created the **Clinical Trust Evaluation (CTE-180)** benchmark dataset. This suite contains 180 professional clinical queries distributed across three medical domains and four analytical ablation tracks:

* **`evidence_level`:** Evaluates retrieval precision when low-tier documents contain high keyword matching density but lower-level clinical authority.
* **`journal_tier`:** Measures the system's ability to prioritize peer-reviewed literature indexed in top quartiles (Q1/Q2) over unranked clinical text.
* **`multi_factor`:** Evaluates the interaction and compounding penalties of overlapping trust factors (e.g., old, lower-tier documents from unranked journals).
* **`adversarial`:** Tests boundary conditions where high-trust primary documentation is sparse or absent, evaluating the system's mathematical fallback behavior.

### Table 1: Benchmark Distribution Matrix

| Evaluation Track | Oncology | Neurology | Dermatology | Total Queries |
| --- | --- | --- | --- | --- |
| `evidence_level` | 15 | 15 | 15 | **45** |
| `journal_tier` | 15 | 15 | 15 | **45** |
| `multi_factor` | 15 | 15 | 15 | **45** |
| `adversarial` | 15 | 15 | 15 | **45** |
| **Total Pool** | **60** | **60** | **60** | **180** |

---

## 5. Empirical Results & Statistical Evaluation

### 5.1 Macro Performance Analysis

Evaluating the full 180-query benchmark demonstrated that TFR consistently outperformed the standard RRF framework across all test dimensions.

### Table 2: Macro-Ablation Performance Metrics

| Ablation Dimension | Pipeline | Mean Top-1 Evidence Level | Mean Top-1 Journal Tier | Mean nDCG@3 (Ev) | Mean nDCG@3 (Tier) | Mean MRR |
| --- | --- | --- | --- | --- | --- | --- |
| **Evidence Level** | Standard | 2.02 | Q3 | 0.869 | 0.619 | 0.574 |
|  | **TFR** | **1.18** | **Q1** | **0.961** | **0.957** | **0.941** |
| **Journal Tier** | Standard | 1.84 | Q3 | 0.884 | 0.601 | 0.496 |
|  | **TFR** | **1.11** | **Q1** | **0.976** | **0.969** | **0.963** |
| **Multi-Factor** | Standard | 2.04 | Q3 | 0.871 | 0.603 | 0.548 |
|  | **TFR** | **1.16** | **Q1** | **0.968** | **0.956** | **0.948** |
| **Adversarial** | Standard | 2.56 | Q3 | 0.820 | 0.597 | 0.365 |
|  | **TFR** | **1.29** | **Q1** | **0.956** | **0.941** | **0.907** |

### 5.2 Statistical Significance Testing

Because clinical information retrieval distributions often exhibit non-normal distributions, performance deltas were validated using a two-tailed **Wilcoxon Signed-Rank test** with a Holm-Bonferroni correction applied for multiple comparisons. Effect sizes were measured using **Cliff's Delta ($\delta$)**.

* **Evidence Level Recovery (nDCG@3):** The TFR framework demonstrated a statistically significant improvement over baseline standard retrieval ($p = 1.69 \times 10^{-17}$), yielding a solid structural effect size ($\delta = +0.4580$, Medium).
* **Journal Tier Prioritization (nDCG@3):** The systemic promotion of peer-reviewed high-impact literature over unranked documents achieved a highly significant score ($p = 1.81 \times 10^{-27}$, adjusted Holm-Bonferroni $p = 5.437 \times 10^{-27}$), establishing a dominant effect size ($\delta = +0.8253$, Large).
* **Mean Reciprocal Rank (MRR):** TFR consistently surfaced authoritative documentation to Rank-1, significantly outperforming standard RRF across all 180 paired queries ($p = 2.45 \times 10^{-21}$, $\delta = +0.7140$, Large).

---

## 6. Discussion, Limitations, and Future Work

### 6.1 Generalized Mathematical Abstraction

While this study implements TWR using clinical metadata, the underlying architecture is domain-agnostic. The Trust Vector $\Gamma$ can be formalized mathematically as a product of an arbitrary set of configurable structural weights $W$:

$$TWR(d) = RRF(d) \cdot \prod_{i=1}^{n} W_i(d)$$

Using this formulation, $W$ can represent any domain-specific hierarchical truth structure or institutional ranking system.

### Table 3: Cross-Domain Structural Mapping Examples

| Target Domain | Weight Factor ($W_1$) | Weight Factor ($W_2$) |
| --- | --- | --- |
| **Jurisprudence (Law)** | **Court Hierarchy**<br>

<br>(Supreme Court > Circuit Court) | **Precedent Age**<br>

<br>(Active/Affirmed > Stale/Overruled) |
| **Corporate Finance** | **Audit Verification Level**<br>

<br>(Big 4 Audited > Unaudited) | **Regulatory Filing Type**<br>

<br>(Form 10-K > Press Release) |

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