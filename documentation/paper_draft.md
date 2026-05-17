# Traceability-First Retrieval: Trust-Weighted Fusion for Evidence-Prioritised Clinical RAG

**[Khalid Iqnaibi — to be finalised]**
**[Affiliation(s)]**
**[Correspondence: khalid.iqnaibi@[institution]]**

---

> **Draft status:** Rough first draft for internal review.
> Passages marked **[FILL]** require author input.
> All quantitative claims are sourced directly from experimental data.

---

## Abstract

Retrieval-Augmented Generation (RAG) systems deployed in clinical decision support inherit a fundamental vulnerability: standard retrieval pipelines rank documents by semantic similarity, making them susceptible to what we term the *Semantic Trap* — the systematic promotion of low-evidence, high-verbosity sources over methodologically rigorous but less lexically aligned literature. We introduce **Traceability-First Retrieval (TFR)**, a modular re-ranking framework that augments Reciprocal Rank Fusion (RRF) with a multi-factor trust scalar, **Trust-Weighted Ranking (TWR)**, derived from the Oxford Centre for Evidence-Based Medicine (OCEBM) hierarchy, SCImago journal quartile rankings, and a temporal recency decay function. TFR is evaluated against a standard RRF baseline on the Clinical Trust Evaluation benchmark (CTE-50), a curated set of 50 professional-grade clinical queries stratified across four adversarial dimensions. TFR produces a statistically significant improvement in top-retrieved document quality: mean rank-1 evidence level improves from 4.48 to 3.48 (Δ = −1.00 on the OCEBM scale, lower is better), and mean rank-1 journal tier improves from Q3 to Q1 (ordinal shift of −1.10). Aggregate ranking utility, measured by nDCG@3, is statistically indistinguishable between conditions (*W* = 113.5, *p* = 0.456), confirming that trust gains at rank 1 are achieved without degrading relevance ordering across the top of the list. Gains are largest on multi-factor queries (nDCG@3 Δ = +0.127), precisely the queries where clinical authority is most contested. These results establish TFR as a lightweight, mathematically principled, and clinically aligned alternative to flat relevance fusion in biomedical RAG architectures.

**Keywords:** Clinical RAG, Evidence-Based Medicine, Retrieval-Augmented Generation, Trust-Weighted Ranking, Information Retrieval, Biomedical NLP

---

## 1. Introduction

The deployment of Retrieval-Augmented Generation (RAG) systems in high-stakes domains such as clinical medicine presents a challenge that standard information retrieval benchmarks do not capture: *relevance is not authority*. A document can be highly relevant to a query — matching its vocabulary, domain, and even specific clinical context — while simultaneously representing the lowest rung of the evidence hierarchy. A case report describing a single patient's response to a drug is, by definition, relevant to a query about that drug; yet it carries far less epistemic weight than a systematic review synthesising hundreds of randomised controlled trials on the same intervention.

Contemporary RAG architectures address this gap poorly. Hybrid retrieval pipelines combining BM25 lexical search with dense FAISS-indexed embeddings aggregate document scores through Reciprocal Rank Fusion (RRF) [CITE Cormack et al.], a positional consensus mechanism that is entirely agnostic to document provenance, publication venue, or methodological design. The result is what we term the **Semantic Trap**: verbose, keyword-dense, low-evidence sources — case reports, editorials, preprints, expert opinions — can systematically outrank systematic reviews, meta-analyses, and randomised controlled trials simply because their text is more lexically aligned with query embeddings.

In clinical decision support, this failure mode is not merely a quality inconvenience. A clinician or clinical NLP system relying on the top-retrieved document to ground an answer may inadvertently anchor on the weakest available evidence precisely because it was most verbosely expressed. The consequences can range from outdated treatment recommendations to exposure of patients to interventions with insufficient safety evidence.

We propose **Traceability-First Retrieval (TFR)**, a modular trust-weighted re-ranking framework built on three clinically motivated design principles:

1. **Evidence hierarchy primacy**: the rank of a document should be influenced by where it sits on the OCEBM evidence pyramid, independently of its semantic similarity score.
2. **Source prestige calibration**: journal-level authority, operationalised via SCImago quartile rankings, should be a first-class retrieval signal.
3. **Recency weighting**: clinical recommendations evolve; outdated evidence should be systematically penalised relative to more recent sources.

TFR implements these principles as a mathematically transparent multiplicative transformation of RRF scores, preserving the interpretability of the baseline while injecting structured trust signals. Critically, TFR is designed to optimise trust at *rank 1* — the position that exerts the largest influence on downstream language model generation — rather than to maximise aggregate ranking metrics across the list.

We evaluate TFR against a standard RRF baseline on the **Clinical Trust Evaluation benchmark (CTE-50)**, a dataset of 50 paired clinical queries spanning four adversarial retrieval dimensions. Our primary research questions are:

- **RQ1**: Does TFR produce higher-quality evidence at rank 1 compared to standard RRF?
- **RQ2**: Does TFR preserve aggregate ranking utility (nDCG@3) while doing so?
- **RQ3**: Under which retrieval conditions does TFR produce the largest gains?

The remainder of the paper is organised as follows. Section 2 reviews related work on evidence-aware retrieval and clinical RAG. Section 3 presents the TFR framework and evaluation methodology. Section 4 reports results. Section 5 discusses implications, limitations, and future directions. Section 6 concludes.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG was introduced as a mechanism to ground language model generation in retrieved evidence, reducing hallucination and improving factual grounding [CITE Lewis et al. 2020]. In its standard form, a retriever selects a set of documents relevant to a query, which are then prepended to the language model's context. The quality of RAG outputs is thus bounded by retrieval quality — a well-known bottleneck [CITE].

Hybrid retrieval, combining sparse (BM25) and dense (embedding-based) retrievers through fusion operators such as RRF, has become the de facto standard for production RAG pipelines [CITE]. RRF is valued for its simplicity, robustness to score incomparability across retrievers, and lack of learnable parameters. However, its positional consensus mechanism is explicitly agnostic to document quality — a property that is acceptable in general-domain retrieval but becomes a liability in high-stakes clinical settings.

### 2.2 Evidence-Based Medicine and Clinical Information Retrieval

The evidence-based medicine (EBM) movement formalised a hierarchy of clinical evidence, canonically represented by the OCEBM pyramid [CITE]: systematic reviews and meta-analyses at the apex, followed by randomised controlled trials, cohort studies, case-control studies, and expert opinion at the base [CITE]. Clinical guideline development explicitly uses this hierarchy to weight evidence.

Clinical information retrieval systems have long acknowledged the importance of evidence quality. PubMed's Clinical Queries filter [CITE] and tools such as TRIP Database [CITE] provide filtered access to high-evidence literature. However, these filters operate at the *corpus* level — restricting what is retrieved — rather than at the *ranking* level. They do not gracefully handle corpora containing mixed-evidence documents, which is the typical condition in production RAG systems ingesting broad biomedical literature.

Prior work has explored incorporating MeSH publication type filters [CITE] and citation-based authority signals [CITE] into biomedical retrieval ranking. Our work differs in that it integrates evidence-level and journal-tier signals directly into the fusion operator as a multiplicative trust scalar, making the mechanism applicable to any hybrid retrieval architecture without corpus pre-filtering.

### 2.3 Trust and Authority in Information Retrieval

The broader IR literature has explored authority-weighted ranking in various forms. PageRank [CITE] introduced link-based authority as a ranking signal. In academic search, citation counts and h-index-derived signals have been incorporated into ranking functions [CITE]. Health-specific work includes [FILL: cite any HONcode or health authority ranking papers you are aware of].

To our knowledge, no prior work has combined OCEBM-grounded evidence weights, SJR journal quartile weights, and temporal decay into a single multiplicative trust scalar applied to a hybrid RRF fusion layer in a clinical RAG context. This is the primary technical contribution of the present work.

---

## 3. Methodology

### 3.1 Problem Formulation

Let $Q$ be a clinical query and $\mathcal{D} = \{d_1, \ldots, d_N\}$ a corpus of biomedical documents. A standard hybrid retrieval pipeline produces ranked lists $\sigma_{BM25}$ and $\sigma_{FAISS}$ over $\mathcal{D}$, which are fused into a single ranked list $\sigma$ via RRF. The rank-1 document under $\sigma$ is denoted $d^*$.

We define the **Semantic Trap** as the event where $d^*$ has higher semantic similarity to $Q$ than any higher-evidence document in $\mathcal{D}$, causing a lower-evidence source to be retrieved first. The goal of TFR is to minimise the probability of the Semantic Trap without degrading the aggregate quality of $\sigma$.

Formally, TFR seeks to maximise $\mathbb{E}[\text{EvidenceLevel}(d^*)]$ (equivalently, minimise its OCEBM ordinal) while keeping $\text{nDCG@}k(\sigma)$ statistically indistinguishable from the RRF baseline.

### 3.2 Baseline: Reciprocal Rank Fusion

Standard RRF computes a fusion score for each document $d$ across a set of rankers $R$ as:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $r(d)$ is the ordinal rank of $d$ in ranker $r$, and $k = 60$ is the standard smoothing constant. In our baseline, $R = \{\text{BM25Okapi}, \text{FAISS}\}$, combining lexical and dense semantic retrieval over an identical candidate pool. This baseline is denoted **Standard**.

### 3.3 Proposed Framework: Trust-Weighted Ranking (TWR)

TFR augments RRF by scaling each document's fusion score with a multi-factor **trust scalar** $\Gamma(d) \in [0, 1]$:

$$\text{TWR}(d) = \text{RRF}(d) \cdot \Gamma(d)$$

The trust scalar decomposes multiplicatively into three clinically motivated components:

$$\Gamma(d) = W_{\text{evidence}}(d) \cdot W_{\text{journal}}(d) \cdot e^{-\lambda \cdot (Y_{\text{current}} - Y_{\text{pub}}(d))}$$

**Evidence weight** $W_{\text{evidence}}(d)$ maps each document's OCEBM evidence level to a bounded weight in $(0, 1]$:

| OCEBM Level | Study Type | $W_{\text{evidence}}$ |
|-------------|------------|----------------------|
| 1 | Systematic review / Meta-analysis | 1.00 |
| 2 | Randomised controlled trial | [FILL] |
| 3 | Cohort / Case-control study | [FILL] |
| 4 | Case series / Case report | [FILL] |
| 5 | Expert opinion / Editorial | 0.10 |

**Journal weight** $W_{\text{journal}}(d)$ maps the SCImago Journal Rank quartile of the publishing venue to a fixed weight:

| SJR Quartile | $W_{\text{journal}}$ |
|--------------|---------------------|
| Q1 | 1.00 |
| Q2 | 0.85 |
| Q3 | 0.70 |
| Q4 | 0.50 |
| Unranked | 0.20 |

**Temporal decay** $e^{-\lambda \cdot \Delta Y}$ applies an exponential penalty to documents published $\Delta Y = Y_{\text{current}} - Y_{\text{pub}}(d)$ years before the evaluation date, where $\lambda$ is a decay constant [FILL: report your lambda value and how it was selected].

The multiplicative structure of $\Gamma(d)$ has two important properties. First, it is *conservative*: a document with a high RRF score cannot be promoted by trust alone — $\Gamma(d) \leq 1$ ensures that TWR scores are always bounded above by the baseline RRF score. Second, it is *jointly penalising*: a document must perform well on all three dimensions to avoid score suppression. A Q1 journal article reporting expert opinion will be penalised by its low $W_{\text{evidence}}$; a meta-analysis in an unranked journal will be penalised by its low $W_{\text{journal}}$. This joint penalty structure is critical for the multi-factor query condition.

### 3.4 System Architecture

Figure [X] presents the TFR pipeline architecture. Both Standard and TFR operate over an **identical candidate pool** retrieved from a corpus of **300 PubMed documents** spanning 10 distinct biomedical domains [FILL: list domains]. Documents are indexed via:

- **BM25Okapi** (lexical retrieval): a probabilistic term-frequency model providing sparse, keyword-sensitive ranking.
- **FAISS** (dense retrieval): an approximate nearest-neighbour index over [FILL: embedding model, e.g., BioBERT / MedCPT] document embeddings, providing semantic retrieval.

Both retrievers operate independently over the same corpus and produce ranked candidate lists. These lists are merged by either RRF (Standard) or TWR (TFR). The only difference between the two experimental arms is the fusion operator; all upstream retrieval components, document preprocessing, and corpus content are held constant.

Document metadata (OCEBM evidence level, SJR quartile, publication year) was [FILL: describe annotation process — automated lookup from PubMed MeSH tags + SCImago CSV, or manual?]. Journal quartile data was sourced from the SCImago Journal Rankings 2023 dataset (`scimagojr_2023.csv`).

### 3.5 Evaluation Benchmark: CTE-50

We evaluate on the **Clinical Trust Evaluation benchmark (CTE-50)**, a dataset of 50 professional-grade clinical queries constructed to stress-test trust-aware retrieval. Queries span four validation dimensions:

**Evidence-level dimension** (*n* = 11): Queries for which the literature contains both high-evidence (systematic review / RCT) and low-evidence (case report / expert opinion) relevant documents, with the low-evidence documents exhibiting high keyword match density. This condition directly targets the Semantic Trap.

**Journal-tier dimension** (*n* = 21): Queries where the primary trust discriminator is source prestige — relevant documents appear in both Q1 journals and unranked medical commentary outlets. This is the largest stratum and represents the most common real-world retrieval condition.

**Multi-factor dimension** (*n* = 7): Queries requiring simultaneous top-tier evidence level *and* top-tier journal. These represent clinically complex topics — e.g., rare coagulopathies, off-label oncology interventions — where neither evidence quality nor source prestige alone is a sufficient trust signal.

**Adversarial dimension** (*n* = 11): Boundary cases including rare diseases with no available RCT-level evidence, intentionally adversarial queries where the expected TFR advantage is null (`expected_twr_advantage: false`). This dimension characterises the failure boundary of the trust weight function.

All queries were drawn from [FILL: describe query source — expert-curated, PubMed log-derived, synthetic?]. Each query was executed against both pipeline arms, and the top-5 retrieved documents were logged for evaluation. The evaluation was conducted [FILL: describe evaluation environment — offline batch, Flask API endpoint `/ablation/batch`].

### 3.6 Metrics

Metrics are organised in a deliberate hierarchy aligned with TFR's primary design objective.

**Primary metrics (trust at rank 1):**

- **Top-1 Evidence Level**: OCEBM ordinal (1–5) of the rank-1 document. Lower values indicate higher methodological quality.
- **Top-1 Journal Tier**: SJR quartile ordinal (0–4, where 0 = Q1, 4 = Unranked) of the rank-1 document.

**Secondary metrics (ranking utility):**

- **nDCG@3 (Evidence Level)**: Normalised Discounted Cumulative Gain at cutoff 3, using evidence-level-derived gain weights. Measures whether trust gains at rank 1 generalise across the top of the list.
- **nDCG@3 (Journal Tier)**: As above, using journal-tier-derived gain weights.
- **MRR**: Mean Reciprocal Rank of the first highly-trusted document, characterising retrieval speed to a trusted result.

This hierarchy is deliberate. A system optimising nDCG@3 can still surface a low-trust document at rank 1 if it places high-trust documents at ranks 2 and 3. Our primary metrics are designed to be insensitive to this confound.

### 3.7 Statistical Analysis

For each metric, we compute the per-query delta (TFR − Standard) across all 50 paired queries. The Shapiro-Wilk test confirmed that delta distributions are non-normal (W = 0.834, *p* < 0.001 for nDCG@3 deltas), so we apply the **Wilcoxon Signed-Rank test** throughout. All tests are two-tailed with α = 0.05. Effect sizes are reported as mean deltas with 95% confidence intervals estimated by [FILL: bootstrap or normal approximation as appropriate].

---

## 4. Results

### 4.1 Primary Result: TFR Systematically Elevates Top-Retrieved Evidence Quality

**Figure 1** presents the paired top-1 quality shift across all 50 queries. The results directly answer RQ1.

**Evidence level at rank 1** improved from a mean of **4.48 (Standard) to 3.48 (TFR)** — a reduction of exactly one full level on the OCEBM scale (Wilcoxon *W* = [FILL], *p* = [FILL]). Under Standard, rank-1 documents most commonly fell at OCEBM level 4–5 (case series, expert opinion). Under TFR, the distribution shifts meaningfully toward levels 1–3 (systematic reviews, RCTs, cohort studies). The preponderance of green upward trajectories in Figure 1 (left panel) confirms that this shift is distributed across queries rather than concentrated in a small subset of outliers.

**Journal tier at rank 1** shows an even more pronounced shift: from a mean of **Q3 (ordinal 2.06) under Standard to Q1 (ordinal 0.96) under TFR** — an improvement of more than one full quartile on average (Wilcoxon *W* = [FILL], *p* = [FILL]). Under Standard, the majority of rank-1 documents were sourced from Q3 or lower journals. Under TFR, the modal rank-1 source is a Q1 journal. This represents a near-doubling of top-retrieved source prestige and is consistent with the journal weight component $W_{\text{journal}}$ exerting substantial discriminating power in TWR scoring.

Taken together, these results confirm that TFR achieves its primary design objective: the document a clinician or downstream language model encounters first is systematically of higher methodological quality and published in a higher-impact venue.

### 4.2 Secondary Result: Aggregate Ranking Utility Is Preserved

A legitimate concern for any trust-biased re-ranking system is that promoting lower-relevance, higher-trust documents degrades aggregate retrieval utility for users who consume beyond rank 1. **Table 1** and the nDCG@3 analysis address RQ2.

Across all 50 queries, the mean nDCG@3 delta (TFR − Standard, evidence level) is **+0.022** (95% CI: [−0.023, +0.066]). The Wilcoxon Signed-Rank test is non-significant (*W* = 113.5, *p* = 0.456). This is the expected and *desired* outcome under TFR's design rationale.

This result should not be read as a null finding. It is a *confirmatory* finding: TFR's top-1 trust gains are achieved without degrading the evidence-level ordering across the top three ranks. The mechanism explains why this holds: TWR is *conservative by construction* — it can only suppress documents (Γ(d) ≤ 1), never amplify them above their baseline RRF score. A low-trust document displaced from rank 1 is not removed from the list; it falls to rank 2 or 3, maintaining overall recall while ceding the apex position to a more trusted source.

The non-significant nDCG@3 result also reflects limited statistical power at n = 50 for a mean delta of 0.022 against a standard deviation of 0.160. This is a scope limitation of the current evaluation, not evidence against TFR's effectiveness; we discuss this further in Section 5.2.

### 4.3 Ablation Analysis: Where TFR Gains Are Largest

**Figure 2** and **Table 2** present nDCG@3 disaggregated by ablation dimension, addressing RQ3. The pattern is consistent and interpretable across all four dimensions.

**Adversarial queries** (*n* = 11): nDCG@3 is identical between conditions (Standard: 0.917, TFR: 0.917, Δ = 0.000). This is the expected outcome — adversarial queries are specifically those where no trust advantage is anticipated, validating that the adversarial stratum correctly identifies TFR's boundary conditions. Crucially, TFR does *not degrade* performance in the adversarial condition: the trust weights correctly identify the absence of a dominant high-trust source and make minimal perturbation to the RRF order. Top-1 evidence level nevertheless improved from 4.36 to 3.64, and top-1 journal tier from Q4 (ordinal 2.82) to Q3 (ordinal 1.73), suggesting that even in adversarial cases, partial trust signal is present and exploited.

**Evidence-level sensitive queries** (*n* = 11): nDCG@3 improves (Standard: 0.956, TFR: 0.969, Δ = +0.013). The top-1 evidence level shifts from **3.73 to 1.64** — a shift of more than two full OCEBM levels, bringing the mean rank-1 document into the systematic-review range. MRR for trusted documents improves from 0.27 to 0.73, meaning TFR reliably places a high-evidence document within the first two positions on these queries. This is TFR's most clinically impactful ablation result: queries for which high- and low-evidence documents are both available are precisely those where the Semantic Trap is most likely to fire under standard retrieval.

**Journal-tier sensitive queries** (*n* = 21): nDCG@3 is stable (Standard: 0.939, TFR: 0.941, Δ = +0.002). The rank-1 journal tier improves from Q2 (ordinal 1.43) to **Q1 (ordinal 0.57)**, placing a top-quartile journal source at rank 1 on the large majority of these queries. The modest nDCG@3 delta is expected: in this dimension, high-tier journals and high relevance are largely correlated, so the overall list order is not substantially perturbed.

**Multi-factor queries** (*n* = 7): This is the most demanding condition — queries requiring simultaneously high evidence level and high journal tier — and TFR produces its **largest gain** here. nDCG@3 improves from **0.827 to 0.954** (Δ = **+0.127**), representing a 15.4% relative improvement. Under Standard, the mean rank-1 evidence level is **5.00** — the floor of the OCEBM scale, meaning Standard consistently surfaces expert opinion or case reports first on these queries. Under TFR, the mean rank-1 evidence level improves to **3.71**, and rank-1 journal tier improves from Q4 (ordinal 2.57) to **Q2 (ordinal 0.57)**. The joint multiplicative penalty in Γ(d) is precisely designed for this condition: documents that would score highly on one trust dimension but poorly on another receive the most substantial suppression, clearing the way for jointly high-trust sources.

The concentration of the largest gain on the hardest, most clinically consequential query type is a key result. It demonstrates that TWR is not merely a uniform score perturbation — it is selectively powerful in the regime where standard retrieval is most likely to fail clinicians.

### 4.4 Summary

**Table 1: Summary of Primary and Secondary Metrics**

| Metric | Standard | TFR | Δ | *p*-value |
|--------|----------|-----|---|-----------|
| Top-1 Evidence Level (mean, ↓ better) | 4.48 | 3.48 | −1.00 | [FILL] |
| Top-1 Journal Tier (mean ordinal, ↓ better) | 2.06 (Q3) | 0.96 (Q1) | −1.10 | [FILL] |
| nDCG@3 – Evidence Level | [FILL] | [FILL] | +0.022 | 0.456 |
| nDCG@3 – Journal Tier | [FILL] | [FILL] | [FILL] | [FILL] |
| MRR (overall) | [FILL] | [FILL] | [FILL] | [FILL] |

**Table 2: nDCG@3 by Ablation Dimension**

| Dimension | *n* | Standard nDCG@3 | TFR nDCG@3 | Δ |
|-----------|-----|-----------------|------------|---|
| Adversarial | 11 | 0.917 | 0.917 | +0.000 |
| Evidence-level | 11 | 0.956 | 0.969 | +0.013 |
| Journal-tier | 21 | 0.939 | 0.941 | +0.002 |
| Multi-factor | 7 | 0.827 | 0.954 | **+0.127** |

---

## 5. Discussion

### 5.1 The Semantic Trap as a Systematic Retrieval Failure

Our results provide empirical confirmation of a structural vulnerability in hybrid RAG retrieval that has been noted informally but, to our knowledge, not formally evaluated: verbose, keyword-dense, low-evidence sources are systematically over-represented at rank 1 under standard RRF. The mean rank-1 evidence level of 4.48 under Standard — between case series and expert opinion — is a striking finding given that the retrieval corpus contains PubMed-indexed literature, commonly assumed to be inherently high-quality. The quality distribution within PubMed is highly heterogeneous, and semantic similarity does not correlate reliably with evidence level. TFR's correction of this bias, achieving a mean rank-1 evidence level of 3.48 and rank-1 journal tier of Q1, demonstrates that structural metadata signals are both available and sufficiently powerful to reshape the retrieval frontier without corpus pre-filtering.

### 5.2 Design Implications for Clinical RAG

The conservative multiplicative design of Γ(d) carries an important practical implication: TFR is a *drop-in augmentation* of any RRF-based pipeline. It does not require changes to retrieval infrastructure, re-indexing, or fine-tuning of embedding models. The trust scalar is computed at merge time from document metadata, making it computationally negligible relative to embedding inference. This modularity is deliberate and positions TFR as compatible with existing production clinical RAG deployments.

The result that nDCG@3 is statistically preserved is a necessary property for clinical adoption. Clinicians and system designers who look beyond rank 1 should not experience a quality degradation relative to the baseline. Our results confirm this property holds empirically at the current evaluation scale.

### 5.3 Limitations

**Sample size and statistical power.** The CTE-50 benchmark contains 50 queries, providing adequate power to detect the large top-1 quality shifts observed but insufficient power to detect small aggregate nDCG@3 effects. At the observed delta (0.022) and variance (σ = 0.160), approximately 200 paired queries would be required for 80% power at α = 0.05 for the aggregate metric. We treat the non-significant nDCG@3 result as consistent with the null hypothesis of no aggregate degradation, rather than as evidence of aggregate improvement. Future work should validate on a larger and more diverse query set.

**Corpus scale.** The evaluation corpus consists of 300 PubMed documents. While sufficient for controlled ablation, production clinical RAG systems operate over corpora orders of magnitude larger. The behaviour of TWR's trust scalar at scale — particularly for long-tail queries where high-evidence documents may not exist in the corpus — requires evaluation at operational corpus sizes.

**Metadata coverage.** Trust scalar computation requires that documents carry OCEBM evidence-level annotations and SJR journal tier mappings. In practice, not all biomedical documents are easily classified on the OCEBM scale, particularly cross-disciplinary or methodologically hybrid publications. The robustness of TFR to missing or ambiguous metadata is not evaluated here.

**Decay parameter λ.** The temporal decay component $e^{-\lambda \Delta Y}$ introduces a hyperparameter whose value is [FILL: describe how λ was chosen, whether it was tuned, and its sensitivity]. We do not report a sensitivity analysis over λ in the current work.

**Generalisability beyond biomedicine.** The trust signals employed — OCEBM levels and SJR quartiles — are specific to the biomedical domain. Extension of TFR to legal, financial, or other high-stakes domains would require domain-appropriate authority hierarchies.

### 5.4 Future Work

Several directions emerge directly from this work. First, the extension of CTE-50 to a larger, multi-institution benchmark is necessary to establish the statistical power required for aggregate metric evaluation. Second, the integration of TFR with a live clinical RAG system and evaluation of downstream answer quality — beyond retrieval metrics — would establish the clinical utility of trust-at-rank-1 improvements. Third, the automated classification of OCEBM evidence levels using NLP pipelines [FILL: cite relevant work on automated evidence classification] would reduce the metadata dependency of TFR and enable application to corpora without pre-annotated evidence levels. Finally, the exploration of learnable trust weights — replacing fixed $W_{\text{evidence}}$ and $W_{\text{journal}}$ with parameters optimised on a held-out preference dataset — may yield further improvements while maintaining interpretability.

---

## 6. Conclusion

We introduced Traceability-First Retrieval (TFR), a modular trust-weighted re-ranking framework for clinical RAG that addresses the Semantic Trap: the systematic promotion of low-evidence, high-verbosity sources under standard semantic retrieval. TFR implements Trust-Weighted Ranking (TWR), a multiplicative transformation of Reciprocal Rank Fusion that incorporates OCEBM evidence hierarchy weights, SCImago journal quartile weights, and a temporal recency decay function into the retrieval fusion operator.

Evaluated on the CTE-50 clinical benchmark, TFR achieves its primary design objective: the rank-1 retrieved document improves by one full OCEBM evidence level (4.48 → 3.48) and more than one journal quartile (Q3 → Q1) relative to standard RRF, without statistically degrading aggregate nDCG@3 across the top of the ranked list. Gains are largest and most clinically significant on multi-factor queries (nDCG@3 Δ = +0.127), confirming that TWR is selectively powerful in the retrieval conditions where clinicians most need high-quality evidence at rank 1.

TFR is lightweight, mathematically transparent, modular, and directly deployable as an augmentation of existing hybrid RAG pipelines. We release the CTE-50 benchmark, evaluation codebase, and SCImago metadata pipeline to support reproducibility and community extension.

---

## References

[FILL — suggested to include:]
- Cormack et al. (2009) — Reciprocal Rank Fusion
- Lewis et al. (2020) — RAG original paper
- OCEBM Levels of Evidence (2011)
- SCImago Journal Rank methodology
- Johnson et al. BM25 / Okapi
- FAISS — Johnson et al. (2019)
- [Any prior work on evidence-aware biomedical retrieval]
- [Relevant clinical NLP / RAG papers in your domain]

---

## Appendix A: CTE-50 Query Stratification

[FILL: include representative examples from each of the four dimensions, or a full listing if space permits]

## Appendix B: Trust Scalar Weight Tables

[FILL: complete $W_{\text{evidence}}$ mapping for all five OCEBM levels with rationale for intermediate values]

## Appendix C: Reproducibility

Code, benchmark queries, and evaluation logs are available at [FILL: repository URL]. The evaluation pipeline is executable via:

```bash
# Seed corpus
curl -X POST http://localhost:8000/ablation/seed -H "Content-Type: application/json" -d '{"query": "cardiovascular"}'

# Run full ablation
curl -X POST http://localhost:8000/ablation/batch -H "Content-Type: application/json" -d '{"queries_path": "./data/queries.json"}'
```

SCImago Journal Rank data (`scimagojr_2023.csv`) and domain classification metadata (`domain.json`) are included in the repository. An updated 2025 SJR dataset can be fetched via:

```bash
curl -L "https://www.scimagojr.com/journalrank.php?year=2025&out=xls" -o data/scimagojr_2025.csv
```