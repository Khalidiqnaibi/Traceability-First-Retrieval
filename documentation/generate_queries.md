# Generate 50 queries  to  make an abalation for : 

```
def calculate_trust_score(self, doc: ClinicalDocument) -> float:
        """
        Enhanced Trust function incorporating Journal Tier, Evidence Level, and Recency.
        """
        # Evidence Level (OCEBM): 1 is best (RCT/Meta-analysis), 5 is lowest (Expert Opinion)
        ocebm_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        evidence_score = ocebm_weights.get(doc.evidence_level, 0.1)
        
        # Journal Authority
        tier_weights = {
            "Q1": 1.0,
            "Q2": 0.85,
            "Q3": 0.70,
            "Q4": 0.55,
            "Unranked": 0.40
        }
        journal_multiplier = tier_weights.get(doc.journal_tier, 0.40)
        
        # Recency Decay
        if doc.publication_year <= 0:
            age = 10 
        else:
            age = max(0, self.current_year - doc.publication_year)
        
        recency_multiplier = np.exp(-0.05 * age) 
        
        # Final TWR Calculation
        return evidence_score * journal_multiplier * recency_multiplier

    def trust_weighted_rrf(self, bm25_ranking, faiss_ranking) -> List[tuple]:
        twr_scores = {}
        
        for rank_lists in [bm25_ranking, faiss_ranking]:
            for rank, doc_idx in enumerate(rank_lists):
                doc = self.corpus[doc_idx]
                
                # Calculate TWR Math: trust(source) / (k + rank)
                trust_score = self.calculate_trust_score(doc)
                score_contribution = trust_score / (self.k_rrf + rank + 1)
                
                if doc_idx in twr_scores:
                    twr_scores[doc_idx] += score_contribution
                else:
                    twr_scores[doc_idx] = score_contribution
                    
        # Sort by the new TWR score
        sorted_indices = sorted(twr_scores.keys(), key=lambda x: twr_scores[x], reverse=True)
        return sorted_indices
```

vs 
```
def rrf(self, bm25_ranking, faiss_ranking) -> List[int]:
        scores = {}
        
        for rank_lists in [bm25_ranking, faiss_ranking]:
            for rank, doc_idx in enumerate(rank_lists):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1 / (self.k_rrf + rank + 1)
                
        sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        
        return sorted_indices  
```        
# and then put them into a json file with this structure:
```
{
    "id": 1,
    "query": "What is the cardiovascular benefit of Aspirin?",
    "domain": "Cardiovascular Diseases",
    "ablation_dimension": "evidence_level",
    "expected_twr_advantage": true/false,
    "rationale": "TWR should outperform RRF because it incorporates evidence level, which is crucial for clinical relevance. Documents with higher evidence levels (e.g., RCTs) will receive a higher trust score, thus being ranked higher in TWR compared to RRF which treats all documents equally.",
    "tags": ["Aspirin", "Cardiovascular Diseases", "Evidence Level"]
  }
```