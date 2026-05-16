import re
from unittest import result
import numpy as np
import faiss
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from datetime import datetime
from utils.llm.llm_client import LLMClient
import json

from infra.clinical_document import ClinicalDocument
from utils.pipeline.audit import PipelineAudit

audit = PipelineAudit("logs/pipeline_audit_log.csv")


class TFRPipeline:
    def __init__(
            self, 
            corpus: List[ClinicalDocument],
            k_doc:int=5
        ):
        self.corpus = corpus
        self.k_doc = k_doc
        self.k_rrf = 60
        self.current_year = datetime.now().year
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 

        self._build_bm25_index()
        self._build_faiss_index()

    def _build_bm25_index(self):
        """Fixes the critical integer tokenization bug using regex string tokens."""
        tokenized_corpus = [re.findall(r"\b\w+\b", doc.text.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_faiss_index(self):
        """Builds a FAISS index. (Note: TurboQuant wrapper would be applied here in prod)."""
        texts = [doc.text for doc in self.corpus]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        # Using IndexFlatIP (Inner Product / Cosine Similarity)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings) # Normalize for Cosine Similarity
        self.faiss_index.add(embeddings)

    def hybrid_retrieval(self, query: str, top_n: int = 10):
        # 1. Sparse Retrieval (BM25)
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1][:top_n]
        
        # 2. Dense Retrieval (FAISS)
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        faiss_scores, faiss_indices = self.faiss_index.search(query_emb, top_n)
        faiss_ranking = faiss_indices[0]

        return bm25_ranking, faiss_ranking

    # Trust-Weighted Ranking (TWR)
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

    def provenance_enrichment(self, final_indices: List[int]) -> List[Dict[str, Any]]:
        results = []
        for rank, idx in enumerate(final_indices):
            doc = self.corpus[idx]
            results.append({
                "tfr_rank": rank + 1,
                "text": doc.text,
                "provenance": {
                    "chunk_id": doc.chunk_id,
                    "source": doc.source,
                    "journal_tier": doc.journal_tier,
                    "evidence_level": doc.evidence_level, # OCEBM
                    "publication_year": doc.publication_year,
                    "domain": doc.domain
                }
            })
        return results

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        start_time = datetime.now()
        
        # 1. Hybrid Retrieval
        bm25_ranks, faiss_ranks = self.hybrid_retrieval(query, top_n=10)
        
        # 2. TWR Fusion
        fused_indices = self.trust_weighted_rrf(bm25_ranks, faiss_ranks)
        
        # 3. Enrichment
        result = self.provenance_enrichment(fused_indices[:self.k_doc])
        
        latency = (datetime.now() - start_time).total_seconds()

        audit.log_event(
            query=query, 
            sparse_results=bm25_ranks, 
            dense_results=faiss_ranks,  
            rrf_twr_results=fused_indices,
            results_count=len(result),
            top_pmid=result[0]["provenance"].get("chunk_id","N/A") if result else "N/A",
            latency=latency,
            pipeline="TFR"
        )
        
        return  result



if __name__ == "__main__":
    # Mock Medical Dataset
    dummy_corpus = [
        ClinicalDocument(
            chunk_id="chunk_001",
            text="Aspirin reduces the risk of cardiovascular events in patients with a history of heart disease.",
            source="New England Journal of Medicine", journal_tier="Q1",
            evidence_level=1, publication_year=2023, domain="cardiology" # High Trust (RCT, New)
        ),
        ClinicalDocument(
            chunk_id="chunk_002",
            text="Aspirin is used for pain relief and reducing fever.",
            source="Medical Blog Post", journal_tier="Unranked",
            evidence_level=5, publication_year=2015, domain="general" # Low Trust (Expert Opinion, Old)
        ),
        ClinicalDocument(
            chunk_id="chunk_003",
            text="Statins are the primary treatment for lowering cholesterol and preventing cardiovascular issues.",
            source="Lancet", journal_tier="Q1",
            evidence_level=2, publication_year=2021, domain="cardiology" # Med Trust
        )
    ]

    # Initialize Architecture
    tfr = TFRPipeline(dummy_corpus)
    
    # Run a Query
    query = "What is the cardiovascular benefit of Aspirin?"
    print(f"Query: '{query}'")
    
    results = tfr.retrieve(query)
    
    # Output the results
    print("Final Output")
    for res in results:
        print(f"Rank {res['tfr_rank']}:")
        print(f"  Text: {res['text']}")
        print(f"  Provenance: {res['provenance']}\n")