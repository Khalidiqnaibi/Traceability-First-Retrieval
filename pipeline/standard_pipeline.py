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

class Pipeline:
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

    def rrf(self, bm25_ranking, faiss_ranking) -> List[int]:
        scores = {}
        
        for rank_lists in [bm25_ranking, faiss_ranking]:
            for rank, doc_idx in enumerate(rank_lists):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1 / (self.k_rrf + rank + 1)
                
        sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        
        return sorted_indices

    def provenance_enrichment(self, final_indices: List[int]) -> List[Dict[str, Any]]:
        results = []
        for rank, idx in enumerate(final_indices):
            doc = self.corpus[idx]
            results.append({
                "rank": rank + 1,
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
        
        # 2. RRF Fusion
        fused_indices = self.rrf(bm25_ranks, faiss_ranks)
        
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
            pipeline="Standard", 
            status="success"
        )

        return result 



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
    p = Pipeline(dummy_corpus)
    
    # Run a Query
    query = "What is the cardiovascular benefit of Aspirin?"
    print(f"Query: '{query}'")
    
    results = p.retrieve(query)
    
    # Output the results
    print("Final Output")
    for res in results:
        print(f"Rank {res['rank']}:")
        print(f"  Text: {res['text']}")
        print(f"  Provenance: {res['provenance']}\n")