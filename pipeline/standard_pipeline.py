import re
import numpy as np
import faiss
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from datetime import datetime
from utils.llm.llm_client import LLMClient
import json

from infra.clinical_document import ClinicalDocument


class Pipeline:
    def __init__(
            self, 
            corpus: List[ClinicalDocument],
            api_key:str,
            k_doc:int=5,
            model:str="inclusionai/ring-2.6-1t:free",
            domain_data_path:str="./data/domain.json"
        ):
        self.corpus = corpus
        self.k_doc = k_doc
        self.k_rrf = 60
        self.current_year = datetime.now().year
        
        self.LLM = LLMClient(api_key=api_key,model=model)
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        
        with open(domain_data_path,"r") as f:
            self.domains = json.load(f)
        
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

    def self_query(self, raw_query: str) -> Dict[str, Any]:
        """
        Translates a raw query into a structured filter using an LLm pass
        """
        system_prompt = (
            "You are a query-to-structured-filter translator. "
            f"Available domains: {list(self.domains.values())}. "
            "Output ONLY JSON. Example: {'domain': 'cardiology'}. Output {} if no domain matches."
        )
        response = self.LLM.chat(system_prompt, f"User Query: {raw_query}", json_mode=True)
        try:
            filters = json.loads(response)
            
            if "domain" in filters:
                if isinstance(filters["domain"], str):
                    filters["domain"] = [filters["domain"]]
                elif not isinstance(filters["domain"], list):
                    filters["domain"] = ["general"]
            else:
                filters["domain"] = ["general"]

            if "general" not in filters["domain"]:
                filters["domain"].append("general")

            return filters
        except (json.JSONDecodeError, TypeError):
            return {"domain": ["general"]}

    def hybrid_retrieval(self, query: str, filters: Dict[str, Any], top_n: int = 10):
        valid_domains = filters.get("domain", ["general"])
        
        if ["general"] in valid_domains:
            allowed_ids = list(range(len(self.corpus)))
        else:
            allowed_ids = [i for i, doc in enumerate(self.corpus) if doc.domain in valid_domains]

        if not allowed_ids:
            return [], []

        # 1. Dense Retrieval (FAISS) with ID Filtering
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        id_selector = faiss.IDSelectorBatch(allowed_ids)
        
        params = faiss.SearchParameters(sel=id_selector)
        
        faiss_scores, faiss_indices = self.faiss_index.search(
            query_emb, 
            top_n, 
            params=params
        )
        
        # 2. Sparse Retrieval (BM25) with ID Filtering
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # we set scores of disallowed IDs to -inf to ensure they are ranked at the bottom
        mask = np.full(len(doc_scores), -np.inf)
        mask[allowed_ids] = 0
        masked_bm25_scores = doc_scores + mask
        
        bm25_ranking = np.argsort(masked_bm25_scores)[::-1][:top_n]

        return bm25_ranking.tolist(), faiss_indices[0].tolist()

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
        # 1. Self-Query
        filters = self.self_query(query)
        
        # 2. Hybrid Retrieval
        bm25_ranks, faiss_ranks = self.hybrid_retrieval(query, filters, top_n=10)
        
        # 3. RRF Fusion
        fused_indices = self.rrf(bm25_ranks, faiss_ranks)
        
        # 4. Enrichment
        return self.provenance_enrichment(fused_indices[:self.k_doc])



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