
from dataclasses import dataclass

@dataclass
class ClinicalDocument:
    chunk_id: str
    text: str
    source: str
    journal_tier: str
    evidence_level: int  # OCEBM Level: 1 (RCT) to 5 (Expert Opinion)
    publication_year: int
    domain: str