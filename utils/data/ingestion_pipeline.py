"""
TFR Ingestion Pipeline -- v2
========================================
Fixed classification strategy:

Evidence Level (OCEBM 1-5)
  Signal priority:  PublicationType  ->  MeSH study-design terms  ->  abstract text patterns
  The original code only checked PublicationType, missing the 40+ study-design
  MeSH terms PubMed adds to plain "Journal Article" records.

Domain
  Signal priority:  MeSH headings  ->  author Keywords  ->  title  ->  abstract (first 300 chars)
  The original code searched only MeSH + title with a narrow keyword list, missing
  ~44 % of articles.  The new keyword map is 4x larger and ordered from most
  specific to most general so the first match wins correctly.
"""

import xml.etree.ElementTree as ET
import sqlite3
import json
import os
import csv
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

from data.domain import DOMAIN_KEYWORDS


@dataclass
class ClinicalChunk:
    chunk_id: str
    text: str
    source: str
    journal_tier: str
    evidence_level: int   # OCEBM 1-5
    publication_year: int
    domain: str


# Evidence level: study-design MeSH terms mapped to OCEBM level
MESH_DESIGN_TO_LEVEL: List[Tuple[str, int]] = [
    # Level 1 -- Highest evidence
    ("meta-analysis",              1),
    ("systematic review",          1),
    ("randomized controlled trial",1),
    ("randomised controlled trial",1),
    # Level 2 -- Prospective observational
    ("cohort studies",             2),
    ("cohort study",               2),
    ("prospective studies",        2),
    ("prospective study",          2),
    ("longitudinal studies",       2),
    # Level 3 -- Retrospective / comparative
    ("case-control studies",       3),
    ("case-control study",         3),
    ("cross-sectional studies",    3),
    ("cross-sectional study",      3),
    ("retrospective studies",      3),
    ("retrospective study",        3),
    ("observational study",        3),
    ("comparative study",          3),
    ("controlled before-after",    3),
    # Level 4 -- Descriptive
    ("case reports",               4),
    ("case report",                4),
    ("case series",                4),
    ("pilot study",                4),
]

# Abstract text regex patterns used only when PubType + MeSH are both uninformative
_ABSTRACT_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r'\brandomly\s+(assigned|allocated|selected)\b', re.I), 1),
    (re.compile(r'\brandomized\b|\brandomised\b', re.I),                1),
    (re.compile(r'\bopen.label\s+rct\b|\bplacebo.controlled\b', re.I),  1),
    (re.compile(r'\bprospective\s+cohort\b', re.I),                     2),
    (re.compile(r'\bcase.control\b', re.I),                             3),
    (re.compile(r'\bretrospective\s+(analysis|study|review)\b', re.I),  3),
    (re.compile(r'\bcross.sectional\b', re.I),                          3),
    (re.compile(r'\bwe\s+report\s+(a|an)\s+(case|patient)\b', re.I),   4),
    (re.compile(r'\bcase\s+report\b', re.I),                            4),
]

class TFRDataPreprocessor:

    def __init__(
        self,
        domain_keywords: List[Tuple[str, str]] = None,
        sjr_csv_path: str = "./data/scimagojr_2023.csv",
    ):
        # Domain map: ordered list of (lowercase_substring, label) pairs.
        # Can be overridden at construction time; falls back to the module-level constant.
        self.domain_keywords: List[Tuple[str, str]] = domain_keywords or DOMAIN_KEYWORDS

        # Journal tier lookup from SCImago CSV
        self.tier_db: Dict[str, str] = self._load_sjr_database(sjr_csv_path)

    def _load_sjr_database(self, csv_path: str) -> Dict[str, str]:
        """Loads the SCImago Journal Rank CSV into a lowercase-keyed lookup dict."""
        tier_map: Dict[str, str] = {}
        if not os.path.exists(csv_path):
            print(f"[WARN] SJR CSV not found at {csv_path}. All journals default to 'Unranked'.")
            return tier_map

        print("Loading SCImago Journal Rank database ...")
        try:
            with open(csv_path, mode="r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                for row in reader:
                    title = row.get("Title", "").strip().lower()
                    quartile = row.get("SJR Best Quartile", "").strip()
                    if not quartile or quartile == "-":
                        quartile = "Unranked"
                    if title:
                        tier_map[title] = quartile
            print(f"Loaded {len(tier_map):,} journal rankings.")
        except Exception as exc:
            print(f"[ERROR] Failed to parse SJR database: {exc}")

        return tier_map

    # Evidence level

    def _classify_evidence_level(
        self,
        pub_types: List[str],
        mesh_terms: List[str],
        abstract_text: str,
    ) -> int:
        """
        Three-signal waterfall: PublicationType -> MeSH study-design -> abstract text.

        Returns OCEBM level 1-5.
        """
        # Signal 1: PublicationType (most reliable when present)
        pt_blob = " | ".join(pub_types).lower()

        if "meta-analysis" in pt_blob or "systematic review" in pt_blob:
            return 1
        if "randomized controlled trial" in pt_blob or "randomised controlled trial" in pt_blob:
            return 1
        if "observational study" in pt_blob or "comparative study" in pt_blob:
            # PubType alone doesn't tell us prospective vs retrospective;
            # fall through to MeSH for a tighter assignment.
            pass
        if "case reports" in pt_blob:
            return 4
        if "letter" in pt_blob or "editorial" in pt_blob or "comment" in pt_blob:
            return 5
        if "review" in pt_blob:
            # Narrative reviews without "systematic" -> Level 5 (expert synthesis)
            return 5

        # Signal 2: MeSH study-design terms
        mesh_blob = " | ".join(mesh_terms).lower()

        for design_term, level in MESH_DESIGN_TO_LEVEL:
            if design_term in mesh_blob:
                return level

        # Signal 3: Abstract text patterns (last resort)
        if abstract_text:
            for pattern, level in _ABSTRACT_PATTERNS:
                if pattern.search(abstract_text):
                    return level

        return 5  # Expert opinion / unclassifiable

    def _resolve_domain(
        self,
        mesh_terms: List[str],
        keywords: List[str],
        title: str,
        abstract_text: str,
    ) -> str:
        """
        Four-signal waterfall: MeSH -> Keywords -> title -> abstract snippet.

        Searches each signal in order; returns the label of the first keyword match.
        Uses the ordered DOMAIN_KEYWORDS list so specific terms beat generic ones.
        """
        # Build candidate signals separately so we can try them in order
        # and stop at the first signal layer that gives a match.
        signals = [
            " ".join(mesh_terms),           # most authoritative
            " ".join(keywords),             # author-provided, often disease-specific
            title,                          # concise and on-topic
            abstract_text[:400],            # first 400 chars -- usually the background
        ]

        for signal in signals:
            if not signal.strip():
                continue
            lower = signal.lower()
            for keyword, label in self.domain_keywords:
                if keyword in lower:
                    return label
            # If this signal layer produced no match, try the next layer

        return "general"

    def parse_pubmed_xml(self, xml_file_path: str) -> List[ClinicalChunk]:
        """Parses a PubMed XML export and returns a list of ClinicalChunk objects."""
        print(f"Parsing {xml_file_path} ...")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        chunks: List[ClinicalChunk] = []
        skipped = 0

        for article in root.findall(".//PubmedArticle"):
            try:
                medline      = article.find(".//MedlineCitation")
                article_data = medline.find("Article")

                pmid         = medline.findtext("PMID") or "unknown"
                title_node   = article_data.find("ArticleTitle")
                title_text   = (title_node.text or "") if title_node is not None else ""

                journal_node = article_data.find("Journal/Title")
                journal_name = (journal_node.text or "").strip() if journal_node is not None else ""

                pub_date_node = article_data.find(".//JournalIssue/PubDate")
                year: int = -1
                if pub_date_node is not None:
                    year_str = pub_date_node.findtext("Year")
                    if not year_str:
                        medline_date = pub_date_node.findtext("MedlineDate")
                        year_str = medline_date[:4] if medline_date else None
                    try:
                        year = int(year_str) if year_str else -1
                    except ValueError:
                        year = -1

                abstract_parts = [
                    node.text
                    for node in article_data.findall(".//AbstractText")
                    if node.text
                ]
                abstract_text = " ".join(abstract_parts)
                    
                pub_types = [
                    pt.text
                    for pt in article_data.findall(".//PublicationType")
                    if pt.text
                ]

                mesh_terms = [
                    m.text
                    for m in medline.findall(".//MeshHeading/DescriptorName")
                    if m.text
                ]
                keywords = [
                    k.text
                    for k in article.findall(".//Keyword")
                    if k.text
                ]
                evidence_level = self._classify_evidence_level(
                    pub_types, mesh_terms, abstract_text
                )
                domain = self._resolve_domain(
                    mesh_terms, keywords, title_text, abstract_text
                )

                tier = self.tier_db.get(journal_name.strip().lower(), "Unranked")

                chunks.append(ClinicalChunk(
                    chunk_id=f"pmid_{pmid}",
                    text=abstract_text,
                    source=journal_name,
                    journal_tier=tier,
                    evidence_level=evidence_level,
                    publication_year=year,
                    domain=domain,
                ))

            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skipping PMID={medline.findtext('PMID') if medline is not None else '?'}: {exc}")
                continue

        print(f"Processed {len(chunks)} chunks ({skipped} skipped).")
        return chunks

    def export_to_sqlite(self, chunks: List[ClinicalChunk], db_path: str) -> None:
        """Upserts processed chunks into a SQLite database."""
        print(f"Exporting {len(chunks)} chunks -> {db_path} ...")
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS clinical_chunks (
                chunk_id         TEXT PRIMARY KEY,
                text             TEXT,
                source           TEXT,
                journal_tier     TEXT,
                evidence_level   INTEGER,
                publication_year INTEGER,
                domain           TEXT
            )
        """)

        cur.executemany(
            """
            INSERT OR REPLACE INTO clinical_chunks
              (chunk_id, text, source, journal_tier, evidence_level, publication_year, domain)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (c.chunk_id, c.text, c.source, c.journal_tier,
                 c.evidence_level, c.publication_year, c.domain)
                for c in chunks
            ],
        )

        conn.commit()
        conn.close()
        print(f"Export complete -> {db_path}")


if __name__ == "__main__":
    processor = TFRDataPreprocessor()

    xml_path = "raw_pubmed_data.xml"
    db_path  = "processed_clinical_dataset.db"

    try:
        chunks = processor.parse_pubmed_xml(xml_path)
        processor.export_to_sqlite(chunks, db_path)
    except FileNotFoundError:
        print(f"[ERROR] XML file not found: {xml_path}")