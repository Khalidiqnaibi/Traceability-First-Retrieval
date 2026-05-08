import xml.etree.ElementTree as ET
import json
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClinicalChunk:
    chunk_id: str
    text: str
    source: str
    journal_tier: str
    evidence_level: int # OCEBM 1-5
    publication_year: int
    domain: str

class TFRDataPreprocessor:
    def __init__(self):
        # 1. MeSH to Domain Mapping 
        self.clinical_branches = {
            "C01": "Infections", "C02": "Infections", "C03": "Infections",
            "C04": "Neoplasms (Oncology)",
            "C05": "Musculoskeletal Diseases",
            "C06": "Digestive System Diseases (Gastroenterology)",
            "C07": "Stomatognathic Diseases",
            "C08": "Respiratory Tract Diseases",
            "C09": "Otorhinolaryngologic Diseases",
            "C10": "Nervous System Diseases (Neurology)",
            "C11": "Eye Diseases (Ophthalmology)",
            "C12": "Urogenital Diseases",
            "C13": "Female Urogenital Diseases and Pregnancy (OBGYN)",
            "C14": "Cardiovascular Diseases",
            "C15": "Hemic and Lymphatic Diseases (Hematology)",
            "C16": "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)",
            "C17": "Skin and Connective Tissue Diseases (Dermatology)",
            "C18": "Nutritional and Metabolic Diseases",
            "C19": "Endocrine System Diseases",
            "C20": "Immune System Diseases",
            "C21": "Disorders of Environmental Origin",
            "C22": "Animal Diseases (Veterinary)",
            "C23": "Pathological Conditions, Signs and Symptoms",
            "C24": "Occupational Diseases",
            "C25": "Chemically-Induced Disorders (Toxicology)",
            "C26": "Wounds and Injuries",
            "F03": "Mental Disorders (Psychiatry)"
        }

        # Journal Tier Database (Mock .. will link to SCImago/SJR csv)
        self.tier_db = {
            "New England Journal of Medicine": "Q1",
            "The Lancet": "Q1",
            "Journal of Clinical Oncology": "Q1",
            "Medical Blog Daily": "Unranked"
        }

    def _classify_evidence_level(self, pub_types: List[str]) -> int:
        """
        Maps PubMed Publication Types to OCEBM Levels
        Level 1: Systematic Review / RCT
        Level 2: Cohort Study
        Level 3: Case-Control
        Level 4: Case Series
        Level 5: Expert Opinion / Editorial
        """
        pt_string = "|".join(pub_types).lower()
        
        if "meta-analysis" in pt_string or "systematic review" in pt_string:
            return 1
        if "randomized controlled trial" in pt_string:
            return 1
        if "cohort study" in pt_string:
            return 2
        if "case-control" in pt_string:
            return 3
        if "case reports" in pt_string:
            return 4
        
        return 5 # Default to Expert Opinion/General text

    def _resolve_domain(self, mesh_terms: List[str]) -> str:
        """Maps MeSH headings to TFR Domains."""
        for term in mesh_terms:
            if term in self.clinical_branches:
                return self.clinical_branches[term]
        return "general"

    def parse_pubmed_xml(self, xml_file_path: str) -> List[ClinicalChunk]:
        """
        Parses PubMed XML and extracts structured TFR objects
        """
        logger.info(f"Starting ingestion of {xml_file_path}...")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        processed_chunks = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # basic metadata
                medline = article.find(".//MedlineCitation")
                article_data = medline.find("Article")
                
                title = article_data.findtext("ArticleTitle")
                journal_name = article_data.find("Journal/Title").text
                year = int(article_data.find(".//PubRefDate/Year").text) if article_data.find(".//PubRefDate/Year") is not None else -1
                
                # abstract chunks
                abstract_nodes = article_data.findall(".//AbstractText")
                full_abstract = " ".join([node.text for node in abstract_nodes if node.text])
                
                # evidence level detection
                pub_types = [pt.text for pt in article_data.findall(".//PublicationType")]
                evidence_level = self._classify_evidence_level(pub_types)
                
                # domain detection (MeSH)
                mesh_headings = [mh.findtext("DescriptorName") for mh in medline.findall(".//MeshHeading")]
                domain = self._resolve_domain(mesh_headings)
                
                # journal tier Lookup
                tier = self.tier_db.get(journal_name, "UNRANKED")
                
                # Create chunk
                chunk = ClinicalChunk(
                    chunk_id=f"pmid_{medline.findtext('PMID')}",
                    text=full_abstract,
                    source=journal_name,
                    journal_tier=tier,
                    evidence_level=evidence_level,
                    publication_year=year,
                    domain=domain
                )
                processed_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Skipping article due to parsing error: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_chunks)} chunks.")
        return processed_chunks

    def export_to_tfr_json(self, chunks: List[ClinicalChunk], output_path: str):
        """Saves processed data for the TFR Pipeline."""
        data = [asdict(c) for c in chunks]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Dataset exported to {output_path}")

if __name__ == "__main__":
    processor = TFRDataPreprocessor()
    
    try:
        chunks = processor.parse_pubmed_xml("raw_pubmed_data.xml")
        
        processor.export_to_tfr_json(chunks, "processed_clinical_dataset.json")
    except FileNotFoundError:
        logger.error("No XML file found. Please provide a PubMed XML export.")