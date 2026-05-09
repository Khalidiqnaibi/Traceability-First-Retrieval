from Bio import Entrez
import os
import logging

logger = logging.getLogger(__name__)

def fetch_pubmed_xml_to_file(
        query: str, 
        max_results: int = 50, 
        email: str = None, 
        api_key: str = None, 
        output_path: str = "raw_pubmed_data.xml"
    ):
    """
    Automated PubMed XML acquisition for reproducible research
    """
    if not email:
        raise ValueError("NCBI requires an email address for API usage")

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    try:
        logger.info(f"Searching for query: {query}")
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        count = int(search_results["Count"])
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]

        if count == 0:
            logger.warning("No results found for this query")
            return None

        # efficient for large batches
        logger.info(f"Fetching {min(count, max_results)} articles...")
        fetch_handle = Entrez.efetch(
            db="pubmed",
            retmax=max_results,
            webenv=webenv,
            query_key=query_key,
            rettype="xml",
            retmode="text"
        )
        
        data = fetch_handle.read()
        fetch_handle.close()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        logger.info(f"Successfully saved XML to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to fetch PubMed data: {e}")
        return None