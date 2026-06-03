

import json
import os
from time import time
from dotenv import load_dotenv

from utils.data.get_pubmed_xml import fetch_pubmed_xml_to_file

load_dotenv()

def run_batch_seed(queries_file_path,doc_db_path, processor, audit):
    with open(queries_file_path, "r", encoding="utf-8") as f:
        queries_list = json.load(f)

    
    print(f"Initializing seeding of {len(queries_list)} items...")
    start_time = time.time()
    total_ingested = 0
    for item in queries_list:
        query = item.get("query")
        max_results = item.get("max_results", 100)
        xml_path = fetch_pubmed_xml_to_file(
            query=query,
            max_results=max_results,
            email=os.getenv("NCBI_EMAIL"),
            api_key=os.getenv("NCBI_API_KEY"),
            output_path=f"./data_in/seed_pubmed_data.xml"
        )
        if xml_path:
            chunks = processor.parse_pubmed_xml(xml_path)
            processor.export_to_sqlite(chunks, doc_db_path)
            total_ingested += len(chunks)
            print(f"Seeded query '{query}' with {len(chunks)} documents.")
    latency = time.time() - start_time
    print(f"Completed seeding {total_ingested} documents.")

    audit.log_event(action="batch_seed", query=f"Batch seeding from {queries_file_path}", status=f"Seeded {total_ingested} documents", latency=latency)

    return total_ingested
    

if __name__ == "__main__":
    from utils.data.ingestion_pipeline import TFRDataPreprocessor
    from utils.api.audit import AuditTrail

    processor = TFRDataPreprocessor()
    audit = AuditTrail("data_out/audit_log.csv")
    run_batch_seed("./data_in/seed_queries.json", "./data_in/documents.db", processor, audit)