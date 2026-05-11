from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os 
import time

from utils.data.ingestion_pipeline import TFRDataPreprocessor
from utils.api.make_response import make_response
from utils.pipeline.init_pipline import initialize_pipeline
from utils.data.get_pubmed_xml import fetch_pubmed_xml_to_file
from utils.api.audit import AuditTrail

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DOMAIN_DATA_PATH = os.environ.get("DOMAIN_DATA_PATH", "./data/domain.json")
DOC_DB_PATH = os.environ.get("DOC_DB_PATH", "document.db")
SJR_CSV_PATH = os.environ.get("SJR_CSV_PATH", "./data/scimagojr_2023.csv")

audit = AuditTrail("logs/audit_log.csv")    

processor = TFRDataPreprocessor(DOMAIN_DATA_PATH, SJR_CSV_PATH)

# for hot-reloading
pipeline = None

pipeline = initialize_pipeline(DOC_DB_PATH,API_KEY,DOMAIN_DATA_PATH)

app = Flask(__name__)


@app.route("/seed", methods=["POST"])
def seed_database():
    """
    Builds a corpus for a specific clinical topic
    
    args:
    - query: str (e.g. "diabetes AND treatment")

    returns:
    - count of ingested documents
    """
    data = request.get_json()
    medical_query = data.get("query")
    start_time = time.time()
    
    # download XML from PubMed
    xml_path = fetch_pubmed_xml_to_file(
        query=medical_query,
        max_results=100,
        email=os.getenv("NCBI_EMAIL"),
        api_key=os.getenv("NCBI_API_KEY"),
        output_path="./data/seed_pubmed_data.xml"
    )
    
    if xml_path:
        # ingestion
        chunks = processor.parse_pubmed_xml(xml_path)
        processor.export_to_sqlite(chunks, DOC_DB_PATH)
        
        # Refresh Pipeline
        global pipeline
        pipeline = initialize_pipeline(DOC_DB_PATH,API_KEY,DOMAIN_DATA_PATH)
        
        end_time = time.time()
        latency = end_time - start_time
        audit.log_event(action="seed", query=medical_query, results_count=len(chunks), latency=latency)
        return make_response({"count": len(chunks)}, message="Database seeded successfully")
    
    audit.log_event(action="seed", query=medical_query, status="error")
    return make_response( message="Seeding failed", status="error"),500

@app.route("/ingest", methods=["POST"])
def ingest_data():
    """
    Endpoint to process new PubMed XMLs into the SQLite DB
    
    args:
    - path: str (path to the PubMed XML file)

    returns:
    - count of ingested documents
    """
    data = request.get_json()
    doc_path = data.get("path")
    start_time = time.time()
    
    if not doc_path or not os.path.exists(doc_path):
        return make_response( message="Error: Valid XML path required", status="error"),400

    try:
        result_chunks = processor.parse_pubmed_xml(doc_path)
        processor.export_to_sqlite(result_chunks, DOC_DB_PATH)
        
        # Refresh Pipeline
        global pipeline
        pipeline = initialize_pipeline(DOC_DB_PATH,API_KEY,DOMAIN_DATA_PATH)
        end_time = time.time()
        latency = end_time - start_time
        audit.log_event(action="ingest", query=doc_path, results_count=len(result_chunks), latency=latency)
        return make_response(
            {"count": len(result_chunks)}, 
            message=f"Successfully ingested and indexed: {doc_path}"
        )
    except Exception as e:
        print(f"Ingestion failed: {str(e)}")
        audit.log_event(action="ingest", query=doc_path, status="error")
        return make_response( message=f"Ingestion failed: {str(e)}", status="error"),500

@app.route("/query", methods=["POST"])
def run_query():
    """
    Endpoint to perform Trust-Weighted Retrieval
    
    args:
    - query: str (the clinical question to retrieve for)
    
    returns:
    - list of retrieved documents with metadata   
    """
    if pipeline is None:
        return make_response( message="Pipeline not initialized. Please ingest data first.", status="error"),400

    data = request.get_json()
    query_text = data.get("query")
    
    if not query_text:
        return make_response(message="Query text is required", status="error"),400

    try:
        start_time = time.time()
        results = pipeline.retrieve(query=query_text)
        end_time = time.time()
        latency = end_time - start_time
        audit.log_event(action="query", query=query_text, results_count=len(results), top_pmid=results[0]["provenance"].get("chunk_id","N/A") if results else "N/A", latency=latency)
        return make_response(results, message="Retrieved with success")
    except Exception as e:
        print(f"Query failed: {e}")
        audit.log_event(action="query", query=query_text, status="error")
        return make_response( message=f"Retrieval error: {str(e)}", status="error"),500

@app.route("/health", methods=["GET"])
def health_check():
    """health check endpoint"""
    return make_response({"status": "ok"}, message="Health check passed")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    app.run(debug=debug, host=host, port=port)