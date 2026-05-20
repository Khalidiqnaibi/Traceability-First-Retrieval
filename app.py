from flask import Flask, json, request, jsonify
from dotenv import load_dotenv
import os 
import time

from utils.pipeline.init_standard_pipeline import initialize_standard_pipeline
from utils.pipeline.init_pipline import initialize_pipeline

from utils.data.ingestion_pipeline import TFRDataPreprocessor
from utils.api.make_response import make_response
from utils.data.get_pubmed_xml import fetch_pubmed_xml_to_file
from utils.api.audit import AuditTrail

load_dotenv()

DOC_DB_PATH = os.environ.get("DOC_DB_PATH", "document.db")
SJR_CSV_PATH = os.environ.get("SJR_CSV_PATH", "./data/scimagojr_2023.csv")

audit = AuditTrail("logs/audit_log.csv")    

processor = TFRDataPreprocessor(sjr_csv_path=SJR_CSV_PATH)

# for hot-reloading
tfr_pipeline = None
standard_pipeline =None


tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)

app = Flask(__name__)

@app.route("/seed/batch", methods=["POST"])
def batch_seed():
    '''Endpoint to seed the database with a batch of PubMed queries from a JSON file'''
    data = request.get_json() or {}

    path = os.getenv("SEED_QUERIES_PATH", "./data/seed_queries.json")

    queries_file_path = data.get("queries_path", path)

    if not os.path.exists(queries_file_path):
        return make_response( message=f"Queries seed file not found at {queries_file_path}", status="error"),400
    
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
            output_path=f"./data/seed_pubmed_data.xml"
        )
        if xml_path:
            chunks = processor.parse_pubmed_xml(xml_path)
            processor.export_to_sqlite(chunks, DOC_DB_PATH)
            total_ingested += len(chunks)
            print(f"Seeded query '{query}' with {len(chunks)} documents.")
    latency = time.time() - start_time
    print(f"Completed seeding {total_ingested} documents.")

    # Refresh Pipelines after batch seeding
    global tfr_pipeline, standard_pipeline
    tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
    standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)
    
    audit.log_event(action="batch_seed", query=f"Batch seeding from {queries_file_path}", status=f"Seeded {total_ingested} documents", latency=latency)
    return make_response({"count": total_ingested}, message=f"Batch seeding completed with {total_ingested} documents ingested")

@app.route("/seed", methods=["POST"])
def seed_database():
    """
    Builds a corpus for a specific clinical topic
    
    args:
    - query: str (e.g. "diabetes AND treatment")
    - max_results: int (number of PubMed articles to fetch and ingest, default=100)

    returns:
    - count of ingested documents
    """
    data = request.get_json()
    medical_query = data.get("query")
    max_results = data.get("max_results",100)
    start_time = time.time()
    
    # download XML from PubMed
    xml_path = fetch_pubmed_xml_to_file(
        query=medical_query,
        max_results=max_results,
        email=os.getenv("NCBI_EMAIL"),
        api_key=os.getenv("NCBI_API_KEY"),
        output_path="./data/seed_pubmed_data.xml"
    )
    
    if xml_path:
        # ingestion
        chunks = processor.parse_pubmed_xml(xml_path)
        processor.export_to_sqlite(chunks, DOC_DB_PATH)
        
        # Refresh Pipeline
        global tfr_pipeline, standard_pipeline
        tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
        standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)

        end_time = time.time()
        latency = end_time - start_time
        audit.log_event(action="seed", query=medical_query, latency=latency)
        return make_response({"count": len(chunks)}, message="Database seeded successfully")
    
    audit.log_event(action="seed", query=medical_query, status=f"Seeding error: {xml_path} not found")
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
        global tfr_pipeline, standard_pipeline
        tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
        standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)
        end_time = time.time()
        latency = end_time - start_time
        audit.log_event(action="ingest", query=doc_path, latency=latency)
        return make_response(
            {"count": len(result_chunks)}, 
            message=f"Successfully ingested and indexed: {doc_path}"
        )
    except Exception as e:
        print(f"Ingestion failed: {str(e)}")
        audit.log_event(action="ingest", query=doc_path, status=f"Ingestion error: {str(e)}")
        return make_response( message=f"Ingestion failed: {str(e)}", status="error"),500

@app.route("/reload", methods=["POST"])
def reload_pipelines():
    """Endpoint to manually trigger pipelines reload (e.g. after new data ingestion)"""
    global tfr_pipeline, standard_pipeline
    try:
        tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
        standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)
        return make_response(message="Pipelines reloaded successfully")
    except Exception as e:
        print(f"Pipeline reload failed: {e}")
        return make_response( message=f"Pipelines reload error: {str(e)}", status="error"),500
 
@app.route("/tfr/query", methods=["POST"])
def run_query():
    """
    Endpoint to perform Trust-Weighted Retrieval
    
    args:
    - query: str (the clinical question to retrieve for)
    
    returns:
    - list of retrieved documents with metadata   
    """
    if tfr_pipeline is None:
        return make_response( message="Pipeline not initialized. Please ingest data first.", status="error"),400

    data = request.get_json()
    query_text = data.get("query")
    
    if not query_text:
        return make_response(message="Query text is required", status="error"),400

    try:
        results = tfr_pipeline.retrieve(query=query_text)
        return make_response(results, message="Retrieved with success")
    except Exception as e:
        print(f"Query failed: {e}")
        audit.log_event(action="query", query=query_text, status=f"Retrieval error: {str(e)}")
        return make_response( message=f"Retrieval error: {str(e)}", status="error"),500

@app.route("/ablation/query", methods=["POST"])
def run_ablation():
    """Endpoint to run ablation study comparing Standard RRF vs TWR"""
    if tfr_pipeline is None or standard_pipeline is None:
        return make_response( message="Pipelines not initialized. Please initialize and ingest data first.", status="error"),400

    data = request.get_json()
    query_text = data.get("query")
    
    if not query_text:
        return make_response(message="Query text is required", status="error"),400

    try:
        start_time = time.time()
        tfr_results = tfr_pipeline.retrieve(query=query_text)
        standard_results = standard_pipeline.retrieve(query=query_text)
        latency = time.time() - start_time
        audit.log_event(action="ablation_query", query=query_text, latency=latency)
        return make_response({
            "TFR_results": tfr_results,
            "Standard_RRF_results": standard_results
        }, message="Ablation results retrieved successfully")
    except Exception as e:
        print(f"Ablation query failed: {e}")
        audit.log_event(action="ablation_query", query=query_text, status=f"Ablation Retrieval error: {str(e)}")
        return make_response( message=f"Ablation Retrieval error: {str(e)}", status="error"),500

@app.route("/ablation/batch", methods=["POST"])
def run_batch_ablation():
    """
    Executes a comprehensive batch ablation study across all queries in a dataset.
    Extracts intermediate sparse/dense rankings and logs detailed metrics to PipelineAudit.
    """
    if tfr_pipeline is None or standard_pipeline is None:
        return make_response( message="Pipelines not initialized. Please seed or ingest data first.", status="error"),400

    data = request.get_json() or {}
    queries_file_path = data.get("queries_path", "./data/queries.json")

    if not os.path.exists(queries_file_path):
        return make_response( message=f"Queries benchmark file not found at {queries_file_path}", status="error"),400

    with open(queries_file_path, "r", encoding="utf-8") as f:
        queries_list = json.load(f)

    summary_metrics = []
    start_time = time.time()

    for item in queries_list:
        query_id = item.get("id")
        query_text = item.get("query")
        dimension = item.get("ablation_dimension")
        expected_advantage = item.get("expected_twr_advantage")

        try:
            tfr_results = tfr_pipeline.retrieve(query=query_text)
            standard_results = standard_pipeline.retrieve(query=query_text)
        except Exception as e:
            print(f"Ablation query failed: {e}")
            audit.log_event(action="batch_ablation", query=query_text, status=f"Ablation Retrieval error: {str(e)}")
            return make_response( message=f"Ablation Retrieval error: {str(e)}", status="error"),500

        summary_metrics.append({
            "id": query_id,
            "query": query_text,
            "ablation_dimension": dimension,
            "expected_twr_advantage": expected_advantage,
            "tfr_results": tfr_results,
            "standard_results": standard_results
        })

    audit.log_event(
        action="batch_ablation",
        query=f"Batch ablation over {len(queries_list)} queries",
        latency=time.time() - start_time
    )

    return make_response(
        message=f"Ablation evaluation completed over {len(queries_list)} benchmark queries.",
        data=summary_metrics
    ), 200
   
@app.route("/standard/query", methods=["POST"])
def standard_query():
    """Endpoint for standard query processing"""
    if standard_pipeline is None:
        return make_response( message="Standard Pipeline not initialized. Please ingest data first.", status="error"),400

    data = request.get_json()
    query_text = data.get("query")
    
    if not query_text:
        return make_response(message="Query text is required", status="error"),400

    try:
        results = standard_pipeline.retrieve(query=query_text)
        return make_response(results, message="Retrieved with success")
    except Exception as e:
        print(f"Standard query failed: {e}")
        audit.log_event(action="query", query=query_text, status=f"Retrieval error: {str(e)}", pipeline="Standard")
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