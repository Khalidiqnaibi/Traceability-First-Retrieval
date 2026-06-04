from flask import Flask, json, request, jsonify
from dotenv import load_dotenv
import os 
import time

from eval.dr_blinded_review import generate_blinded_review_workbooks
from eval.llm_eval import run_blinded_llm_pass
from eval.run_eval import run_eval
from eval.batch_ablation import run_batch_ablation

from utils.data.save_dr_review import save_blinded_review_log
from utils.pipeline.init_standard_pipeline import initialize_standard_pipeline
from utils.pipeline.init_pipline import initialize_pipeline
from utils.data.batch_seed_db import run_batch_seed
from utils.data.ingestion_pipeline import TFRDataPreprocessor
from utils.api.make_response import make_response
from utils.data.get_pubmed_xml import fetch_pubmed_xml_to_file
from utils.api.audit import AuditTrail

load_dotenv()

DOC_DB_PATH = os.environ.get("DOC_DB_PATH", "document.db")
SJR_CSV_PATH = os.environ.get("SJR_CSV_PATH", "./data_in/scimagojr_2023.csv")

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
    
    total_ingested = run_batch_seed(queries_file_path, DOC_DB_PATH, processor, audit)
    
    # Refresh Pipelines after batch seeding
    global tfr_pipeline, standard_pipeline
    tfr_pipeline = initialize_pipeline(DOC_DB_PATH)
    standard_pipeline = initialize_standard_pipeline(DOC_DB_PATH)
    
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
        output_path="./data_in/seed_pubmed_data.xml"
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
def run_batch_ablation_route():
    """
    Executes a comprehensive batch ablation study across all queries in a dataset.
    Extracts intermediate sparse/dense rankings and logs detailed metrics to PipelineAudit.
    """
    if tfr_pipeline is None or standard_pipeline is None:
        return make_response( message="Pipelines not initialized. Please seed or ingest data first.", status="error"),400

    data = request.get_json() or {}
    queries_file_path = data.get("queries_path", "./data_in/queries.json")

    if not os.path.exists(queries_file_path):
        return make_response( message=f"Queries benchmark file not found at {queries_file_path}", status="error"),400

    results = run_batch_ablation(queries_file_path, tfr_pipeline, standard_pipeline, audit)

    return make_response(
        message=f"Ablation evaluation completed over {len(results)} benchmark queries.",
        data=results
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

@app.route("/eval", methods=["GET"])
def run_evaluation():
    """Endpoint to run evaluation"""
    args = request.args

    arg_evaluation_log_path = args.get("log")
    arg_queries_path = args.get("queries")
    arg_out_dir = args.get("out_dir")

    run_eval(evaluation_log_path=arg_evaluation_log_path, queries_path=arg_queries_path, out_dir=arg_out_dir)
    return make_response(message="Evaluation executed. Check logs for details.")

@app.route("/ablation/llm", methods=["POST"])
def run_llm_ablation():
    """Endpoint to run LLM ablation study"""
    data = request.get_json() or {}
    log_csv_path = data.get("log_csv_path", "data_out/pipeline_audit_log.csv")
    output_csv_path = data.get("output_csv_path", "data_in/blinded_clinical_review.csv")
    queries_json_path = data.get("queries_json_path", "data_in/queries.json")

    if not os.path.exists(log_csv_path):
        return make_response( message=f"Audit log file not found at {log_csv_path}", status="error"),400
    
    if not os.path.exists(queries_json_path):
        return make_response( message=f"Queries file not found at {queries_json_path}", status="error"),400

    run_blinded_llm_pass(queries_json_path, log_csv_path, output_csv_path)
    return make_response(message=f"LLM ablation completed. Blinded evaluation sheet saved to {output_csv_path}")

@app.route("/ablation/dr/generate", methods=["POST"])
def run_dr_ablation():
    """Endpoint to run blinded review workbooks generation"""
    data = request.get_json() or {}
    log_csv_path = data.get("log_csv_path", "data_out/blinded_clinical_review.csv")
    queries_json_path = data.get("queries_json_path", "data_in/queries.json")

    if not os.path.exists(log_csv_path):
        return make_response( message=f"Blinded log file not found at {log_csv_path}", status="error"),400

    if not os.path.exists(queries_json_path):
        return make_response( message=f"Queries file not found at {queries_json_path}", status="error"),400

    generate_blinded_review_workbooks(log_csv_path, queries_json_path)
    return make_response(message=f"Blinded review workbooks generated.")

@app.route("/ablation/dr/review", methods=["POST"])
def run_dr_review():
    """
    # Endpoint to run blinded dr review
    
    ## Expects:
    - data : {
        "log_csv_path": "path/to/{domain}_Workbook_Orthopedics.csv",
        "Clinical Accuracy": int (1-5),
        "Evidence Basis": int (1-5), # Does the answer cite appropriate evidence?
        "Safety/Risk": int (1-5), # Does the advice pose any danger?
        "out_log_path": "path/to/save/dr_review_results.csv"
    }
    """
    data = request.get_json() or {}
    log_csv_path = data.get("log_csv_path", "data_out/blinded_clinical_review.csv")
    queries_json_path = data.get("queries_json_path", "data_in/queries.json")

    if not os.path.exists(log_csv_path):
        return make_response( message=f"Blinded log file not found at {log_csv_path}", status="error"),400

    if not os.path.exists(queries_json_path):
        return make_response( message=f"Queries file not found at {queries_json_path}", status="error"),400

    if "Clinical Accuracy" not in data or "Evidence Basis" not in data or "Safety/Risk" not in data:
        return make_response( message=f"Missing review scores. Please provide 'Clinical Accuracy', 'Evidence Basis', and 'Safety/Risk' scores.", status="error"),400
    
    save_blinded_review_log(log_csv_path, data)
    return make_response(message=f"Blinded review saved successfully.")

@app.route("/health", methods=["GET"])
def health_check():
    """health check endpoint"""
    return make_response({"status": "ok"}, message="Health check passed")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    app.run(debug=debug, host=host, port=port)