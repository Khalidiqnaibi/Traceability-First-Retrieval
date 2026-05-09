from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging

from utils.data.ingestion_pipeline import TFRDataPreprocessor
from utils.api.make_response import make_response
from utils.pipeline.init_pipline import initialize_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DOMAIN_DATA_PATH = os.environ.get("DOMAIN_DATA_PATH", "./data/domain.json")
DOC_DB_PATH = os.environ.get("DOC_DB_PATH", "document.db")
SJR_CSV_PATH = os.environ.get("SJR_CSV_PATH", "./data/scimagojr_2023.csv")

processor = TFRDataPreprocessor(DOMAIN_DATA_PATH, SJR_CSV_PATH)

# for hot-reloading
pipeline = None

initialize_pipeline(DOC_DB_PATH,API_KEY,DOMAIN_DATA_PATH)

app = Flask(__name__)

@app.route("/ingest", methods=["POST"])
def ingest_data():
    """Endpoint to process new PubMed XMLs into the SQLite DB."""
    data = request.get_json()
    doc_path = data.get("path")
    
    if not doc_path or not os.path.exists(doc_path):
        return make_response([], message="Error: Valid XML path required", status="error"),400

    try:
        result_chunks = processor.parse_pubmed_xml(doc_path)
        processor.export_to_sqlite(result_chunks, DOC_DB_PATH)
        
        # Re-initialize pipeline
        initialize_pipeline(DOC_DB_PATH,API_KEY,DOMAIN_DATA_PATH)
        
        return make_response(
            {"count": len(result_chunks)}, 
            message=f"Successfully ingested and indexed: {doc_path}"
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return make_response([], message=f"Ingestion failed: {str(e)}", status="error"),500

@app.route("/query", methods=["POST"])
def run_query():
    """Endpoint to perform Trust-Weighted Retrieval."""
    if pipeline is None:
        return make_response([], message="Pipeline not initialized. Please ingest data first.", status="error"),400

    data = request.get_json()
    query_text = data.get("query")
    
    if not query_text:
        return make_response([], message="Query text is required", status="error"),400

    try:
        results = pipeline.retrieve(query=query_text)
        return make_response(results, message="Retrieved with success")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return make_response([], message=f"Retrieval error: {str(e)}", status="error"),500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    app.run(debug=debug, host=host, port=port)