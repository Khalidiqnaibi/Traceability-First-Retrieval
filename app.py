from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from src.retrieval import TFRPipeline, ClinicalDocument
from utils.ingestion_pipeline import TFRDataPreprocessor
from utils.make_response import make_response

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DOMAIN_DATA_PATH = os.environ.get("DOMAIN_DATA_PATH")
DOC_DB_PATH = os.environ.get("DOC_DB_PATH")

processor = TFRDataPreprocessor(DOMAIN_DATA_PATH)
pipeline = TFRPipeline(api_key=API_KEY)

app = Flask(__name__)


@app.route("/ingest", methods=["POST"])
def query():
    data = request.get_json()
    path = data.get("path")
    
    result = processor.parse_pubmed_xml(path)
    processor.export_to_sqlite(result,DOC_DB_PATH)
    
    return make_response(result,message=f"ingested document with path: {path}")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    
    # Perform the query
    results = pipeline.query(query_text)
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=bool(os.getenv("DEBUG")), host=os.getenv("HOST"), port=int(os.getenv("PORT")))