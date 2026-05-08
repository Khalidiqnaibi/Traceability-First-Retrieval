from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from retrieval import TFRPipeline, ClinicalDocument

load_dotenv()

# Initialize the retrieval pipeline
pipeline = TFRPipeline()

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query")
    
    # Perform the query
    results = pipeline.query(query_text)
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=bool(os.getenv("DEBUG")), host=os.getenv("HOST"), port=int(os.getenv("PORT")))