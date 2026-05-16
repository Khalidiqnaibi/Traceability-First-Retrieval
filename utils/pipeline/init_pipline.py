from utils.data.load_docs import load_from_sqlite
from pipeline.retrieval import TFRPipeline


def initialize_pipeline(db_path:str) -> TFRPipeline:
    """Helper to (re)load the corpus and (re)build search indices"""
    print("Initializing/Reloading TFR Pipeline...")
    corpus = load_from_sqlite(db_path)
    
    if not corpus:
        print("Corpus is empty. Pipeline will wait for ingestion")
        pipeline = None
    else:
        pipeline = TFRPipeline(
            corpus
        )
        print(f"Pipeline ready with {len(corpus)} documents")

    return pipeline