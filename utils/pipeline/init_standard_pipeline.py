from utils.data.load_docs import load_from_sqlite
from pipeline.standard_pipeline import Pipeline


def initialize_standard_pipeline(db_path:str,domain_data_path:str,model:str,api_key:str) -> Pipeline:
    """Helper to (re)load the corpus and (re)build search indices"""
    print("Initializing/Reloading Standard Pipeline...")
    corpus = load_from_sqlite(db_path)
    
    if not corpus:
        print("Corpus is empty. Pipeline will wait for ingestion")
        pipeline = None
    else:
        pipeline = Pipeline(
            corpus,
            api_key=api_key,
            domain_data_path=domain_data_path,
            model=model
        )
        print(f"Pipeline ready with {len(corpus)} documents")

    return pipeline