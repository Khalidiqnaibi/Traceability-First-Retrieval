import logging  
from utils.data.load_docs import load_from_sqlite
from pipeline.retrieval import TFRPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_pipeline(db_path:str,api_key:str,domain_data_path:str):
    """Helper to (re)load the corpus and (re)build search indices."""
    global pipeline
    logger.info("Initializing/Reloading TFR Pipeline...")
    corpus = load_from_sqlite(db_path)
    
    if not corpus:
        logger.warning("Corpus is empty. Pipeline will wait for ingestion.")
        pipeline = None
    else:
        pipeline = TFRPipeline(
            corpus,
            api_key=api_key,
            domain_data_path=domain_data_path
        )
        logger.info(f"Pipeline ready with {len(corpus)} documents.")