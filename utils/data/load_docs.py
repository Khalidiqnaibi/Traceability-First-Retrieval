from pipeline.retrieval import ClinicalDocument
import sqlite3,logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_from_sqlite(db_path: str):
    """Bridge function to convert SQLite rows back into Pipeline objects."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor = conn.cursor()
        
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT,
            source TEXT,
            journal_tier TEXT,
            evidence_level INTEGER,
            publication_year INTEGER,
            domain TEXT
        )
    ''')

    cursor.execute("SELECT chunk_id, text, source, journal_tier, evidence_level, publication_year, domain FROM clinical_chunks")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.error("Database is empty. No documents loaded.")
        return []

    return [
        ClinicalDocument(
            chunk_id=row[0],
            text=row[1],
            source=row[2],
            journal_tier=row[3],
            evidence_level=row[4],
            publication_year=row[5],
            domain=row[6]
        ) for row in rows
    ]