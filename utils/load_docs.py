from src.retrieval import ClinicalDocument
import sqlite3

def load_from_sqlite(db_path: str):
    """Bridge function to convert SQLite rows back into Pipeline objects."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, text, source, journal_tier, evidence_level, publication_year, domain FROM clinical_chunks")
    rows = cursor.fetchall()
    conn.close()

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