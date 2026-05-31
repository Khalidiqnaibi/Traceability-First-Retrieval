import pandas as pd
import random
import json
import ast

def get_sampled_queries(queries_json_path: str) -> list:
    """Loads metadata, identifies top 3 domains, and samples 10 queries from each."""
    with open(queries_json_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    meta_df = pd.DataFrame(queries_data)
    
    top_domains = meta_df['domain'].value_counts().nlargest(3).index.tolist()
    print(f"Identified Top 3 Domains for Validation: {top_domains}")
    
    sampled_queries = []
    for domain in top_domains:
        domain_pool = meta_df[meta_df['domain'] == domain]['query'].tolist()
        sampled_subset = random.sample(domain_pool, min(10, len(domain_pool)))
        sampled_queries.extend(sampled_subset)
        print(f"   -> Sampled {len(sampled_subset)} queries from '{domain}'")
        
    return sampled_queries


def save_documents_to_csv(docs_data, csv_path="documents.csv"):
    """
    Save lists of document dictionaries (one per query) into CSV file.
    
    Parameters:
        docs_data: list of dicts, each with keys: 'query', 'documents' (list of doc dicts)
        csv_path: output file path for RRF documents
    """
    def flatten_docs(data):
        rows = []
        for entry in data:
            query = entry['query']
            for rank, doc in enumerate(entry['documents'], start=1):
                row = {
                    'query': query,
                    'rank': rank,
                    'doc_text': doc.get('text', '')
                }
                rows.append(row)
        return rows
    
    rows = flatten_docs(docs_data)
    
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved {len(rows)} document rows to {csv_path}")
    else:
        print("No documents to save.")
        

def save_RRF_TWR_results(queries_json_path: str, log_csv_path: str):
    target_queries = get_sampled_queries(queries_json_path)
    
    print("Loading pipeline audit log...")
    audit_df = pd.read_csv(log_csv_path)
    
    # Filter log down to only include the sampled queries
    filtered_df = audit_df[audit_df['query'].isin(target_queries)]
    unique_queries = filtered_df['query'].unique()
    
    rrf_export_data = []   # list of {'query': q, 'documents': list_of_doc_dicts}
    twr_export_data = []
    
    print(f"Saving results for {len(unique_queries)} queries...")
    
    for idx, query in enumerate(unique_queries, 1):
        query_data = filtered_df[filtered_df['query'] == query]
        
        rrf_rows = query_data[query_data['pipeline'] == 'Standard']
        rrf_docs = []
        rrf_context = "No context found."
        if not rrf_rows.empty and pd.notna(rrf_rows['results'].iloc[0]):
            try:
                rrf_docs = ast.literal_eval(rrf_rows['results'].iloc[0])
                # Build text context for the .txt file (first 5 docs)
                rrf_context = "\n\n".join([f"Doc {i+1}: {doc['text']}" for i, doc in enumerate(rrf_docs[:5]) if 'text' in doc])
            except Exception as e:
                print(f"   [error] Parsing error on Standard row for query {idx}: {e}")
                rrf_context = "Error parsing retrieved context."
        
        # Save for CSV (all documents, not just first 5)
        rrf_export_data.append({'query': query, 'documents': rrf_docs})
        
        twr_rows = query_data[query_data['pipeline'] == 'TWR']
        twr_docs = []
        twr_context = "No context found."
        if not twr_rows.empty and pd.notna(twr_rows['results'].iloc[0]):
            try:
                twr_docs = ast.literal_eval(twr_rows['results'].iloc[0])
                twr_context = "\n\n".join([f"Doc {i+1}: {doc['text']}" for i, doc in enumerate(twr_docs[:5]) if 'text' in doc])
            except Exception as e:
                print(f"   [error] Parsing error on TWR row for query {idx}: {e}")
                twr_context = "Error parsing retrieved context."
        
        twr_export_data.append({'query': query, 'documents': twr_docs})
        
        print(f" [{idx}/{len(unique_queries)}] saving responses for: {query[:60]}...")
        # save_to_file(query + ": " + rrf_context, "RRF_res.txt")
        # save_to_file(query + ": " + twr_context, "TWR_res.txt")
    
    save_documents_to_csv(rrf_export_data,csv_path="logs/rrf_documents.csv")
    save_documents_to_csv(twr_export_data,csv_path="logs/twr_documents.csv")

def save_to_file(text, name):
    with open(name, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")

if __name__ == "__main__":
    save_RRF_TWR_results("data/queries.json", "logs/pipeline_audit_log.csv")