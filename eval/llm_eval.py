import pandas as pd
import random
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import ast
load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

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
        # Sample 10 queries
        sampled_subset = random.sample(domain_pool, min(10, len(domain_pool)))
        sampled_queries.extend(sampled_subset)
        print(f"   -> Sampled {len(sampled_subset)} queries from '{domain}'")
        
    return sampled_queries

def generate_clinical_answer(query: str, context_text: str) -> str:
    SYSTEM_PROMPT = """You are a clinical assistant.  
You will be given a clinical query and a set of retrieved documents.  

Your task is to produce a **direct, evidence‑based answer** to the query using **only the information in the retrieved documents**.  
Do not add external knowledge. Answer the query as clearly and concisely as possible. 

If the documents do not answer the query, state that clearly (e.g., "The retrieved documents do not address this query").  

Keep the answer focused – typically 2‑5 sentences.  
"""
    
    USER_PROMPT = f"""## Clinical Query
    {query}
    
    ## Retrieved Context Documents
    
    {context_text}
    
   """
    
    response = client.chat.completions.create(
        model="deepseek-chat", # Use V3
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

def run_blinded_llm_pass(queries_json_path: str, log_csv_path: str, output_csv_path: str):
    target_queries = get_sampled_queries(queries_json_path)
    
    print("Loading pipeline audit log...")
    audit_df = pd.read_csv(log_csv_path)
    
    # Filter log down to only include the sampled queries
    filtered_df = audit_df[audit_df['query'].isin(target_queries)]
    unique_queries = filtered_df['query'].unique()
    
    results = []
    print(f"Generating LLM passes for {len(unique_queries)} queries...")
    
    for idx, query in enumerate(unique_queries, 1):
        query_data = filtered_df[filtered_df['query'] == query]
        
        rrf_rows = query_data[query_data['pipeline'] == 'Standard']
        if not rrf_rows.empty and pd.notna(rrf_rows['results'].iloc[0]):
            try:
                rrf_docs = ast.literal_eval(rrf_rows['results'].iloc[0])
                rrf_context = "\n\n".join([f"Doc {i+1}: {doc['text']}" for i, doc in enumerate(rrf_docs[:5]) if 'text' in doc])
            except Exception as e:
                print(f"   [error] Parsing error on Standard row for query {idx}: {e}")
                rrf_context = "Error parsing retrieved context."
        else:
            rrf_context = "No context found."

        twr_rows = query_data[query_data['pipeline'] == 'TWR']
        if not twr_rows.empty and pd.notna(twr_rows['results'].iloc[0]):
            try:
                twr_docs = ast.literal_eval(twr_rows['results'].iloc[0])
                twr_context = "\n\n".join([f"Doc {i+1}: {doc['text']}" for i, doc in enumerate(twr_docs[:5]) if 'text' in doc])
            except Exception as e:
                print(f"   [error] Parsing error on TWR row for query {idx}: {e}")
                twr_context = "Error parsing retrieved context."
        else:
            twr_context = "No context found."
            
        print(f" [{idx}/{len(unique_queries)}] Generating responses for: {query[:60]}...")
        ans_rrf = generate_clinical_answer(query, rrf_context)
        ans_twr = generate_clinical_answer(query, twr_context)
        
        # Symmetric Blinding Logic
        is_rrf_a = random.choice([True, False])
        ans_A = ans_rrf if is_rrf_a else ans_twr
        ans_B = ans_twr if is_rrf_a else ans_rrf
        secret_key = "A=RRF, B=TWR" if is_rrf_a else "A=TWR, B=RRF"
        
        results.append({
            "Query_ID": f"CLIN-EVAL-{idx:02d}",
            "Query": query,
            "Response A": ans_A,
            "Response B": ans_B,
            "Secret_Key_Mapping": secret_key
        })
        
    # Export evaluation
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n Target generation complete! Review file saved to: {output_csv_path}")

if __name__ == "__main__":
    run_blinded_llm_pass("data/queries.json", "logs\pipeline_audit_log.csv", "logs/blinded_clinical_review.csv")