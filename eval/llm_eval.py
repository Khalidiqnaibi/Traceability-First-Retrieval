import pandas as pd
import random
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
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
    system_prompt = (
        "You are an expert medical AI assistant. Your sole purpose is to answer "
        "clinical queries strictly using the provided Context documents. \n\n"
        "RULES:\n"
        "1. Synthesize the provided evidence to answer the Question.\n"
        "2. You MUST NOT use your internal training data. If the Context lacks the answer, "
        "state: 'The provided evidence is insufficient.'\n"
        "3. Keep your answer concise, clinical, and directly address the query."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="deepseek-chat", # Use V3
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
    for query in unique_queries:
        query_data = filtered_df[filtered_df['query'] == query]
        
        # 1. Isolate top contexts for RRF
        rrf_data = query_data[query_data['pipeline'] == 'RRF'].sort_values('rank')
        rrf_context = "\n\n".join([f"Doc {i+1}: {row['text']}" for i, row in rrf_data.head(5).iterrows()])
        
        # 2. Isolate top contexts for TWR
        twr_data = query_data[query_data['pipeline'] == 'TWR'].sort_values('rank')
        twr_context = "\n\n".join([f"Doc {i+1}: {row['text']}" for i, row in twr_data.head(5).iterrows()])
        
        # 3. Generate answers
        print(f"Processing: {query[:50]}...")
        ans_rrf = generate_clinical_answer(query, rrf_context)
        ans_twr = generate_clinical_answer(query, twr_context)
        
        # 4. The Blinding Logic (Randomize A and B)
        is_rrf_a = random.choice([True, False])
        
        ans_A = ans_rrf if is_rrf_a else ans_twr
        ans_B = ans_twr if is_rrf_a else ans_rrf
        
        # Save the secret key for later de-blinding
        secret_key = "A=RRF, B=TWR" if is_rrf_a else "A=TWR, B=RRF"
        
        results.append({
            "Query": query,
            "Response A": ans_A,
            "Response B": ans_B,
            "Secret_Key": secret_key
        })
        
    # 5. Export
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Blinded evaluation sheet saved to {output_csv_path}")

if __name__ == "__main__":
    run_blinded_llm_pass("data/queries.json", "log/pipeline_audit_log.csv", "log/blinded_clinical_review.csv")