import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import ast

from utils.data.sample import get_sampled_queries

load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def generate_clinical_answer(query: str, context_text: str) -> str:
    SYSTEM_PROMPT = """You are a clinical decision-support assistant for doctors and medical students.
You receive retrieved medical literature as numbered documents: Doc1, Doc2, etc.

ANSWER STYLE:
- Be direct and clinically precise. No preamble, no filler.
- Condense to the minimum words that preserve clinical accuracy.
- Use medical terminology appropriate for a physician audience.
- Bullet points for differentials, drug options, or step-wise reasoning.
- Prose for mechanistic explanations.

USING THE DOCUMENTS:
- Base your answer exclusively on the provided documents.
- At the end of your answer, output two lines:

  Used: Doc2, Doc5
  Not used: Doc1, Doc3, Doc4 — not relevant to the query

- If a document partially contributed, include it in Used.
- If no document addresses part of the question, state exactly what is missing:

  Gap: No retrieved evidence on pediatric dosing for this indication.

CONSTRAINTS:
- Never fabricate a clinical claim not present in the documents.
- If the documents conflict, state the conflict — do not resolve it yourself.
- For dosing, contraindications, or procedural thresholds: flag with [weak] if only weak evidence is available.
- If the documents are collectively insufficient to answer safely, say so plainly before attempting any answer.
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
        
        # Symmetric blinding — intentionally unseeded so A/B assignment is
        # unpredictable to anyone reading the code before unblinding.
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
    run_blinded_llm_pass("data_in/queries.json", "data_out/pipeline_audit_log.csv", "data_out/blinded_clinical_review.csv")