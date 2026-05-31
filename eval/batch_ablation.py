import os
import json
import time


from utils.api.make_response import make_response



def run_batch_ablation(queries_file_path, tfr_pipeline, standard_pipeline, audit):
    
    with open(queries_file_path, "r", encoding="utf-8") as f:
        queries_list = json.load(f)

    summary_metrics = []
    start_time = time.time()

    for item in queries_list:
        query_id = item.get("id")
        query_text = item.get("query")
        dimension = item.get("ablation_dimension")
        expected_advantage = item.get("expected_twr_advantage")

        try:
            tfr_results = tfr_pipeline.retrieve(query=query_text)
            standard_results = standard_pipeline.retrieve(query=query_text)
        except Exception as e:
            print(f"Ablation query failed: {e}")
            audit.log_event(action="batch_ablation", query=query_text, status=f"Ablation Retrieval error: {str(e)}")
            return make_response( message=f"Ablation Retrieval error: {str(e)}", status="error"),500

        summary_metrics.append({
            "id": query_id,
            "query": query_text,
            "ablation_dimension": dimension,
            "expected_twr_advantage": expected_advantage,
            "tfr_results": tfr_results,
            "standard_results": standard_results
        })

    audit.log_event(
        action="batch_ablation",
        query=f"Batch ablation over {len(queries_list)} queries",
        latency=time.time() - start_time
    )

    return summary_metrics


if __name__ == "__main__":
    # Example usage
    from pipeline.retrieval import TFRPipeline
    from pipeline.standard_pipeline import Pipeline as StandardPipeline
    from ..utils.api.audit import AuditTrail

    tfr_pipeline = TFRPipeline()
    standard_pipeline = StandardPipeline()
    audit = AuditTrail("logs/audit_log.csv")   

    results = run_batch_ablation("./data/queries.json", tfr_pipeline, standard_pipeline, audit)
    print(json.dumps(results, indent=2))