import csv
import os
from datetime import datetime

class PipelineAudit:
    def __init__(self, log_path: str = "pipeline_audit_log.csv"):
        self.log_path = log_path
        self.headers = [
            "timestamp",
            "query",  
            "sparse_results",
            "dense_results",
            "RRF / TWR results",
            "results_count",
            "results", 
            "latency" ,
            "pipeline", 
            "status"
            ]
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_event(
            self,
            query: str = "", 
            sparse_results: list = [], 
            dense_results: list = [],  
            rrf_twr_results: list = [],
            results_count: int = 0, 
            results: list = [],
            latency: float = 0.0, 
            pipeline: str = "standard", 
            status: str = "success"
            ):
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "sparse_results": sparse_results,
                "dense_results": dense_results,
                "RRF / TWR results": rrf_twr_results,
                "results_count": results_count,
                "results": results,
                "latency": latency,
                "pipeline": pipeline,
                "status": status,
            })