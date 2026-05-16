import csv
import os
from datetime import datetime

class AuditTrail:
    def __init__(self, log_path: str = "route_audit_log.csv"):
        self.log_path = log_path
        self.headers = ["timestamp", "action", "query", "latency" , "status"]
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_event(self, action: str, query: str = "", latency: float = 0.0, status: str = "success"):
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action,
                "query": query,
                "latency": latency,
                "status": status,
            })