import json
import requests
import sys
import os
import dotenv

dotenv.load_dotenv()

path = os.getenv("SEED_QUERIES_PATH", "./data/seed_queries.json")
host = os.getenv("API_HOST", "127.0.0.1")
port = os.getenv("API_PORT", "8000")

def seed_database(json_path, api_url=f"http://{host}:{port}/seed"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for i, item in enumerate(data):
            response = requests.post(api_url, json=item)
            if response.status_code != 200:
                print(f"Error seeding item {i}: {response.status_code} - {response.text}")
            else:
                print(f"Successfully seeded item {i+1}/{len(data)}")
                
    except Exception as e:
        print(f"Critical error during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    seed_database(path)