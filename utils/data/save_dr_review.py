import csv
import pandas as pd
import json


def save_blinded_review_log(blinded_log_path: str, review_data: dict):
    '''
    Saves the blinded review data to a CSV file.
        - If the file does not exist, it creates a new one with headers.
        - If the file already exists, it appends the new review data as a new row.
        - The review_data dictionary should contain keys: "Query", "Response A", "Response B", and "Secret_Key_Mapping".
    '''
    required_fields = ["Query", "Response A", "Response B", "Secret_Key_Mapping" ,"Clinical Accuracy", "Evidence Basis", "Safety/Risk"]

    # Check if all required fields are present in the review_data
    if not all(field in review_data for field in required_fields):
        raise ValueError(f"Review data must contain the following fields: {', '.join(required_fields)}")

    # Create a DataFrame from the review_data
    df = pd.DataFrame([review_data])

    # Check if the file exists to determine if we need to write headers
    try:
        with open(blinded_log_path, 'x', newline='', encoding='utf-8') as csvfile:
            df.to_csv(csvfile, index=False)
    except FileExistsError:
        with open(blinded_log_path, 'a', newline='', encoding='utf-8') as csvfile:
            df.to_csv(csvfile, mode='a', index=False, header=False)
