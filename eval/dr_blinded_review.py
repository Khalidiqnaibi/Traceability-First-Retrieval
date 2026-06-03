import pandas as pd
import json
from pathlib import Path


def generate_blinded_review_workbooks(path_to_blinded_log: str, path_to_queries_json: str):
    '''
    Generates blinded review workbooks for different medical specialties.
        - Loads the master blinded log and the queries metadata.
        - Maps domains back to the blinded dataframe.
        - Creates subsets for Orthopedics (Orthopaedics + Rehabilitation) and Anesthesiology (Pain & Anesthesiology + Rehabilitation).
        - Exports the subsets as separate CSV files for review.
        - Prints confirmation upon successful generation of workbooks.
        - Note: The exported workbooks will not contain any identifying information about the queries or their domains to ensure a blinded review process.
        - Output files: "Evaluation_Workbook_Orthopedics.csv" and "Evaluation_Workbook_Anesthesiology.csv"
    '''
    # 1. Load the master blinded log and the queries metadata
    blinded_df = pd.read_csv(path_to_blinded_log)
    with open(path_to_queries_json, "r") as f:
        queries_meta = json.load(f)

    # 2. Map domains back to the blinded dataframe
    domain_map = {q['query']: q['domain'] for q in queries_meta}
    blinded_df['Domain'] = blinded_df['Query'].map(domain_map)

    # 3. Create the Orthopedist's subset (Ortho + Rehab)
    ortho_df = blinded_df[blinded_df['Domain'].isin(['Orthopaedics', 'Rehabilitation'])]
    # Drop the secret key and domain before sending to the doctor
    ortho_export = ortho_df.drop(columns=['Secret_Key_Mapping', 'Domain'])
    ortho_export.to_csv("data_out/Evaluation_Workbook_Orthopedics.csv", index=False)

    # 4. Create the Anesthesiologist's subset (Pain + Rehab)
    pain_df = blinded_df[blinded_df['Domain'].isin(['Pain & Anesthesiology', 'Rehabilitation'])]
    # Drop the secret key and domain before sending to the doctor
    pain_export = pain_df.drop(columns=['Secret_Key_Mapping', 'Domain'])
    pain_export.to_csv("data_out/Evaluation_Workbook_Anesthesiology.csv", index=False)

    print("Workbooks generated successfully!")

if __name__ == "__main__":
    generate_blinded_review_workbooks("data_out/blinded_clinical_review.csv","data_in/queries.json")