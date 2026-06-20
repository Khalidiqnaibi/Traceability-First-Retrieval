import json
import random
import pandas as pd

def get_sampled_queries(queries_json_path: str, n_total: int = 30, seed: int = 42) -> list:
    """
    Stratified sample across ablation_dimension × expected_twr_advantage, 
    while dynamically enforcing domain quotas to guarantee balanced doctor workbooks.
    """
    random.seed(seed)

    with open(queries_json_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    meta_df = pd.DataFrame(queries_data)

    # Only queries that will land in a doctor workbook
    SPECIALTY_DOMAINS = ['Orthopaedics', 'Rehabilitation', 'Pain & Anesthesiology']
    pool = meta_df[meta_df['domain'].isin(SPECIALTY_DOMAINS)].copy()

    # Build strata: 4 ablation dims × 2 advantage flags = 8 cells
    strata = list(pool.groupby(['ablation_dimension', 'expected_twr_advantage']))
    strata.sort(key=lambda x: str(x[0]))   # deterministic order
    n_strata = len(strata)

    base_per_cell = n_total // n_strata
    remainder     = n_total % n_strata

    # Target 10 queries per domain to ensure 20/20 workbook splits
    target_per_domain = n_total // len(SPECIALTY_DOMAINS)
    domain_counts = {d: 0 for d in SPECIALTY_DOMAINS}

    sampled_queries = []
    print(f"Stratified sampling — {n_total} queries across {n_strata} cells (seed={seed}):")

    for i, (key, group) in enumerate(strata):
        n = base_per_cell + (1 if i < remainder else 0)
        available_total = len(group)
        n_to_pick = min(n, available_total)

        # Shuffle the group deterministically so our iteration is randomized
        group_shuffled = group.sample(frac=1, random_state=seed+i)
        chosen_for_cell = []

        # Quota-aware picking
        for _ in range(n_to_pick):
            # 1. Calculate how many more we need to hit our target for each domain
            needed = {d: target_per_domain - domain_counts[d] for d in SPECIALTY_DOMAINS}

            # 2. Find out which domains are actually left in this cell
            available_rows = group_shuffled[~group_shuffled['query'].isin(chosen_for_cell)]
            available_domains = available_rows['domain'].unique()

            # 3. Filter our needs to only what's possible to pick right now
            valid_needed = {d: needed[d] for d in available_domains}
            
            if not valid_needed:
                break # Failsafe if cell runs dry

            # 4. Find the domain that needs representation the most
            max_need = max(valid_needed.values())
            best_domains = [d for d, need in valid_needed.items() if need == max_need]

            # Deterministic random choice breaks ties (e.g., all need 10 initially)
            best_domain = random.choice(best_domains)

            # 5. Pick the first available query from that prioritized domain
            q = available_rows[available_rows['domain'] == best_domain].iloc[0]['query']
            chosen_for_cell.append(q)
            domain_counts[best_domain] += 1

        sampled_queries.extend(chosen_for_cell)
        dim, adv = key
        print(f"   [{dim} | advantage={adv}]  {len(chosen_for_cell)} / {available_total} available")

    # Sanity-check balance before returning
    result_df = meta_df[meta_df['query'].isin(sampled_queries)]
    adv_counts = result_df['expected_twr_advantage'].value_counts()
    dim_counts  = result_df['ablation_dimension'].value_counts()
    dom_counts  = result_df['domain'].value_counts()
    
    print(f"\nBalance check ({len(sampled_queries)} total):")
    print(f"  advantage True/False : {adv_counts.get(True, 0)} / {adv_counts.get(False, 0)}")
    print(f"  ablation dims        : { dict(dim_counts) }")
    print(f"  domains              : { dict(dom_counts) }")
    
    # Prove the workbook load logic
    ortho_load = dom_counts.get('Orthopaedics', 0) + dom_counts.get('Rehabilitation', 0)
    pain_load = dom_counts.get('Pain & Anesthesiology', 0) + dom_counts.get('Rehabilitation', 0)
    print(f"\nWorkbook Loads:")
    print(f"  Orthopedist (Ortho + Rehab)         : {ortho_load}")
    print(f"  Anesthesiologist (Pain + Rehab)     : {pain_load}")

    return sampled_queries