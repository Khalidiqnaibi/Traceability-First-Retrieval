
from typing import List, Tuple


# Domain keyword map (ordered: specific → general)
# Order matters: the first match wins, so put narrow terms before broad ones.
DOMAIN_KEYWORDS: List[Tuple[str, str]] = [
    # ── Psychiatry / Mental Health ──────────────────────────────────────────
    ("bipolar",           "Mental Disorders (Psychiatry)"),
    ("mood disorder",     "Mental Disorders (Psychiatry)"),
    ("depressive",        "Mental Disorders (Psychiatry)"),
    ("schizophreni",      "Mental Disorders (Psychiatry)"),
    ("anxiety disorder",  "Mental Disorders (Psychiatry)"),
    ("psychosis",         "Mental Disorders (Psychiatry)"),
    ("psychiat",          "Mental Disorders (Psychiatry)"),
    ("mental",            "Mental Disorders (Psychiatry)"),

    # ── Skin / Connective Tissue / Rheumatology ──────────────────────────────
    ("psoriasis",         "Skin and Connective Tissue Diseases (Dermatology)"),
    ("psoriatic",         "Skin and Connective Tissue Diseases (Dermatology)"),
    ("vitiligo",          "Skin and Connective Tissue Diseases (Dermatology)"),
    ("lichen planus",     "Skin and Connective Tissue Diseases (Dermatology)"),
    ("pemphigoid",        "Skin and Connective Tissue Diseases (Dermatology)"),
    ("pemphigus",         "Skin and Connective Tissue Diseases (Dermatology)"),
    ("alopecia",          "Skin and Connective Tissue Diseases (Dermatology)"),
    ("effluvium",         "Skin and Connective Tissue Diseases (Dermatology)"),
    ("seborrheic",        "Skin and Connective Tissue Diseases (Dermatology)"),
    ("demodicosis",       "Skin and Connective Tissue Diseases (Dermatology)"),
    ("bullous",           "Skin and Connective Tissue Diseases (Dermatology)"),
    ("urticaria",         "Skin and Connective Tissue Diseases (Dermatology)"),
    ("eczema",            "Skin and Connective Tissue Diseases (Dermatology)"),
    ("atopic dermatit",   "Skin and Connective Tissue Diseases (Dermatology)"),
    ("scleroderma",       "Skin and Connective Tissue Diseases (Dermatology)"),
    ("ankylosing spondyl","Skin and Connective Tissue Diseases (Dermatology)"),
    ("lupus erythematosus","Skin and Connective Tissue Diseases (Dermatology)"),
    ("rheumatoid arthrit","Skin and Connective Tissue Diseases (Dermatology)"),
    ("dermatol",          "Skin and Connective Tissue Diseases (Dermatology)"),
    ("skin",              "Skin and Connective Tissue Diseases (Dermatology)"),

    # ── Immune System ────────────────────────────────────────────────────────
    ("autoimmune",        "Immune System Diseases"),
    ("hla-",              "Immune System Diseases"),
    ("immune-mediated",   "Immune System Diseases"),
    ("immunodeficien",    "Immune System Diseases"),
    ("immun",             "Immune System Diseases"),

    # ── Infections ───────────────────────────────────────────────────────────
    ("herpes zoster",     "Infections"),
    ("postherpetic",      "Infections"),
    ("leprosy",           "Infections"),
    ("tuberculosis",      "Infections"),
    ("hiv",               "Infections"),
    ("sars-cov",          "Infections"),
    ("covid",             "Infections"),
    ("infecti",           "Infections"),
    ("virus",             "Infections"),
    ("bacteria",          "Infections"),
    ("sepsis",            "Infections"),
    ("antimicrobial",     "Infections"),
    ("antiviral",         "Infections"),

    # ── Oncology ─────────────────────────────────────────────────────────────
    ("neoplasm",          "Neoplasms (Oncology)"),
    ("cancer",            "Neoplasms (Oncology)"),
    ("tumor",             "Neoplasms (Oncology)"),
    ("tumour",            "Neoplasms (Oncology)"),
    ("malignan",          "Neoplasms (Oncology)"),
    ("carcinoma",         "Neoplasms (Oncology)"),
    ("lymphoma",          "Neoplasms (Oncology)"),
    ("leukemia",          "Neoplasms (Oncology)"),
    ("leukaemia",         "Neoplasms (Oncology)"),
    ("melanoma",          "Neoplasms (Oncology)"),
    ("oncol",             "Neoplasms (Oncology)"),

    # ── Gastroenterology ─────────────────────────────────────────────────────
    ("colostomy",         "Digestive System Diseases (Gastroenterology)"),
    ("colorectal",        "Digestive System Diseases (Gastroenterology)"),
    ("colitis",           "Digestive System Diseases (Gastroenterology)"),
    ("gastro",            "Digestive System Diseases (Gastroenterology)"),
    ("liver",             "Digestive System Diseases (Gastroenterology)"),
    ("hepatic",           "Digestive System Diseases (Gastroenterology)"),
    ("hepatitis",         "Digestive System Diseases (Gastroenterology)"),
    ("intestin",          "Digestive System Diseases (Gastroenterology)"),
    ("crohn",             "Digestive System Diseases (Gastroenterology)"),
    ("bowel",             "Digestive System Diseases (Gastroenterology)"),
    ("pancrea",           "Digestive System Diseases (Gastroenterology)"),

    # ── Respiratory ──────────────────────────────────────────────────────────
    ("respirat",          "Respiratory Tract Diseases"),
    ("pulmonary",         "Respiratory Tract Diseases"),
    ("pneumonia",         "Respiratory Tract Diseases"),
    ("asthma",            "Respiratory Tract Diseases"),
    ("copd",              "Respiratory Tract Diseases"),
    ("lung",              "Respiratory Tract Diseases"),
    ("bronch",            "Respiratory Tract Diseases"),
    ("rhinosinusit",      "Respiratory Tract Diseases"),

    # ── Neurology ────────────────────────────────────────────────────────────
    ("neuralgia",         "Nervous System Diseases (Neurology)"),
    ("neuropath",         "Nervous System Diseases (Neurology)"),
    ("parkinson",         "Nervous System Diseases (Neurology)"),
    ("alzheimer",         "Nervous System Diseases (Neurology)"),
    ("epilep",            "Nervous System Diseases (Neurology)"),
    ("dementia",          "Nervous System Diseases (Neurology)"),
    ("migrain",           "Nervous System Diseases (Neurology)"),
    ("neurol",            "Nervous System Diseases (Neurology)"),
    ("brain",             "Nervous System Diseases (Neurology)"),
    ("stroke",            "Nervous System Diseases (Neurology)"),
    ("cognitive",         "Nervous System Diseases (Neurology)"),
    ("spinal cord",       "Nervous System Diseases (Neurology)"),

    # ── Cardiovascular ───────────────────────────────────────────────────────
    ("antiplatelet",      "Cardiovascular Diseases"),
    ("anticoagul",        "Cardiovascular Diseases"),
    ("thrombosis",        "Cardiovascular Diseases"),
    ("thromboemboli",     "Cardiovascular Diseases"),
    ("cardiovasc",        "Cardiovascular Diseases"),
    ("myocardial",        "Cardiovascular Diseases"),
    ("atheroscler",       "Cardiovascular Diseases"),
    ("cholesterol",       "Cardiovascular Diseases"),
    ("venous",            "Cardiovascular Diseases"),
    ("hypertens",         "Cardiovascular Diseases"),
    ("arrhythmia",        "Cardiovascular Diseases"),
    ("coronary",          "Cardiovascular Diseases"),
    ("heart failure",     "Cardiovascular Diseases"),
    ("atrial fibrill",    "Cardiovascular Diseases"),
    ("heart",             "Cardiovascular Diseases"),
    ("fontan",            "Cardiovascular Diseases"),

    # ── Hematology ───────────────────────────────────────────────────────────
    ("hematol",           "Hemic and Lymphatic Diseases (Hematology)"),
    ("haematol",          "Hemic and Lymphatic Diseases (Hematology)"),
    ("anemia",            "Hemic and Lymphatic Diseases (Hematology)"),
    ("anaemia",           "Hemic and Lymphatic Diseases (Hematology)"),
    ("coagulat",          "Hemic and Lymphatic Diseases (Hematology)"),
    ("platelet",          "Hemic and Lymphatic Diseases (Hematology)"),
    ("hemoglobin",        "Hemic and Lymphatic Diseases (Hematology)"),
    ("blood",             "Hemic and Lymphatic Diseases (Hematology)"),

    # ── Endocrine ────────────────────────────────────────────────────────────
    ("thyroid",           "Endocrine System Diseases"),
    ("diabetes",          "Endocrine System Diseases"),
    ("diabetic",          "Endocrine System Diseases"),
    ("insulin",           "Endocrine System Diseases"),
    ("endocrin",          "Endocrine System Diseases"),
    ("diabet",            "Endocrine System Diseases"),
    ("adrenal",           "Endocrine System Diseases"),
    ("pituitary",         "Endocrine System Diseases"),

    # ── Metabolic / Nutritional ───────────────────────────────────────────────
    ("adiposity",         "Nutritional and Metabolic Diseases"),
    ("metabol",           "Nutritional and Metabolic Diseases"),
    ("obesity",           "Nutritional and Metabolic Diseases"),
    ("overweight",        "Nutritional and Metabolic Diseases"),
    ("lipid",             "Nutritional and Metabolic Diseases"),
    ("vitamin",           "Nutritional and Metabolic Diseases"),
    ("nutrition",         "Nutritional and Metabolic Diseases"),

    # ── Pediatrics ───────────────────────────────────────────────────────────
    ("pediatr",           "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)"),
    ("paediatr",          "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)"),
    ("neonatal",          "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)"),
    ("congenital",        "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)"),
    ("children",          "Congenital, Hereditary, and Neonatal Diseases (Pediatrics)"),

    # ── Veterinary ───────────────────────────────────────────────────────────
    ("feline",            "Animal Diseases (Veterinary)"),
    ("canine",            "Animal Diseases (Veterinary)"),
    ("veterinary",        "Animal Diseases (Veterinary)"),
]
