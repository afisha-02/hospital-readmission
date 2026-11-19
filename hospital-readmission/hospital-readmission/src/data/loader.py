"""
Data loader for the UCI Diabetes 130-US Hospitals (1999-2008) dataset.
Handles missing token normalization ('?' -> NaN) and IDS mappings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

def parse_ids_mapping(mapping_path: Path) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """Parse the multi-table IDS_mapping.csv into 3 distinct dictionaries."""
    if not mapping_path.exists():
        logger.warning(f"IDS mapping file not found at {mapping_path}. Using fallback labels.")
        return {}, {}, {}
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        
    admission_types = {}
    discharge_dispositions = {}
    admission_sources = {}
    
    current_section = None
    for line in lines:
        if not line or line.startswith(","):
            continue
        if "admission_type_id" in line:
            current_section = "admission_type"
            continue
        elif "discharge_disposition_id" in line:
            current_section = "discharge_disposition"
            continue
        elif "admission_source_id" in line:
            current_section = "admission_source"
            continue
            
        parts = line.split(",", 1)
        if len(parts) == 2:
            try:
                code_id = int(parts[0].strip())
                desc = parts[1].strip().strip('"')
                if current_section == "admission_type":
                    admission_types[code_id] = desc
                elif current_section == "discharge_disposition":
                    discharge_dispositions[code_id] = desc
                elif current_section == "admission_source":
                    admission_sources[code_id] = desc
            except ValueError:
                continue
                
    return admission_types, discharge_dispositions, admission_sources

def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw diabetic data with '?' treated as NaN.
    """
    config = load_config()
    if data_path is None:
        csv_path = resolve_path(config["paths"]["raw_data"])
    else:
        csv_path = resolve_path(data_path)
        
    if not csv_path.exists():
        from src.data.download_dataset import download_and_extract
        logger.info(f"Data file {csv_path} not found. Attempting download...")
        download_and_extract()
        
    logger.info(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path, na_values=["?", "None", "NULL"], low_memory=False)
    logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    return df

def load_mapped_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load data and map integer codes for admission and discharge to descriptive labels.
    """
    config = load_config()
    df = load_raw_data(data_path)
    ids_path = resolve_path(config["paths"]["ids_mapping"])
    
    adm_types, disch_disps, adm_sources = parse_ids_mapping(ids_path)
    
    # Map descriptions with clean fallbacks
    df["admission_type_desc"] = df["admission_type_id"].map(adm_types).fillna("Other/Unknown")
    df["discharge_disposition_desc"] = df["discharge_disposition_id"].map(disch_disps).fillna("Other/Unknown")
    df["admission_source_desc"] = df["admission_source_id"].map(adm_sources).fillna("Other/Unknown")
    
    # Group rare discharge descriptions for modeling stability
    def clean_discharge(desc):
        desc_l = str(desc).lower()
        if "home" in desc_l and "hospice" not in desc_l:
            return "Home"
        elif "snf" in desc_l or "nursing" in desc_l:
            return "SNF/Nursing"
        elif "rehab" in desc_l:
            return "Rehabilitation"
        elif "hospice" in desc_l:
            return "Hospice"
        elif "expired" in desc_l:
            return "Expired"
        elif "short term hospital" in desc_l:
            return "Short Term Hospital"
        elif "ama" in desc_l:
            return "Left AMA"
        else:
            return "Other"

    df["discharge_disposition_desc"] = df["discharge_disposition_desc"].apply(clean_discharge)

    def clean_adm_type(desc):
        desc_l = str(desc).lower()
        if "emergency" in desc_l:
            return "Emergency"
        elif "urgent" in desc_l:
            return "Urgent"
        elif "elective" in desc_l:
            return "Elective"
        else:
            return "Other"

    df["admission_type_desc"] = df["admission_type_desc"].apply(clean_adm_type)

    def clean_adm_source(desc):
        desc_l = str(desc).lower()
        if "emergency" in desc_l:
            return "Emergency Room"
        elif "referral" in desc_l or "physician" in desc_l:
            return "Physician/Clinic Referral"
        elif "transfer" in desc_l:
            return "Transfer from Hospital/Facility"
        else:
            return "Other"

    df["admission_source_desc"] = df["admission_source_desc"].apply(clean_adm_source)

    return df
