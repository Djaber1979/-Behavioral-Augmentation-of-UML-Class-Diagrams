#!/usr/bin/env python3
"""
RQ1_SR_Statistical_Tests.py

Performs statistical analysis for Signature Richness (SR) components:
- SR1: Visibility Markers
- SR4: Return Types

Reads method details via MetricCache (from JSONs).
Saves statistical results to: reports/article/stats/
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter

# Statistical libraries
from scipy import stats
import pingouin as pg # For Cramér's V from chi2_matrix

# --- Import shared pipeline components from main3.py ---
# Ensure main3.py is in the same directory or accessible via PYTHONPATH
try:
    # Assuming JSON_INPUT_DIR, BASELINE_JSON_FNAME etc. are defined in main3.py and needed by MetricCache
    from main3 import MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME, GOLD_STANDARD_MAP_FNAME 
    print("Successfully imported MetricCache from main3.py")
except ImportError as e:
    print(f"Error: Could not import from main3.py: {e}")
    print("Please ensure main3.py is in the correct location and all its dependencies are met.")
    print("This script cannot run without MetricCache.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import from main3.py: {e}")
    exit()


# Configuration
REPORTS_DIR = Path("reports")
STATS_OUTPUT_DIR = REPORTS_DIR / "article" / "stats"
STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'model' # MetricCache.global_details_df uses 'model'
RUN_COL = 'run'     # MetricCache.global_details_df uses 'run'
SIGNATURE_COL = 'signature' # MetricCache.global_details_df uses 'signature'
VISIBILITY_JSON_COL = 'visibility_json' # From your main3.py modification

P_VALUE_THRESHOLD = 0.05

print(f"DEBUG: Script started. STATS_OUTPUT_DIR: {STATS_OUTPUT_DIR}")

def parse_method_signature_for_sr(signature_str: str, visibility_json_field: str | None) -> dict:
    """
    Parses a method signature string to extract components for SR.
    Prioritizes visibility_json_field for visibility.
    """
    parsed = {
        'visibility_symbol': None, # Store the raw symbol first
        'visibility_category': 'none_or_default', # More descriptive category
        'return_type': 'void' # Default
        # 'parameter_count': 0 # Can be added later for SR3
    }
    if not isinstance(signature_str, str) and not isinstance(visibility_json_field, str): # If both are bad
         return parsed

    # 1. Visibility (Priority to dedicated JSON field)
    raw_vis_symbol = None
    if isinstance(visibility_json_field, str):
        vis_cleaned = visibility_json_field.strip()
        if vis_cleaned in ['+', '-', '#', '~']: raw_vis_symbol = vis_cleaned
        elif vis_cleaned.lower() == "public": raw_vis_symbol = "+"
        elif vis_cleaned.lower() == "private": raw_vis_symbol = "-"
        elif vis_cleaned.lower() == "protected": raw_vis_symbol = "#"
    
    if raw_vis_symbol is None and isinstance(signature_str, str): # Fallback to signature parsing
        sig_norm = signature_str.strip()
        vis_match_sig = re.match(r"^\s*([+\-#~])\s*", sig_norm)
        if vis_match_sig:
            raw_vis_symbol = vis_match_sig.group(1)
            signature_str_proc = sig_norm[vis_match_sig.end():].strip()
        else: # Check for keywords if no symbol
            sig_lower = sig_norm.lower()
            if sig_lower.startswith("public "): raw_vis_symbol = "+"
            elif sig_lower.startswith("private "): raw_vis_symbol = "-"
            elif sig_lower.startswith("protected "): raw_vis_symbol = "#"
            signature_str_proc = sig_norm # Keep original if no keyword found at start
    elif isinstance(signature_str, str):
        signature_str_proc = signature_str.strip() # Use if visibility_json_field was definitive
    else: # If signature_str is also not a string (e.g. NaN)
        signature_str_proc = ""


    parsed['visibility_symbol'] = raw_vis_symbol
    vis_map = {'+': 'public', '-': 'private', '#': 'protected', '~': 'package'}
    parsed['visibility_category'] = vis_map.get(raw_vis_symbol, 'none_or_default')


    # 2. Return Type
    # Remove params part to simplify return type extraction
    sig_no_params = re.sub(r"\(.*\)", "", signature_str_proc).strip()

    # Try to find return type after potential method name (if a colon is present)
    # e.g. "methodName() : String" or "methodName : String" (if params already removed)
    name_return_match = re.match(r"^(.*?)\s*:\s*(.+)$", sig_no_params)
    if name_return_match:
        # potential_name_part = name_return_match.group(1).strip()
        return_type_part = name_return_match.group(2).strip()
        if return_type_part:
            parsed['return_type'] = return_type_part
    else:
        # Try to see if there's a type before the method name
        # e.g. "String methodName"
        parts = sig_no_params.split()
        if len(parts) > 1: # More than just a method name
            # Avoid common modifiers
            potential_type = " ".join(parts[:-1])
            if not any(mod in potential_type.lower() for mod in ['static', 'final', 'abstract', 'synchronized', 'native']):
                parsed['return_type'] = potential_type
        # If only one part or above heuristic fails, it defaults to 'void'

    # Clean up return type
    if not parsed['return_type'] or parsed['return_type'].lower() == 'void':
        parsed['return_type'] = 'void'
    else:
        parsed['return_type'] = re.sub(r"^(final|static|const)\s+", "", parsed['return_type']).strip()
        if not parsed['return_type']: # If it became empty after stripping modifiers
            parsed['return_type'] = 'void'

    return parsed


def extract_sr_features_from_cache(cache: MetricCache) -> pd.DataFrame | None:
    """
    Extracts per-method features (visibility, return type) from MetricCache.global_details_df.
    """
    print("DEBUG: Entering extract_sr_features_from_cache")
    if cache.global_details_df.empty:
        print("Error: MetricCache.global_details_df is empty. Cannot extract features.")
        return None

    df = cache.global_details_df.copy()
    
    # Ensure necessary columns for parsing exist
    if SIGNATURE_COL not in df.columns:
        print(f"Warning: '{SIGNATURE_COL}' column missing. Some features might be inaccurate.")
        df[SIGNATURE_COL] = "" # Add empty column to prevent crash
    if VISIBILITY_JSON_COL not in df.columns:
        print(f"Warning: '{VISIBILITY_JSON_COL}' column missing. Visibility parsing will rely solely on signature string.")
        df[VISIBILITY_JSON_COL] = None # Add empty column


    print(f"DEBUG: Parsing {len(df)} method entries for SR features...")
    
    # Apply parsing row-wise
    parsed_features_list = []
    for _, row in df.iterrows():
        sig = row.get(SIGNATURE_COL)
        vis_json = row.get(VISIBILITY_JSON_COL)
        parsed = parse_method_signature_for_sr(sig, vis_json)
        parsed_features_list.append(parsed)
    
    parsed_df = pd.DataFrame(parsed_features_list, index=df.index)

    df['visibility_category'] = parsed_df['visibility_category']
    df['return_type_str'] = parsed_df['return_type']
    df['is_non_void_return'] = df['return_type_str'].apply(
        lambda x: isinstance(x, str) and x.lower().strip() != 'void' and x.strip() != ''
    )
    
    print("DEBUG: Finished extracting SR features. Columns added: visibility_category, return_type_str, is_non_void_return.")
    # print("DEBUG: Sample of extracted SR features:\n", df[[SIGNATURE_COL, 'visibility_category', 'return_type_str', 'is_non_void_return']].head())
    # print("DEBUG: Visibility category counts:\n", df['visibility_category'].value_counts())
    # print("DEBUG: Non-void return counts:\n", df['is_non_void_return'].value_counts())
    
    # Select relevant columns for SR analysis
    # Assuming 'model' and 'run' are already in cache.global_details_df
    relevant_cols = [MODEL_COL, RUN_COL, 'visibility_category', 'is_non_void_return'] # Add more as needed
    df_sr_features = df[relevant_cols].copy()
    
    return df_sr_features


def perform_sr1_visibility_stats(df_sr_features: pd.DataFrame, results_summary: list):
    """Performs statistical analysis for SR1 - Visibility Markers."""
    print("\n--- SR1: Visibility Markers Statistical Analysis ---")
    results_summary.append("\n--- SR1: Visibility Markers Statistical Analysis ---")

    if df_sr_features.empty or 'visibility_category' not in df_sr_features.columns:
        msg = "Warning: Data for visibility analysis is empty or missing 'visibility_category' column."
        print(msg); results_summary.append(msg)
        return

    contingency_table = pd.crosstab(df_sr_features[MODEL_COL], df_sr_features['visibility_category'])
    all_vis_cats = ['public', 'private', 'protected', 'package', 'none_or_default']
    for cat in all_vis_cats:
        if cat not in contingency_table.columns:
            contingency_table[cat] = 0
    contingency_table = contingency_table[all_vis_cats]

    results_summary.append("Visibility Marker Counts per Model:\n" + contingency_table.to_string())
    print("Visibility Marker Counts per Model:\n", contingency_table)

    contingency_table_cleaned = contingency_table.loc[(contingency_table.sum(axis=1) > 0), (contingency_table.sum(axis=0) > 0)]

    if contingency_table_cleaned.shape[0] < 2 or contingency_table_cleaned.shape[1] < 2:
        msg = "Chi-squared test for visibility not performed (not enough data rows/cols after cleaning for valid contingency table)."
        print(msg); results_summary.append(msg)
    else:
        print("\nPerforming Chi-squared test for visibility distributions...")
        try:
            chi2_vis, p_vis, dof_vis, expected_vis = stats.chi2_contingency(contingency_table_cleaned)
            results_summary.append(f"Chi-squared Test (Visibility): Chi2={chi2_vis:.3f}, p={p_vis:.4f}, df={dof_vis}")
            print(f"Chi-squared Test (Visibility): Chi2={chi2_vis:.3f}, p={p_vis:.4f}, df={dof_vis}")

            if p_vis < P_VALUE_THRESHOLD:
                # Calculate Cramér's V manually
                n_vis = contingency_table_cleaned.sum().sum()
                if n_vis == 0: # Avoid division by zero if table sum is zero
                    cramers_v_vis = np.nan
                else:
                    phi2_vis = chi2_vis / n_vis
                    r_vis, k_vis = contingency_table_cleaned.shape # r=rows, k=cols
                    # Denominator for Cramer's V can't be zero
                    min_dim = min(k_vis - 1, r_vis - 1)
                    if min_dim == 0:
                        cramers_v_vis = np.nan # Or 0, depending on convention for 2x1 or 1x2 tables
                        print("  Cramér's V not well-defined for this table shape (min(k-1,r-1) is 0).")
                    else:
                        cramers_v_vis = np.sqrt(phi2_vis / min_dim)
                
                if not pd.isna(cramers_v_vis):
                    results_summary.append(f"  Cramér's V (Visibility): {cramers_v_vis:.3f}")
                    print(f"  Cramér's V (Visibility): {cramers_v_vis:.3f}")
                else:
                    results_summary.append("  Cramér's V (Visibility): Not calculated (e.g., due to table dimensions or zero sum).")
                    print("  Cramér's V (Visibility): Not calculated (e.g., due to table dimensions or zero sum).")

        except ValueError as e_chi2: # Catch potential errors from chi2_contingency (e.g., all zeros)
            msg = f"Chi-squared test for visibility failed: {e_chi2}"
            print(msg); results_summary.append(msg)


def perform_sr4_return_type_stats(df_sr_features: pd.DataFrame, results_summary: list):
    """Performs statistical analysis for SR4 - Return Types."""
    print("\n--- SR4: Return Types Statistical Analysis ---")
    results_summary.append("\n--- SR4: Return Types Statistical Analysis ---")

    if df_sr_features.empty or 'is_non_void_return' not in df_sr_features.columns:
        msg = "Warning: Data for return type analysis is empty or missing 'is_non_void_return' column."
        print(msg); results_summary.append(msg)
        return

    contingency_table = pd.crosstab(df_sr_features[MODEL_COL], df_sr_features['is_non_void_return'])
    contingency_table = contingency_table.rename(columns={True: 'Non-Void', False: 'Void'})
    if 'Non-Void' not in contingency_table.columns: contingency_table['Non-Void'] = 0
    if 'Void' not in contingency_table.columns: contingency_table['Void'] = 0
    contingency_table = contingency_table[['Non-Void', 'Void']]

    results_summary.append("Return Type Counts (Non-Void vs. Void) per Model:\n" + contingency_table.to_string())
    print("Return Type Counts (Non-Void vs. Void) per Model:\n", contingency_table)

    contingency_table_cleaned = contingency_table.loc[(contingency_table.sum(axis=1) > 0), (contingency_table.sum(axis=0) > 0)]

    if contingency_table_cleaned.shape[0] < 2 or contingency_table_cleaned.shape[1] < 2:
        msg = "Chi-squared test for return types not performed (not enough data rows/cols after cleaning)."
        print(msg); results_summary.append(msg)
    else:
        print("\nPerforming Chi-squared test for return type distributions...")
        try:
            chi2_ret, p_ret, dof_ret, expected_ret = stats.chi2_contingency(contingency_table_cleaned)
            results_summary.append(f"Chi-squared Test (Return Types): Chi2={chi2_ret:.3f}, p={p_ret:.4f}, df={dof_ret}")
            print(f"Chi-squared Test (Return Types): Chi2={chi2_ret:.3f}, p={p_ret:.4f}, df={dof_ret}")

            if p_ret < P_VALUE_THRESHOLD:
                # Calculate Cramér's V manually
                n_ret = contingency_table_cleaned.sum().sum()
                if n_ret == 0:
                    cramers_v_ret = np.nan
                else:
                    phi2_ret = chi2_ret / n_ret
                    r_ret, k_ret = contingency_table_cleaned.shape
                    min_dim_ret = min(k_ret - 1, r_ret - 1)
                    if min_dim_ret == 0 :
                        cramers_v_ret = np.nan
                        print("  Cramér's V not well-defined for this table shape (min(k-1,r-1) is 0).")
                    else:
                        cramers_v_ret = np.sqrt(phi2_ret / min_dim_ret)
                
                if not pd.isna(cramers_v_ret):
                    results_summary.append(f"  Cramér's V (Return Types): {cramers_v_ret:.3f}")
                    print(f"  Cramér's V (Return Types): {cramers_v_ret:.3f}")
                else:
                    results_summary.append("  Cramér's V (Return Types): Not calculated.")
                    print("  Cramér's V (Return Types): Not calculated.")

        except ValueError as e_chi2_ret: # Catch potential errors from chi2_contingency
            msg = f"Chi-squared test for return types failed: {e_chi2_ret}"
            print(msg); results_summary.append(msg)

def perform_sr4_return_type_stats(df_sr_features: pd.DataFrame, results_summary: list):
    """Performs statistical analysis for SR4 - Return Types."""
    print("\n--- SR4: Return Types Statistical Analysis ---")
    results_summary.append("\n--- SR4: Return Types Statistical Analysis ---")

    if df_sr_features.empty or 'is_non_void_return' not in df_sr_features.columns:
        msg = "Warning: Data for return type analysis is empty or missing 'is_non_void_return' column."
        print(msg); results_summary.append(msg)
        return

    # Contingency table: Models x (NonVoid_Count, Void_Count)
    contingency_table = pd.crosstab(df_sr_features[MODEL_COL], df_sr_features['is_non_void_return'])
    # Rename columns for clarity if they are True/False
    contingency_table = contingency_table.rename(columns={True: 'Non-Void', False: 'Void'})
    
    # Ensure both categories are present
    if 'Non-Void' not in contingency_table.columns: contingency_table['Non-Void'] = 0
    if 'Void' not in contingency_table.columns: contingency_table['Void'] = 0
    contingency_table = contingency_table[['Non-Void', 'Void']] # Ensure order

    results_summary.append("Return Type Counts (Non-Void vs. Void) per Model:\n" + contingency_table.to_string())
    print("Return Type Counts (Non-Void vs. Void) per Model:\n", contingency_table)

    # Chi-squared test
    contingency_table_cleaned = contingency_table.loc[(contingency_table.sum(axis=1) > 0), (contingency_table.sum(axis=0) > 0)]

    if contingency_table_cleaned.shape[0] < 2 or contingency_table_cleaned.shape[1] < 2:
        msg = "Chi-squared test for return types not performed (not enough data rows/cols after cleaning)."
        print(msg); results_summary.append(msg)
    else:
        print("\nPerforming Chi-squared test for return type distributions...")
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table_cleaned)
        results_summary.append(f"Chi-squared Test (Return Types): Chi2={chi2:.3f}, p={p_value:.4f}, df={dof}")
        print(f"Chi-squared Test (Return Types): Chi2={chi2:.3f}, p={p_value:.4f}, df={dof}")

        if p_value < P_VALUE_THRESHOLD:
            cramers_v_results = pg.chi2_independence(data=df_sr_features, x=MODEL_COL, y='is_non_void_return', correction=False)
            # print("DEBUG pg.chi2_independence results (Return Types):\n", cramers_v_results)
            cramers_v = cramers_v_results[cramers_v_results['test'] == 'pearson']['cramer'].iloc[0]
            results_summary.append(f"  Cramér's V (Return Types): {cramers_v:.3f}")
            print(f"  Cramér's V (Return Types): {cramers_v:.3f}")


def main():
    print("--- Starting SR Statistical Tests (SR1 Visibility, SR4 Return Types) ---")
    
    # Initialize MetricCache
    # Assuming CLASS_NAMES is empty for SR to consider all methods from JSONs
    try:
        # Pass empty list for class_names_list to consider all classes for cache.global_details_df
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=[]) 
        print("MetricCache initialized successfully for SR analysis.")
    except Exception as e:
        print(f"Fatal Error: MetricCache initialization failed: {e}")
        print("This script relies on MetricCache to access raw method details from JSONs.")
        return

    # Extract SR features from all methods in the cache
    df_sr_features = extract_sr_features_from_cache(cache)

    if df_sr_features is None or df_sr_features.empty:
        print("Feature extraction for SR failed or resulted in no data. Aborting SR statistical tests.")
        return
    print(f"DEBUG: df_sr_features shape after extraction: {df_sr_features.shape}")
    if 'visibility_category' not in df_sr_features.columns or 'is_non_void_return' not in df_sr_features.columns:
        print("DEBUG: Critical feature columns missing in df_sr_features. Aborting.")
        return

    statistical_results_all_sr = []

    # Perform SR1 - Visibility Markers statistics
    perform_sr1_visibility_stats(df_sr_features, statistical_results_all_sr)
    
    # Perform SR4 - Return Types statistics
    perform_sr4_return_type_stats(df_sr_features, statistical_results_all_sr)

    # Save all statistical results to a single file
    summary_file_path = STATS_OUTPUT_DIR / "SR_Visibility_ReturnTypes_Statistical_Summary.txt"
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            for line in statistical_results_all_sr:
                f.write(line + "\n")
        print(f"\nSaved combined SR statistical summary (Visibility & Return Types) to {summary_file_path}")
    except Exception as e:
        print(f"Error saving combined SR statistical summary: {e}")

    print("\n--- SR Statistical Tests (Visibility & Return Types) Finished ---")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main()