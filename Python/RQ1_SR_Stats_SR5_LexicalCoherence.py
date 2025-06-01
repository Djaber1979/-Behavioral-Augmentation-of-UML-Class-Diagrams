#!/usr/bin/env python3
"""
RQ1_SR_Stats_SR5_LexicalCoherence.py

Analyzes SR5 - Lexical Coherence (LexDiv & Redundancy).
- Normalizes method names.
- Calculates LexDiv per run per model.
- Calculates aggregate LexDiv and Redundancy per model.
- Performs Kruskal-Wallis test on per-run LexDiv scores to compare models.
- Generates a scatter plot of LexDiv vs. Redundancy.

Reads raw method data via MetricCache from main3.py.
Saves outputs to: reports/article/stats/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import Levenshtein # pip install python-Levenshtein

# Statistical libraries
from scipy import stats
import scikit_posthocs as sp # pip install scikit-posthocs
import pingouin as pg       # pip install pingouin

# --- Import shared pipeline components from main3.py ---
try:
    from main3 import MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME, CLASS_NAMES # Use global CLASS_NAMES from main3
except ImportError:
    print("Error: Could not import from main3.py. Ensure it's in the Python path.")
    print("Define dummy values for MetricCache components for standalone testing.")
    JSON_INPUT_DIR = Path("JSON_path_not_set") # Dummy path
    BASELINE_JSON_FNAME = "baseline_not_set.json"
    CLASS_NAMES = [] # Default to all classes if main3.py not found
    # Fallback MetricCache if main3.py is not available
    class MetricCache:
        def __init__(self, json_dir, baseline_fname, class_names_list):
            print("Using DUMMY MetricCache. No data will be loaded.")
            self.global_details_df = pd.DataFrame(columns=['model', 'run', 'name', 'signature']) # Ensure schema

# Configuration
REPORTS_DIR = Path("reports")
ARTICLE_DIR = REPORTS_DIR / "article"
STATS_OUTPUT_DIR = ARTICLE_DIR / "stats"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'Model' # Standardized to 'Model' after processing cache.global_details_df
RUN_COL = 'Run'
RAW_METHOD_NAME_COL = 'name' # From cache.global_details_df
SIGNATURE_COL = 'signature'  # From cache.global_details_df

P_VALUE_THRESHOLD = 0.05

# --- Helper Functions ---
def parse_method_signature_core_name(signature_str: str) -> str | None:
    """Simplified parser to primarily extract the core method name before parentheses."""
    if not isinstance(signature_str, str) or pd.isna(signature_str):
        return None
    
    # Remove visibility and other prefixes (basic)
    signature_str = re.sub(r"^\s*([+\-#~]|[a-zA-Z_][a-zA-Z0-9_]*\s+)*", "", signature_str).strip()

    name_match = signature_str.split('(', 1)
    if name_match:
        core_name_parts = name_match[0].strip().split()
        if core_name_parts:
            return core_name_parts[-1] # The last word before '('
    return signature_str.strip() # Fallback if no parentheses

def normalize_method_name_for_lexdiv(name: str) -> str:
    """ Normalizes method name for lexical diversity: lowercase, strip non-alphanumeric."""
    if not isinstance(name, str) or pd.isna(name): return ""
    normalized = name.lower()
    normalized = re.sub(r'[^a-z0-9]', '', normalized) # Keep only letters and numbers
    return normalized

def calculate_normalized_levenshtein(s1: str, s2: str) -> float:
    if not isinstance(s1, str): s1 = str(s1)
    if not isinstance(s2, str): s2 = str(s2)
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 0.0
    return Levenshtein.distance(s1, s2) / max_len

def calculate_lexical_diversity_for_set(method_names_set: set[str]) -> float | None:
    """Calculates LexDiv for a set of unique normalized method names."""
    if not method_names_set or len(method_names_set) < 2:
        return 0.0 # No diversity if 0 or 1 unique name

    unique_names_list = sorted(list(method_names_set)) # Ensure order for combinations
    
    # Performance consideration for very large number of unique names
    MAX_UNIQUE_NAMES_FOR_LEXDIV_PAIRS = 500 # Arbitrary limit
    if len(unique_names_list) > MAX_UNIQUE_NAMES_FOR_LEXDIV_PAIRS:
        print(f"    Warning: Model has {len(unique_names_list)} unique names. LexDiv calculation might be slow or sampled if implemented.")
        # For now, let's proceed, but be mindful. Sampling would be:
        # import random
        # unique_names_list = random.sample(unique_names_list, MAX_UNIQUE_NAMES_FOR_LEXDIV_PAIRS)

    distances = []
    for name1, name2 in itertools.combinations(unique_names_list, 2):
        distances.append(calculate_normalized_levenshtein(name1, name2))
            
    return np.mean(distances) if distances else 0.0

def extract_sr5_features_from_cache(cache: MetricCache) -> pd.DataFrame:
    """
    Extracts and prepares data for SR5 analysis from MetricCache.
    Adds 'Model', 'Run', 'Raw_Method_Name', 'Method_Name_Core', 'Normalized_Method_Name_LexDiv'.
    """
    print("DEBUG: Entering extract_sr5_features_from_cache")
    if cache is None or cache.global_details_df.empty:
        print("Error: MetricCache not initialized or global_details_df is empty.")
        return pd.DataFrame()

    df = cache.global_details_df.copy()
    # Standardize column names from cache.global_details_df if they are different
    df.rename(columns={'model': MODEL_COL, 'run': RUN_COL, 
                       'name': RAW_METHOD_NAME_COL, 'signature': SIGNATURE_COL}, 
              inplace=True, errors='ignore') # errors='ignore' if some columns don't exist

    required_cols = [MODEL_COL, RUN_COL, RAW_METHOD_NAME_COL, SIGNATURE_COL]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame from cache is missing required columns for SR5: {required_cols}")
        print(f"Available columns after rename attempt: {df.columns.tolist()}")
        return pd.DataFrame()
        
    print(f"DEBUG: Parsing signatures to get method_name_core for {len(df)} methods...")
    df['Method_Name_Core'] = df[SIGNATURE_COL].astype(str).apply(parse_method_signature_core_name)
    
    # Handle cases where core name might be None (e.g. if signature was unparsable)
    # Fallback to raw method name if core name extraction fails
    df['Method_Name_Core'].fillna(df[RAW_METHOD_NAME_COL], inplace=True)
    df.dropna(subset=['Method_Name_Core'], inplace=True) # Drop if still NaN

    print(f"DEBUG: Normalizing method_name_core for LexDiv...")
    df['Normalized_Method_Name_LexDiv'] = df['Method_Name_Core'].astype(str).apply(normalize_method_name_for_lexdiv)
    
    # Filter out methods that became empty after normalization (e.g. only punctuation)
    df = df[df['Normalized_Method_Name_LexDiv'] != ''].copy()
    
    print(f"DEBUG: Finished SR5 feature extraction. DataFrame shape: {df.shape}")
    # print("DEBUG: Sample of SR5 features:\n", df[[MODEL_COL, RUN_COL, RAW_METHOD_NAME_COL, 'Method_Name_Core', 'Normalized_Method_Name_LexDiv']].head())
    return df[[MODEL_COL, RUN_COL, RAW_METHOD_NAME_COL, 'Normalized_Method_Name_LexDiv']]


def analyze_lexical_coherence_sr5(df_sr5_features: pd.DataFrame):
    """Analyzes Lexical Coherence (LexDiv and Redundancy) for SR5."""
    print("\n--- SR5: Lexical Coherence Analysis ---")
    results_summary_sr5 = ["--- SR5: Lexical Coherence ---"]

    if df_sr5_features.empty:
        print("Warning: DataFrame for SR5 features is empty. Skipping Lexical Coherence analysis.")
        return None, None, None

    # --- Calculate LexDiv per Run per Model ---
    print("Calculating LexDiv per run per model...")
    run_lexdiv_data = []
    for (model_name, run_id), group in df_sr5_features.groupby([MODEL_COL, RUN_COL]):
        unique_normalized_names_run = set(group['Normalized_Method_Name_LexDiv'].dropna())
        lexdiv_run_score = calculate_lexical_diversity_for_set(unique_normalized_names_run)
        run_lexdiv_data.append({
            MODEL_COL: model_name, 
            RUN_COL: run_id, 
            'LexDiv_Per_Run': lexdiv_run_score
        })
    
    df_lexdiv_per_run = pd.DataFrame(run_lexdiv_data)
    if df_lexdiv_per_run.empty:
        print("Warning: No per-run LexDiv scores calculated.")
        # Initialize empty df for stats to avoid errors if subsequent steps expect it
        lexdiv_agg_df = pd.DataFrame(columns=[MODEL_COL, 'LexDiv']) 
    else:
        print("Per-Run LexDiv Scores (sample):\n", df_lexdiv_per_run.head())
        results_summary_sr5.append("\nPer-Run LexDiv Scores (sample):\n" + df_lexdiv_per_run.head().to_string())
        # Aggregate LexDiv per Model (for table and scatter plot)
        lexdiv_agg_df = df_lexdiv_per_run.groupby(MODEL_COL)['LexDiv_Per_Run'].mean().reset_index()
        lexdiv_agg_df.rename(columns={'LexDiv_Per_Run': 'LexDiv'}, inplace=True)
        print("\nAggregated Lexical Diversity (LexDiv) per Model:\n", lexdiv_agg_df.round(3))
        results_summary_sr5.append("\nAggregated Lexical Diversity (LexDiv) per Model:\n" + lexdiv_agg_df.round(3).to_string())


    # --- Calculate Redundancy per Model ---
    print("\nCalculating Redundancy per model...")
    redundancy_data = []
    for model_name, group in df_sr5_features.groupby(MODEL_COL):
        total_methods_model = len(group) # Total raw methods for this model
        unique_normalized_names_model = set(group['Normalized_Method_Name_LexDiv'].dropna())
        num_unique_normalized = len(unique_normalized_names_model)
        
        redundancy_score = total_methods_model / num_unique_normalized if num_unique_normalized > 0 else np.nan
        redundancy_data.append({MODEL_COL: model_name, 'Redundancy': redundancy_score})

    redundancy_df = pd.DataFrame(redundancy_data)
    if not redundancy_df.empty:
        print("Method Name Redundancy per Model:\n", redundancy_df.round(2))
        results_summary_sr5.append("\nMethod Name Redundancy per Model:\n" + redundancy_df.round(2).to_string())


    # --- Statistical Test for LexDiv (Kruskal-Wallis on per-run LexDiv scores) ---
    results_summary_sr5.append("\n--- Statistical Test for Per-Run Lexical Diversity ---")
    print("\n--- Statistical Test for Per-Run Lexical Diversity ---")
    if not df_lexdiv_per_run.empty and df_lexdiv_per_run[MODEL_COL].nunique() >= 2:
        print("Performing Kruskal-Wallis Test on per-run LexDiv scores...")
        # Drop models with all NaN LexDiv_Per_Run or insufficient data before K-W
        # Pingouin's kruskal handles NaN in dv, but groups must exist.
        kw_data_lexdiv = df_lexdiv_per_run.dropna(subset=['LexDiv_Per_Run'])
        if kw_data_lexdiv[MODEL_COL].nunique() >=2 :
            kw_results_lexdiv = pg.kruskal(data=kw_data_lexdiv, dv='LexDiv_Per_Run', between=MODEL_COL)
            if kw_results_lexdiv is not None and not kw_results_lexdiv.empty:
                print(kw_results_lexdiv.round(4))
                results_summary_sr5.append("Kruskal-Wallis Results (Per-Run LexDiv):\n" + kw_results_lexdiv.round(4).to_string())
                h_stat_lexdiv = kw_results_lexdiv['H'].iloc[0]
                p_value_kw_lexdiv = kw_results_lexdiv['p-unc'].iloc[0]
                if 'eps-sq' in kw_results_lexdiv.columns and not pd.isna(kw_results_lexdiv['eps-sq'].iloc[0]):
                    epsilon_sq_lexdiv = kw_results_lexdiv['eps-sq'].iloc[0]
                    results_summary_sr5.append(f"Effect Size (Epsilon-squared, ε²): {epsilon_sq_lexdiv:.3f}")
                    print(f"Effect Size (Epsilon-squared, ε²): {epsilon_sq_lexdiv:.3f}")

                if p_value_kw_lexdiv < P_VALUE_THRESHOLD:
                    results_summary_sr5.append("Kruskal-Wallis significant for LexDiv. Dunn's post-hoc:")
                    print("Kruskal-Wallis significant for LexDiv. Dunn's post-hoc:")
                    dunn_results_lexdiv = sp.posthoc_dunn(kw_data_lexdiv, val_col='LexDiv_Per_Run', group_col=MODEL_COL, p_adjust='bonferroni')
                    print(dunn_results_lexdiv.round(4))
                    results_summary_sr5.append("Dunn's Test Results (LexDiv):\n" + dunn_results_lexdiv.round(4).to_string())
                else:
                    results_summary_sr5.append("Kruskal-Wallis not significant for per-run LexDiv.")
                    print("Kruskal-Wallis not significant for per-run LexDiv.")
            else:
                results_summary_sr5.append("Kruskal-Wallis for LexDiv could not be computed (empty results).")
                print("Kruskal-Wallis for LexDiv could not be computed (empty results).")
        else:
            results_summary_sr5.append("Not enough groups with valid data for Kruskal-Wallis on LexDiv.")
            print("Not enough groups with valid data for Kruskal-Wallis on LexDiv.")
    else:
        results_summary_sr5.append("Not enough data or models for Kruskal-Wallis test on LexDiv.")
        print("Not enough data or models for Kruskal-Wallis test on LexDiv.")

    # --- Scatter Plot (Figure SR5) ---
    print("\nGenerating Scatter Plot for LexDiv vs. Redundancy...")
    if not lexdiv_agg_df.empty and not redundancy_df.empty:
        scatter_plot_df = pd.merge(lexdiv_agg_df, redundancy_df, on=MODEL_COL, how='inner') # Use inner merge
        scatter_plot_df.dropna(subset=['LexDiv', 'Redundancy'], inplace=True) # Drop if any crucial val is NaN

        if not scatter_plot_df.empty:
            plt.figure(figsize=(10, 7))
            # Use a consistent color palette for models if possible
            model_order = sorted(scatter_plot_df[MODEL_COL].unique())
            palette = sns.color_palette("viridis", n_colors=len(model_order))
            
            sns.scatterplot(x='Redundancy', y='LexDiv', hue=MODEL_COL, data=scatter_plot_df, 
                            s=120, palette=palette, hue_order=model_order, legend='full')
            
            for i in range(scatter_plot_df.shape[0]):
                plt.text(scatter_plot_df['Redundancy'].iloc[i] + 0.01 * scatter_plot_df['Redundancy'].max(skipna=True), 
                         scatter_plot_df['LexDiv'].iloc[i], 
                         scatter_plot_df[MODEL_COL].iloc[i], fontsize=9)
            
            plt.xlabel("Redundancy (Total Methods / Unique Normalized Names)", fontsize=12)
            plt.ylabel("Lexical Diversity (Mean Normalized Levenshtein)", fontsize=12)
            plt.title("Lexical Diversity vs. Redundancy by LLM", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            # Adjust legend if too many models
            if len(model_order) > 8:
                 plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            else:
                 plt.legend(title='Model')

            plt.tight_layout(rect=[0,0,0.85 if len(model_order) > 8 else 1,1]) # Adjust for legend
            lexdiv_scatter_path = ARTICLE_DIR / "SR_PointCloud_LexDiv_vs_Redundancy.png"
            plt.savefig(lexdiv_scatter_path)
            print(f"Saved LexDiv vs Redundancy scatter plot to {lexdiv_scatter_path}")
            plt.close()
        else:
            print("Warning: Data for LexDiv vs Redundancy scatter plot is empty after merge/dropna.")
    else:
        print("Warning: LexDiv or Redundancy aggregate data is empty, cannot create scatter plot.")
        
    return lexdiv_agg_df, redundancy_df, "\n".join(results_summary_sr5)


def main():
    print("--- Starting SR5 Lexical Coherence Analysis ---")
    
    print("Initializing MetricCache to access raw JSON data...")
    try:
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=CLASS_NAMES) # Respect CLASS_NAMES from main3
        print("MetricCache initialized for SR5.")
    except NameError as e:
        print(f"NameError during MetricCache initialization (likely main3.py components not imported correctly): {e}")
        return
    except FileNotFoundError as e:
        print(f"FileNotFoundError during MetricCache initialization (e.g., baseline file or JSON dir): {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during MetricCache initialization: {e}")
        return

    if cache.global_details_df.empty:
        print("MetricCache global_details_df is empty. Cannot proceed with SR5.")
        return

    df_sr5_features = extract_sr5_features_from_cache(cache)
    if df_sr5_features.empty:
        print("SR5 Feature extraction failed or resulted in empty data. Aborting SR5 analysis.")
        return

    # --- SR5: Lexical Coherence ---
    lexdiv_agg_df, redundancy_agg_df, sr5_stats_summary_text = analyze_lexical_coherence_sr5(df_sr5_features)

    if sr5_stats_summary_text:
        sr5_summary_file_path = STATS_OUTPUT_DIR / "SR5_LexicalCoherence_Stats.txt"
        try:
            with open(sr5_summary_file_path, 'w', encoding='utf-8') as f:
                f.write(sr5_stats_summary_text)
            print(f"\nSaved SR5 Lexical Coherence statistical summary to {sr5_summary_file_path}")
        except Exception as e:
            print(f"Error saving SR5 summary text file: {e}")
    else:
        print("DEBUG: No statistical summary text generated for SR5.")

    # --- Combine data for SR Summary Table (LexDiv and Redundancy part) ---
    sr5_table_data_list = []
    if lexdiv_agg_df is not None and not lexdiv_agg_df.empty:
        sr5_table_data_list.append(lexdiv_agg_df.set_index(MODEL_COL))
    if redundancy_agg_df is not None and not redundancy_agg_df.empty:
        # For a combined table, you might choose to report redundancy or not, depending on Table SR structure
        # For now, let's assume we want it for a specific SR5 table output
        sr5_table_data_list.append(redundancy_agg_df.set_index(MODEL_COL))
    
    if sr5_table_data_list:
        sr5_combined_df = pd.concat(sr5_table_data_list, axis=1).reset_index()
        sr5_table_path = ARTICLE_DIR / "SR5_LexDiv_Redundancy_Summary.csv"
        sr5_combined_df.to_csv(sr5_table_path, index=False, float_format='%.3f')
        print(f"Saved SR5 LexDiv and Redundancy summary table to {sr5_table_path}")
    else:
        print("No LexDiv or Redundancy data to save in summary table for SR5.")

    print("\n--- SR5 Lexical Coherence Analysis Finished ---")


if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib default.")
    main()