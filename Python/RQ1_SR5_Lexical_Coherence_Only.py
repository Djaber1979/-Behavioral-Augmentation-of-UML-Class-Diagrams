#!/usr/bin/env python3
"""
RQ1_SR5_Lexical_Coherence_Only.py

Calculates and visualizes SR5 - Lexical Coherence:
- Normalizes method names.
- Calculates Lexical Diversity (LexDiv) per model (aggregate and per-run).
- Calculates Redundancy per model (aggregate).
- Generates a scatter plot of aggregate LexDiv vs. Redundancy.
- Performs Kruskal-Wallis test on per-run LexDiv scores to compare models.

Reads raw method data via MetricCache (from main3.py components).
Saves outputs to: reports/article/stats_sr5/ and reports/article/
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
import scikit_posthocs as sp
import pingouin as pg

# --- Import shared pipeline components from main3.py ---
try:
    from main3 import MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME
    METRIC_CACHE_AVAILABLE = True
    print("Successfully imported MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME from main3.py.")
except ImportError as e:
    print(f"Error: Could not import MetricCache or config from main3.py: {e}.")
    print("SR5 analysis requiring MetricCache will not be possible. Please ensure main3.py is accessible.")
    MetricCache = None 
    JSON_INPUT_DIR = Path("JSON") # Provide a fallback default
    BASELINE_JSON_FNAME = "methodless.json" # Provide a fallback default
    METRIC_CACHE_AVAILABLE = False
except NameError as e_name: 
    print(f"NameError during import from main3.py (main3.py might have issues): {e_name}")
    MetricCache = None
    JSON_INPUT_DIR = Path("JSON")
    BASELINE_JSON_FNAME = "methodless.json"
    METRIC_CACHE_AVAILABLE = False


# Configuration
REPORTS_DIR = Path("reports")
ARTICLE_DIR = REPORTS_DIR / "article"
STATS_SR5_OUTPUT_DIR = ARTICLE_DIR / "stats_sr5" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_SR5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'model' 
RUN_COL = 'run'     
RAW_METHOD_NAME_COL = 'name' 
METHOD_SIGNATURE_COL = 'signature' 

P_VALUE_THRESHOLD = 0.05

# --- Helper Functions ---

def local_parse_signature_for_core_name(signature_str: str) -> str | None:
    """
    Simplified local parser to extract the core method name from a signature string.
    Assumes core name is before the first '('.
    """
    if not isinstance(signature_str, str) or pd.isna(signature_str):
        return None
    
    # Attempt to get name before the first parenthesis
    name_part = signature_str.split('(', 1)[0].strip()
    
    # Remove visibility markers and common prefixes like return types by splitting and taking the last word
    # This is a heuristic and might not be perfect for all languages/styles.
    # It assumes the actual method name is the last word before '('.
    potential_name_parts = name_part.split()
    if potential_name_parts:
        core_name = potential_name_parts[-1]
        # Further clean if it's a known visibility marker (unlikely if parser worked, but good for raw strings)
        if core_name in ['+', '-', '#', '~']:
            return None # This was just a marker, no actual name found this way
        return core_name
    return None


def normalize_method_name_for_lexdiv(name_str: str, core_name_from_parser: str | None = None) -> str:
    name_to_process = ""
    if core_name_from_parser and isinstance(core_name_from_parser, str) and core_name_from_parser.strip():
        name_to_process = core_name_from_parser
    elif isinstance(name_str, str) and name_str.strip():
        name_to_process = re.sub(r'\([^)]*\)', '', name_str) 
    else:
        return ""
    normalized = name_to_process.lower()
    normalized = re.sub(r'[^a-z0-9]', '', normalized) 
    return normalized

def calculate_normalized_levenshtein(s1: str, s2: str) -> float:
    if not isinstance(s1, str): s1 = ""
    if not isinstance(s2, str): s2 = ""
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 1.0 
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 0.0
    return Levenshtein.distance(s1, s2) / max_len

def calculate_lexdiv_for_set(method_names_set: set[str], model_id_for_print: str = "") -> float:
    if not method_names_set or len(method_names_set) < 2:
        return 0.0 
    unique_names_list = sorted(list(method_names_set))
    LIMIT_UNIQUE_NAMES_FOR_LEXDIV = 750 
    if len(unique_names_list) > LIMIT_UNIQUE_NAMES_FOR_LEXDIV:
        print(f"    Note ({model_id_for_print}): Limiting LexDiv pairwise comparisons to {LIMIT_UNIQUE_NAMES_FOR_LEXDIV} sampled names out of {len(unique_names_list)} for performance.")
        unique_names_sample = np.random.choice(unique_names_list, size=LIMIT_UNIQUE_NAMES_FOR_LEXDIV, replace=False).tolist()
    else:
        unique_names_sample = unique_names_list
    distances = []
    if len(unique_names_sample) >=2:
        for name1, name2 in itertools.combinations(unique_names_sample, 2):
            distances.append(calculate_normalized_levenshtein(name1, name2))
    return np.mean(distances) if distances else 0.0

def main_sr5_analysis():
    print("--- SR5: Lexical Coherence Analysis ---")
    results_summary_sr5_lines = ["--- SR5: Lexical Coherence Statistical Analysis ---"]

    if not METRIC_CACHE_AVAILABLE or MetricCache is None:
        print("MetricCache could not be initialized. Aborting SR5 analysis.")
        results_summary_sr5_lines.append("MetricCache could not be initialized. Aborting SR5 analysis.")
        with open(STATS_SR5_OUTPUT_DIR / "SR5_LexicalCoherence_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(results_summary_sr5_lines))
        return

    try:
        print("Initializing MetricCache for SR5 (accessing all methods)...")
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=[])
        print(f"MetricCache initialized. Found {len(cache.global_details_df)} method instances.")
    except Exception as e:
        print(f"Fatal Error initializing MetricCache: {e}")
        results_summary_sr5_lines.append(f"Fatal Error initializing MetricCache: {e}")
        with open(STATS_SR5_OUTPUT_DIR / "SR5_LexicalCoherence_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(results_summary_sr5_lines))
        return

    if cache.global_details_df.empty:
        print("MetricCache.global_details_df is empty. No method data to analyze for SR5.")
        results_summary_sr5_lines.append("MetricCache.global_details_df is empty.")
        with open(STATS_SR5_OUTPUT_DIR / "SR5_LexicalCoherence_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(results_summary_sr5_lines))
        return

    print("Extracting and normalizing method names for SR5 analysis...")
    methods_for_sr5_list = []
    for idx, row in cache.global_details_df.iterrows():
        raw_name = row.get(RAW_METHOD_NAME_COL) # 'name' column from global_details_df
        signature = row.get(METHOD_SIGNATURE_COL) # 'signature' column from global_details_df
        
        # Use the local simplified parser to get a core name if signature exists
        core_name = local_parse_signature_for_core_name(signature)
        
        # If core_name couldn't be parsed from signature, use the raw_name itself (after stripping params by regex)
        # The normalize_method_name_for_lexdiv function expects either a core name or will process the raw name
        normalized_name = normalize_method_name_for_lexdiv(raw_name, core_name_from_parser=core_name)

        if normalized_name: 
            methods_for_sr5_list.append({
                MODEL_COL: row[MODEL_COL], RUN_COL: row[RUN_COL],
                'Normalized_Name_for_LexDiv': normalized_name
            })
    
    df_sr5_methods = pd.DataFrame(methods_for_sr5_list)
    if df_sr5_methods.empty:
        print("No valid method names found after extraction and normalization for SR5. Aborting."); return

    # --- Calculate LexDiv per run per model ---
    # ... (rest of LexDiv per run, Aggregate LexDiv, Redundancy, Scatter Plot, Kruskal-Wallis as before) ...
    # (For brevity, I'm not repeating the identical code for these sections. 
    #  Ensure they correctly use MODEL_COL and RUN_COL as defined at the top of this script)

    print("\nCalculating LexDiv per run per model...")
    lexdiv_per_run_data = []
    for (model_name_grp, run_id_grp), group in df_sr5_methods.groupby([MODEL_COL, RUN_COL]):
        unique_names_run = set(group['Normalized_Name_for_LexDiv'])
        lexdiv_run_score = calculate_lexdiv_for_set(unique_names_run, f"{model_name_grp}-Run{run_id_grp}")
        lexdiv_per_run_data.append({MODEL_COL: model_name_grp, RUN_COL: run_id_grp, 'LexDiv_Per_Run': lexdiv_run_score})
    lexdiv_per_run_df = pd.DataFrame(lexdiv_per_run_data)
    if lexdiv_per_run_df.empty: results_summary_sr5_lines.append("Warning: LexDiv per run data is empty.")

    print("\nCalculating aggregate LexDiv per model...")
    aggregate_lexdiv_data = []
    for model_name_grp, group in df_sr5_methods.groupby(MODEL_COL):
        unique_names_model = set(group['Normalized_Name_for_LexDiv'])
        lexdiv_model_score = calculate_lexdiv_for_set(unique_names_model, model_name_grp)
        aggregate_lexdiv_data.append({MODEL_COL: model_name_grp, 'LexDiv_Agg': lexdiv_model_score})
    aggregate_lexdiv_df = pd.DataFrame(aggregate_lexdiv_data)
    if not aggregate_lexdiv_df.empty:
        results_summary_sr5_lines.append("\nAggregate Lexical Diversity (LexDiv) per Model:\n" + aggregate_lexdiv_df.round(3).to_string())
    else: results_summary_sr5_lines.append("Warning: Aggregate LexDiv DataFrame is empty.")

    print("\nCalculating Redundancy per model...")
    redundancy_data = []
    for model_name_grp in df_sr5_methods[MODEL_COL].unique():
        total_raw_methods_model = len(cache.global_details_df[cache.global_details_df[MODEL_COL] == model_name_grp])
        unique_normalized_names_model = set(df_sr5_methods[df_sr5_methods[MODEL_COL] == model_name_grp]['Normalized_Name_for_LexDiv'])
        num_unique_normalized = len(unique_normalized_names_model)
        redundancy_score = total_raw_methods_model / num_unique_normalized if num_unique_normalized > 0 else np.nan
        redundancy_data.append({MODEL_COL: model_name_grp, 'Redundancy': redundancy_score})
    redundancy_df = pd.DataFrame(redundancy_data)
    if not redundancy_df.empty:
        results_summary_sr5_lines.append("\nMethod Name Redundancy per Model:\n" + redundancy_df.round(2).to_string())
    else: results_summary_sr5_lines.append("Warning: Redundancy DataFrame is empty.")

    if not aggregate_lexdiv_df.empty and not redundancy_df.empty:
        sr5_aggregates_df = pd.merge(aggregate_lexdiv_df, redundancy_df, on=MODEL_COL)
        sr5_aggregates_path = STATS_SR5_OUTPUT_DIR / "SR5_LexDiv_Redundancy_Aggregates.csv"
        sr5_aggregates_df.to_csv(sr5_aggregates_path, index=False, float_format='%.3f')
        print(f"Saved SR5 aggregate metrics to {sr5_aggregates_path}")
        scatter_plot_df = sr5_aggregates_df.dropna(subset=['LexDiv_Agg', 'Redundancy'])
        if not scatter_plot_df.empty:
            print("\nGenerating Scatter Plot: LexDiv vs. Redundancy...")
            plt.figure(figsize=(12, 8)) # Adjusted size
            num_models_plot = scatter_plot_df[MODEL_COL].nunique()
            markers_list_plot = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
            palette_plot = sns.color_palette("viridis", n_colors=num_models_plot) if num_models_plot > 0 else sns.color_palette()
            
            plot_handles_list = []
            unique_models_in_plot = scatter_plot_df[MODEL_COL].unique()
            for i, model_name_p in enumerate(unique_models_in_plot):
                model_data_p = scatter_plot_df[scatter_plot_df[MODEL_COL] == model_name_p]
                handle = plt.scatter(model_data_p['Redundancy'], model_data_p['LexDiv_Agg'], 
                                     label=model_name_p, 
                                     marker=markers_list_plot[i % len(markers_list_plot)], 
                                     color=palette_plot[i % len(palette_plot)], s=120)
                plot_handles_list.append(handle)
                for _, row_p in model_data_p.iterrows():
                    plt.text(row_p['Redundancy'] + 0.005 * scatter_plot_df['Redundancy'].max(skipna=True), 
                             row_p['LexDiv_Agg'], 
                             row_p[MODEL_COL], fontsize=8, ha='left', va='center')
            
            plt.xlabel("Redundancy (Total Methods / Unique Normalized Names)", fontsize=12)
            plt.ylabel("Lexical Diversity (Mean Norm. Levenshtein)", fontsize=12)
            plt.title("Lexical Diversity vs. Redundancy by LLM", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            if num_models_plot > 0 and num_models_plot <= 10:
                 plt.legend(handles=plot_handles_list, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                 plt.tight_layout(rect=[0,0,0.80,1]) 
            else: plt.tight_layout()
            lexdiv_scatter_path = ARTICLE_DIR / "SR_PointCloud_LexDiv_vs_Redundancy.png"
            plt.savefig(lexdiv_scatter_path); print(f"Saved LexDiv vs Redundancy scatter plot to {lexdiv_scatter_path}"); plt.close()

    results_summary_sr5_lines.append("\n--- Kruskal-Wallis Test for Per-Run Lexical Diversity ---")
    print("\n--- Kruskal-Wallis Test for Per-Run Lexical Diversity ---")
    if not lexdiv_per_run_df.empty and 'LexDiv_Per_Run' in lexdiv_per_run_df.columns and lexdiv_per_run_df[MODEL_COL].nunique() >=2 :
        df_for_kw_lexdiv = lexdiv_per_run_df.dropna(subset=['LexDiv_Per_Run'])
        valid_groups_for_kw = df_for_kw_lexdiv.groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
        if not valid_groups_for_kw.empty and valid_groups_for_kw[MODEL_COL].nunique() >= 2:
            kw_lexdiv_results = pg.kruskal(data=valid_groups_for_kw, dv='LexDiv_Per_Run', between=MODEL_COL)
            if kw_lexdiv_results is not None and not kw_lexdiv_results.empty:
                results_summary_sr5_lines.append("Kruskal-Wallis Results (Per-Run LexDiv):\n" + kw_lexdiv_results.round(4).to_string())
                print(kw_lexdiv_results.round(4))
                p_value_kw_lexdiv = kw_lexdiv_results['p-unc'].iloc[0]
                if 'eps-sq' in kw_lexdiv_results.columns and pd.notna(kw_lexdiv_results['eps-sq'].iloc[0]):
                    results_summary_sr5_lines.append(f"  Effect Size (Epsilon-squared, ε²): {kw_lexdiv_results['eps-sq'].iloc[0]:.3f}")
                if p_value_kw_lexdiv < P_VALUE_THRESHOLD:
                    results_summary_sr5_lines.append("  Kruskal-Wallis significant. Dunn's post-hoc:")
                    print("  Kruskal-Wallis significant. Dunn's post-hoc:")
                    dunn_lexdiv_results = sp.posthoc_dunn(valid_groups_for_kw, val_col='LexDiv_Per_Run', group_col=MODEL_COL, p_adjust='bonferroni')
                    results_summary_sr5_lines.append("  Dunn's Test Results (Per-Run LexDiv):\n" + dunn_lexdiv_results.round(4).to_string())
                    print(dunn_lexdiv_results.round(4))
                else: results_summary_sr5_lines.append("  Kruskal-Wallis not significant for per-run LexDiv.")
            else: results_summary_sr5_lines.append("  Kruskal-Wallis for per-run LexDiv could not be computed.")
        else: results_summary_sr5_lines.append("  Not enough data/groups for Kruskal-Wallis on LexDiv_Per_Run after filtering.")
    else: results_summary_sr5_lines.append("  Not enough data or model groups for Kruskal-Wallis on per-run LexDiv scores.")
        
    sr5_summary_file_path = STATS_SR5_OUTPUT_DIR / "SR5_LexicalCoherence_Stats.txt"
    with open(sr5_summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(results_summary_sr5_lines))
    print(f"\nSaved SR5 Lexical Coherence statistical summary to {sr5_summary_file_path}")
    print("\n--- SR5: Lexical Coherence Analysis Finished ---")

if __name__ == "__main__":
    if not METRIC_CACHE_AVAILABLE:
        print("Exiting script because MetricCache from main3.py could not be imported.")
    else:
        try: plt.style.use('seaborn-v0_8-whitegrid')
        except OSError: print("Warning: Seaborn style not found, using Matplotlib default.")
        main_sr5_analysis()