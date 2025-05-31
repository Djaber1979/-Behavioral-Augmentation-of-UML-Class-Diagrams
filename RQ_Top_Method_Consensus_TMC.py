#!/usr/bin/env python3
"""
RQ_Top_Method_Consensus_TMC.py

Analyzes Top-Method Consensus (TMC):
1.  Identifies the Top-k (k=37) most frequent *normalized* method names (core vocabulary).
2.  Outputs a table of these Top-k normalized methods and their frequencies.
3.  Calculates and visualizes (bar chart) the aggregate coverage of this Top-k set by each LLM (for Table tmc-overlap).
4.  Calculates per-run coverage of Top-k, generates boxplot (Figure tmc-boxplot), and Kruskal-Wallis.
5.  Calculates per-run Jaccard similarity of model's normalized methods vs. Top-k benchmark.
6.  Generates bar chart of mean per-run Jaccard scores (Figure tmc-jaccard-bar) and Kruskal-Wallis.

Reads raw method data via MetricCache.
Saves outputs to: reports/article/stats_tmc/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
from collections import Counter

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
    print(f"Error importing from main3.py: {e}. SR5 analysis may be limited or fail.")
    MetricCache = None; METRIC_CACHE_AVAILABLE = False
    JSON_INPUT_DIR = Path("JSON"); BASELINE_JSON_FNAME = "methodless.json" 
except NameError as e_name: 
    print(f"NameError during import from main3.py: {e_name}")
    MetricCache = None; METRIC_CACHE_AVAILABLE = False
    JSON_INPUT_DIR = Path("JSON"); BASELINE_JSON_FNAME = "methodless.json"


# Configuration
REPORTS_DIR: Path = Path("reports")
ARTICLE_DIR: Path = REPORTS_DIR / "article"
STATS_TMC_OUTPUT_DIR: Path = ARTICLE_DIR / "stats_tmc" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_TMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL: str = 'model' 
RUN_COL: str = 'run'     
RAW_METHOD_NAME_COL: str = 'name' 
METHOD_SIGNATURE_COL: str = 'signature' 

K_TOP_METHODS: int = 37
P_VALUE_THRESHOLD: float = 0.05

# --- Custom Color Palette and Order ---
# IMPORTANT: Keys in LLM_COLORS and names in LLM_ORDER_FOR_PLOTS 
# MUST EXACTLY MATCH the unique names in your DataFrame's 'Model' column.
LLM_COLORS = {
    'Claude3.7': '#E69F00',                    
    'Gemini-2.5-Pro-Preview-05-06': '#D55E00', 
    'ChatGPT-o3': '#009E73',      
    'Qwen3': '#56B4E9',             
    'DeepSeek(R1)': '#CC79A7',      
    'Grok3': '#0072B2',             
    'ChatGPT-4o': '#8C8C00',       
    'Llama4': '#E76F51',            
    'Mistral': '#999999'           
}

LLM_ORDER_FOR_PLOTS = [ 
    'Claude3.7',                    
    'Gemini-2.5-Pro-Preview-05-06', 
    'ChatGPT-o3',
    'Qwen3',
    'DeepSeek(R1)',                  
    'Grok3',
    'ChatGPT-4o',
    'Llama4',
    'Mistral'                        
]
# --- End Custom Color Palette and Order ---


# --- Helper Functions ---
def local_parse_signature_for_core_name(signature_str: str) -> str | None:
    if not isinstance(signature_str, str) or pd.isna(signature_str): return None
    name_part = signature_str.split('(', 1)[0].strip()
    name_part = re.sub(r"^\s*[+\-#~]\s*", "", name_part).strip()
    potential_name_parts = name_part.split()
    if potential_name_parts:
        core_name = potential_name_parts[-1]
        common_modifiers = {'static', 'final', 'abstract', 'public', 'private', 'protected', 'const', 
                            'void', 'int', 'string', 'boolean', 'float', 'double', 'long', 'char', 
                            'byte', 'short', 'list', 'map', 'set', 'array', 'task', 'async', 'await', 'override', 'virtual'}
        if core_name.lower() in common_modifiers or any(c in core_name for c in [':', '<', '>']):
            if len(potential_name_parts) > 1:
                prev_word = potential_name_parts[-2]
                if prev_word.lower() not in common_modifiers and not any(c in prev_word for c in [':', '<', '>']) and not prev_word[0].isupper():
                    return prev_word
            return None
        return core_name
    return None

def normalize_method_name_for_tmc(name_str: str, core_name_from_parser: str | None = None) -> str:
    name_to_process = ""
    if core_name_from_parser and isinstance(core_name_from_parser, str) and core_name_from_parser.strip():
        name_to_process = core_name_from_parser
    elif isinstance(name_str, str) and name_str.strip():
        name_to_process = re.sub(r'\([^)]*\)', '', name_str) 
    else: return ""
    normalized = name_to_process.lower()
    normalized = re.sub(r'[^a-z0-9]', '', normalized) 
    return normalized

def jaccard_similarity(set1: set, set2: set) -> float:
    if not isinstance(set1, set) or not isinstance(set2, set): return 0.0
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1.union(set2))
    if union_len == 0: return 1.0 if intersection_len == 0 else 0.0
    return intersection_len / union_len

def get_all_normalized_methods(cache: MetricCache) -> pd.DataFrame:
    print("Extracting and normalizing all method names...")
    all_extracted_methods = []
    if cache is None or cache.global_details_df.empty:
        print("Warning: MetricCache not available or global_details_df is empty.")
        return pd.DataFrame()
    parsed_count = 0; unparsed_core_name_count = 0
    for _, row in cache.global_details_df.iterrows():
        raw_name = row.get(RAW_METHOD_NAME_COL)
        signature = row.get(METHOD_SIGNATURE_COL)
        core_name = local_parse_signature_for_core_name(signature) if signature else None
        if core_name: parsed_count +=1
        else: unparsed_core_name_count +=1
        normalized_name = normalize_method_name_for_tmc(raw_name, core_name_from_parser=core_name)
        if normalized_name: 
            all_extracted_methods.append({
                MODEL_COL: row[MODEL_COL], RUN_COL: row[RUN_COL],
                'Normalized_TMC_Name': normalized_name
            })
    print(f"DEBUG: Core names parsed by local_parse_signature_for_core_name: {parsed_count}")
    print(f"DEBUG: Core names not found by local parser (used raw name for normalization): {unparsed_core_name_count}")
    df = pd.DataFrame(all_extracted_methods)
    if df.empty: print("Warning: No methods extracted or all normalized names were empty.")
    else: print(f"Extracted {len(df)} method instances with non-empty normalized names.")
    return df

def main_tmc_analysis():
    print("--- Top-Method Consensus (TMC) Analysis ---")
    tmc_summary_lines = ["--- Top-Method Consensus (TMC) Analysis ---"]

    if not METRIC_CACHE_AVAILABLE:
        print("Exiting TMC: MetricCache from main3.py could not be imported.")
        tmc_summary_lines.append("MetricCache could not be imported. Aborting.")
        with open(STATS_TMC_OUTPUT_DIR / "TMC_Analysis_Summary_With_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(tmc_summary_lines))
        return
    try:
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=[])
        print(f"MetricCache initialized. Found {len(cache.global_details_df)} method instances.")
    except Exception as e:
        print(f"Fatal Error initializing MetricCache: {e}")
        tmc_summary_lines.append(f"Fatal Error initializing MetricCache: {e}")
        with open(STATS_TMC_OUTPUT_DIR / "TMC_Analysis_Summary_With_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(tmc_summary_lines))
        return
    if cache.global_details_df.empty:
        print("MetricCache.global_details_df is empty. No method data for TMC.")
        tmc_summary_lines.append("MetricCache.global_details_df is empty.")
        with open(STATS_TMC_OUTPUT_DIR / "TMC_Analysis_Summary_With_Stats.txt", 'w', encoding='utf-8') as f: f.write("\n".join(tmc_summary_lines))
        return

    df_all_norm_methods = get_all_normalized_methods(cache)
    if df_all_norm_methods.empty:
        print("No normalized method data to process for TMC. Aborting."); return

    print(f"\nIdentifying Top-{K_TOP_METHODS} normalized method names (core vocabulary)...")
    global_normalized_name_counts = Counter(df_all_norm_methods['Normalized_TMC_Name'])
    top_k_methods_with_counts = global_normalized_name_counts.most_common(K_TOP_METHODS)
    if not top_k_methods_with_counts:
        print(f"Error: Could not identify any top-{K_TOP_METHODS} methods."); return
    top_k_df = pd.DataFrame(top_k_methods_with_counts, columns=['Normalized_Method_Name', 'Global_Frequency'])
    top_k_set_normalized_benchmark = set(top_k_df['Normalized_Method_Name'])
    tmc_summary_lines.append(f"\nTop-{K_TOP_METHODS} Normalized Method Names (Lexical Benchmark):\n" + top_k_df.to_string(index=False))
    top_k_df.to_csv(STATS_TMC_OUTPUT_DIR / f"TMC_Top_{K_TOP_METHODS}_Normalized_Methods.csv", index=False)
    print(f"Saved Top-{K_TOP_METHODS} methods list.")

    # --- Calculate Per-Run Coverage of Top-K Benchmark ---
    print(f"\nCalculating per-run coverage of Top-{K_TOP_METHODS} normalized benchmark set...")
    per_run_coverage_data = []
    for (model_name_grp, run_id_grp), group in df_all_norm_methods.groupby([MODEL_COL, RUN_COL]):
        run_unique_normalized_names = set(group['Normalized_TMC_Name'])
        covered_top_k_methods_run = run_unique_normalized_names.intersection(top_k_set_normalized_benchmark)
        count_run = len(covered_top_k_methods_run)
        percentage_run = (count_run / K_TOP_METHODS) * 100 if K_TOP_METHODS > 0 else 0.0
        per_run_coverage_data.append({
            MODEL_COL: model_name_grp, RUN_COL: run_id_grp,
            f'TopK_Coverage_Percent_Run': percentage_run # Column for boxplot & stats
        })
    per_run_coverage_df = pd.DataFrame(per_run_coverage_data)

    # Aggregate coverage for Table tmc-overlap
    if not per_run_coverage_df.empty:
        llm_agg_coverage_df = per_run_coverage_df.groupby(MODEL_COL)[f'TopK_Coverage_Percent_Run'].mean().reset_index()
        llm_agg_coverage_df.rename(columns={f'TopK_Coverage_Percent_Run': 'Coverage (%)'}, inplace=True)
        
        # Order for the table based on LLM_ORDER_FOR_PLOTS
        models_in_agg_cov = llm_agg_coverage_df[MODEL_COL].unique()
        table_order_agg_cov = [m for m in LLM_ORDER_FOR_PLOTS if m in models_in_agg_cov]
        for m in models_in_agg_cov: # Add any missing from order list to the end
            if m not in table_order_agg_cov: table_order_agg_cov.append(m)
        
        llm_agg_coverage_df = llm_agg_coverage_df.set_index(MODEL_COL).reindex(table_order_agg_cov).dropna(how='all').reset_index()
        llm_agg_coverage_df = llm_agg_coverage_df.sort_values(by='Coverage (%)', ascending=False) # Final sort for display
        
        tmc_summary_lines.append("\nTable: Coverage of Top-37 normalized method names per model (Table tmc-overlap):\n" + llm_agg_coverage_df.round(1).to_string(index=False))
        print("Table: Coverage of Top-37 normalized method names per model (Table tmc-overlap):\n", llm_agg_coverage_df.round(1))
        llm_agg_coverage_df.to_csv(STATS_TMC_OUTPUT_DIR / "TMC_Table_tmc-overlap_Coverage.csv", index=False, float_format='%.1f')
    else:
        tmc_summary_lines.append("Warning: Per-run coverage data is empty for aggregate table.")

    # --- Kruskal-Wallis Test on Per-Run TMC Coverage Scores ---
    tmc_summary_lines.append("\n--- Kruskal-Wallis Test for Per-Run Top-K Coverage Scores ---")
    print("\n--- Kruskal-Wallis Test for Per-Run Top-K Coverage Scores ---")
    if not per_run_coverage_df.empty and f'TopK_Coverage_Percent_Run' in per_run_coverage_df.columns and \
       per_run_coverage_df[MODEL_COL].nunique() >= 2:
        df_for_kw_coverage = per_run_coverage_df.dropna(subset=[f'TopK_Coverage_Percent_Run'])
        valid_groups_for_kw_coverage = df_for_kw_coverage.groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
        if not valid_groups_for_kw_coverage.empty and valid_groups_for_kw_coverage[MODEL_COL].nunique() >= 2:
            kw_coverage_results = pg.kruskal(data=valid_groups_for_kw_coverage, dv=f'TopK_Coverage_Percent_Run', between=MODEL_COL)
            if kw_coverage_results is not None and not kw_coverage_results.empty:
                tmc_summary_lines.append("K-W Results (Per-Run Top-K Coverage %):\n" + kw_coverage_results.round(4).to_string())
                print(kw_coverage_results.round(4))
                p_value_kw_coverage = kw_coverage_results['p-unc'].iloc[0]
                if 'eps-sq' in kw_coverage_results.columns and pd.notna(kw_coverage_results['eps-sq'].iloc[0]):
                    tmc_summary_lines.append(f"  Effect Size (Epsilon-squared): {kw_coverage_results['eps-sq'].iloc[0]:.3f}")
                if p_value_kw_coverage < P_VALUE_THRESHOLD:
                    tmc_summary_lines.append("  K-W significant (Coverage). Dunn's post-hoc:")
                    print("  K-W significant (Coverage). Dunn's post-hoc:")
                    dunn_coverage_results = sp.posthoc_dunn(valid_groups_for_kw_coverage, val_col=f'TopK_Coverage_Percent_Run', group_col=MODEL_COL, p_adjust='bonferroni')
                    tmc_summary_lines.append("  Dunn's Test (Coverage):\n" + dunn_coverage_results.round(4).to_string())
                    print(dunn_coverage_results.round(4))
                else: tmc_summary_lines.append("  K-W not significant for per-run Top-K coverage %.")
            else: tmc_summary_lines.append("  K-W for per-run Top-K coverage % could not be computed.")
        else: tmc_summary_lines.append("  Not enough data/groups for K-W on per-run Top-K coverage % after filtering.")
    else: tmc_summary_lines.append("  Not enough data or model groups for K-W on per-run Top-K coverage %.")

    # --- Plot: Boxplot for Per-Run TMC Coverage (Figure tmc-boxplot) ---
    if not per_run_coverage_df.empty and f'TopK_Coverage_Percent_Run' in per_run_coverage_df.columns:
        print("\nGenerating Boxplot for Per-Run TMC Coverage (Figure tmc-boxplot)...")
        
        models_present_in_data_bp = per_run_coverage_df[MODEL_COL].unique()
        plot_order_boxplot_tmc = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_data_bp]
        for model in models_present_in_data_bp: # Add any missing from order list to the end
            if model not in plot_order_boxplot_tmc: plot_order_boxplot_tmc.append(model)
        plot_palette_boxplot_tmc = {model: LLM_COLORS.get(model, '#333333') for model in plot_order_boxplot_tmc}

        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(x=MODEL_COL, y=f'TopK_Coverage_Percent_Run', data=per_run_coverage_df, 
                         order=plot_order_boxplot_tmc, palette=plot_palette_boxplot_tmc, showfliers=True)
        
        plt.title(f'Distribution of Per-Run TMC Coverage (Top-{K_TOP_METHODS} Methods) by Model', fontsize=14)
        plt.xlabel('LLM Model', fontsize=12); plt.ylabel(f'Per-Run Coverage of Top-{K_TOP_METHODS} Methods (%)', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(np.arange(0, 101, 10)); plt.ylim(-5, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        tmc_boxplot_path = ARTICLE_DIR / "TMC_Boxplot_PerRun_Coverage.png"
        plt.savefig(tmc_boxplot_path); print(f"Saved TMC Per-Run Coverage boxplot to {tmc_boxplot_path}"); plt.close()
        tmc_summary_lines.append(f"\nBoxplot for Per-Run TMC Coverage saved to {tmc_boxplot_path}")
    else:
        print("Warning: Per-run coverage data is empty, skipping TMC boxplot.")
        tmc_summary_lines.append("Warning: Per-run coverage data is empty, skipping TMC boxplot.")

    # --- Per-Run Jaccard Similarity with Top-K Benchmark (for Figure tmc-jaccard-bar & K-W test) ---
    print("\nCalculating/Using per-run Jaccard similarity with Top-K benchmark...")
    per_run_jaccard_data_list = [] 
    for (model_name, run_id), group in df_all_norm_methods.groupby([MODEL_COL, RUN_COL]):
        run_unique_normalized_names = set(group['Normalized_TMC_Name'])
        j_score = jaccard_similarity(run_unique_normalized_names, top_k_set_normalized_benchmark)
        per_run_jaccard_data_list.append({ MODEL_COL: model_name, RUN_COL: run_id, 'Jaccard_vs_TopK_Benchmark': j_score })
    per_run_jaccard_df = pd.DataFrame(per_run_jaccard_data_list) 

    if per_run_jaccard_df.empty:
        print("Warning: Per-run Jaccard similarity data empty. Skipping Jaccard bar chart and stats.")
        tmc_summary_lines.append("Warning: Per-run Jaccard similarity data for bar chart/stats is empty.")
    else:
        print("\nGenerating Bar Chart for Mean Per-Run Jaccard Similarity (Figure tmc-jaccard-bar)...")
        mean_jaccard_per_model_df = per_run_jaccard_df.groupby(MODEL_COL)['Jaccard_vs_TopK_Benchmark'].mean().reset_index()
        
        models_present_in_jaccard = mean_jaccard_per_model_df[MODEL_COL].unique()
        plot_order_jaccard_bar = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_jaccard]
        for model in models_present_in_jaccard:
            if model not in plot_order_jaccard_bar: plot_order_jaccard_bar.append(model)
        
        mean_jaccard_per_model_df = mean_jaccard_per_model_df.set_index(MODEL_COL).reindex(plot_order_jaccard_bar).dropna(how='all').reset_index()
        mean_jaccard_per_model_df = mean_jaccard_per_model_df.sort_values(by='Jaccard_vs_TopK_Benchmark', ascending=False) # Sort for visual hierarchy
        
        ordered_colors_jaccard_bar = [LLM_COLORS.get(model, '#333333') for model in mean_jaccard_per_model_df[MODEL_COL]]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(mean_jaccard_per_model_df[MODEL_COL], mean_jaccard_per_model_df['Jaccard_vs_TopK_Benchmark'], color=ordered_colors_jaccard_bar)
        plt.xlabel("LLM Model", fontsize=12); plt.ylabel(f'Mean Jaccard Sim. with Top-{K_TOP_METHODS} Benchmark', fontsize=10)
        plt.title(f'Mean Per-Run Jaccard Similarity vs. Top-{K_TOP_METHODS} Norm. Methods', fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(np.arange(0, 1.1, 0.1)); plt.ylim(0, max(1.05, mean_jaccard_per_model_df['Jaccard_vs_TopK_Benchmark'].max() * 1.1 if not mean_jaccard_per_model_df.empty else 1.05 ))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(ARTICLE_DIR / "TMC_Bar_JaccardSimilarity.png"); print(f"Saved TMC Mean Jaccard Sim. bar chart."); plt.close()
        
        tmc_summary_lines.append("\n--- Kruskal-Wallis Test for Per-Run Jaccard Similarity with Top-K Benchmark ---")
        df_for_kw_jaccard = per_run_jaccard_df.dropna(subset=['Jaccard_vs_TopK_Benchmark'])
        valid_groups_for_kw_jaccard = df_for_kw_jaccard.groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
        if not valid_groups_for_kw_jaccard.empty and valid_groups_for_kw_jaccard[MODEL_COL].nunique() >= 2:
            kw_jaccard_results = pg.kruskal(data=valid_groups_for_kw_jaccard, dv='Jaccard_vs_TopK_Benchmark', between=MODEL_COL)
            if kw_jaccard_results is not None and not kw_jaccard_results.empty:
                tmc_summary_lines.append("K-W Results (Per-Run Jaccard vs Top-K):\n" + kw_jaccard_results.round(4).to_string())
                print("K-W Results (Per-Run Jaccard vs Top-K):\n", kw_jaccard_results.round(4))
                p_val_kw_jac = kw_jaccard_results['p-unc'].iloc[0]
                if 'eps-sq' in kw_jaccard_results.columns and pd.notna(kw_jaccard_results['eps-sq'].iloc[0]):
                    tmc_summary_lines.append(f"  Effect Size (Epsilon-squared): {kw_jaccard_results['eps-sq'].iloc[0]:.3f}")
                if p_val_kw_jac < P_VALUE_THRESHOLD:
                    tmc_summary_lines.append("  K-W significant (Jaccard). Dunn's post-hoc:")
                    print("  K-W significant (Jaccard). Dunn's post-hoc:")
                    dunn_jac_res = sp.posthoc_dunn(valid_groups_for_kw_jaccard, val_col='Jaccard_vs_TopK_Benchmark', group_col=MODEL_COL, p_adjust='bonferroni')
                    tmc_summary_lines.append("  Dunn's Test (Jaccard):\n" + dunn_jac_res.round(4).to_string())
                    print("  Dunn's Test Results for Jaccard:\n", dunn_jac_res.round(4))
        else: tmc_summary_lines.append("  Not enough data/groups for Kruskal-Wallis on per-run Jaccard after filtering.")

    summary_file_path = STATS_TMC_OUTPUT_DIR / "TMC_Analysis_Summary_And_Stats.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(tmc_summary_lines))
    print(f"\nSaved TMC analysis summary (with stats) to {summary_file_path}")
    print("\n--- Top-Method Consensus (TMC) Analysis Finished ---")

if __name__ == "__main__":
    if not METRIC_CACHE_AVAILABLE:
        print("Exiting script because MetricCache from main3.py could not be imported.")
    else:
        try: plt.style.use('seaborn-v0_8-whitegrid')
        except OSError: print("Warning: Seaborn style not found, using Matplotlib default.")
        main_tmc_analysis()