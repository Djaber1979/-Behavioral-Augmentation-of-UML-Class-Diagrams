#!/usr/bin/env python3
"""
RQ1_Core_Method_Consensus_CC.py

Analyzes Core-Method Consensus (CC):
1.  Identifies the Top-37 core methods based on global frequency of RAW method names
    from CoreMethods_TopN.csv.
2.  Generates a presence matrix (heatmap) of these core methods across LLMs (Fig. cc-heatmap).
3.  Calculates and plots aggregate coverage of core methods by each LLM (Fig. cc-bar).
4.  Calculates and plots agreement among LLMs for each core method (Fig. cc-agreement).
5.  Calculates per-run coverage of core methods, performs Kruskal-Wallis test,
    and generates a boxplot of these per-run coverages (Fig. cc-boxplot).

Reads from:
1.  reports/CoreMethods_TopN.csv
2.  reports/Annotation_and_Mapping_Combined.csv

Saves outputs to: reports/article/stats_cc/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Statistical libraries
from scipy import stats # For Kruskal-Wallis if pg.kruskal not used
import scikit_posthocs as sp
import pingouin as pg

# Configuration
REPORTS_DIR = Path("reports")
CORE_METHODS_INPUT_CSV = REPORTS_DIR / "CoreMethods_TopN.csv"
ALL_METHODS_INPUT_CSV = REPORTS_DIR / "Annotation_and_Mapping_Combined.csv" 

ARTICLE_DIR = REPORTS_DIR / "article"
STATS_CC_OUTPUT_DIR = ARTICLE_DIR / "stats_cc" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_CC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_CORE_METHODS = 37

# Column names from input CSVs
MODEL_COL_ALL_METHODS = 'Model' 
RUN_COL_ALL_METHODS = 'Run'     
RAW_METHOD_NAME_COL_ALL_METHODS = 'MethodName' 

CORE_METHOD_NAME_COL_CORE = 'MethodName' 
GLOBAL_FREQ_COL_CORE = 'GlobalFrequency' 

P_VALUE_THRESHOLD = 0.05

# --- Custom Color Palette and Order ---
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


def load_data(filepath: Path, required_cols: list) -> pd.DataFrame | None:
    if not filepath.is_file(): print(f"Error: Input CSV not found: {filepath}"); return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: DataFrame missing: {missing_cols}. Available: {df.columns.tolist()}"); return None
        return df
    except Exception as e: print(f"Error loading CSV from {filepath}: {e}"); return None

def main_cc_analysis():
    print("--- Core-Method Consensus (CC) Analysis ---")
    cc_summary_lines = ["--- Core-Method Consensus (CC) Analysis ---"]

    core_methods_df_raw = load_data(CORE_METHODS_INPUT_CSV, [CORE_METHOD_NAME_COL_CORE, GLOBAL_FREQ_COL_CORE])
    if core_methods_df_raw is None: print("Failed to load core methods. Aborting CC analysis."); return

    actual_k_core = min(K_CORE_METHODS, len(core_methods_df_raw))
    if actual_k_core < K_CORE_METHODS: print(f"Warning: Using K={actual_k_core} core methods.")
    print(f"Identifying Top-{actual_k_core} core methods (raw names)...")
    top_k_core_methods_df = core_methods_df_raw.head(actual_k_core)
    core_method_raw_names_set = set(top_k_core_methods_df[CORE_METHOD_NAME_COL_CORE].astype(str))
    if not core_method_raw_names_set: print("Error: Core method set (raw names) is empty."); return
    cc_summary_lines.append(f"\nTop-{actual_k_core} Core Method Names (Raw):\n" + top_k_core_methods_df[[CORE_METHOD_NAME_COL_CORE, GLOBAL_FREQ_COL_CORE]].to_string(index=False))
    top_k_core_methods_df[[CORE_METHOD_NAME_COL_CORE, GLOBAL_FREQ_COL_CORE]].to_csv(STATS_CC_OUTPUT_DIR / f"CC_Top_{actual_k_core}_Core_Raw_Methods.csv", index=False)

    all_methods_df = load_data(ALL_METHODS_INPUT_CSV, [MODEL_COL_ALL_METHODS, RUN_COL_ALL_METHODS, RAW_METHOD_NAME_COL_ALL_METHODS])
    if all_methods_df is None: print("Failed to load all method details. Aborting CC analysis."); return

    llm_unique_raw_method_sets = {} 
    for model_name, group in all_methods_df.groupby(MODEL_COL_ALL_METHODS):
        model_raw_names = set(group[RAW_METHOD_NAME_COL_ALL_METHODS].astype(str).dropna())
        llm_unique_raw_method_sets[str(model_name)] = {name for name in model_raw_names if name and str(name).strip()}
    if not llm_unique_raw_method_sets: print("Error: No unique raw method sets found per LLM."); return

    print(f"\nCalculating per-run coverage of Top-{actual_k_core} raw core methods...")
    per_run_core_coverage_data = []
    for (model_name, run_id), group in all_methods_df.groupby([MODEL_COL_ALL_METHODS, RUN_COL_ALL_METHODS]): 
        run_raw_names_set = set(group[RAW_METHOD_NAME_COL_ALL_METHODS].astype(str).dropna())
        run_raw_names_set = {name for name in run_raw_names_set if name and str(name).strip()} 
        covered_core_methods_run = run_raw_names_set.intersection(core_method_raw_names_set)
        count_run = len(covered_core_methods_run)
        percentage_run = (count_run / actual_k_core) * 100 if actual_k_core > 0 else 0.0
        per_run_core_coverage_data.append({
            MODEL_COL_ALL_METHODS: model_name, RUN_COL_ALL_METHODS: run_id,
            'Core_Coverage_Count_Run': count_run,
            'Core_Coverage_Percent_Run': percentage_run
        })
    per_run_core_coverage_df = pd.DataFrame(per_run_core_coverage_data)

    llm_agg_core_coverage_df = pd.DataFrame() 
    if not per_run_core_coverage_df.empty:
        llm_agg_core_coverage_df = per_run_core_coverage_df.groupby(MODEL_COL_ALL_METHODS)['Core_Coverage_Percent_Run'].mean().reset_index()
        llm_agg_core_coverage_df.rename(columns={'Core_Coverage_Percent_Run': 'Coverage_Percent', MODEL_COL_ALL_METHODS: 'Model'}, inplace=True)
        llm_agg_core_coverage_df = llm_agg_core_coverage_df.sort_values(by='Coverage_Percent', ascending=False)
        cc_summary_lines.append(f"\nLLM Aggregate Coverage of Top-{actual_k_core} Core Methods:\n" + llm_agg_core_coverage_df.to_string(index=False))
        print(f"LLM Aggregate Coverage of Top-{actual_k_core} Core Methods:\n", llm_agg_core_coverage_df)
        llm_agg_core_coverage_df.to_csv(STATS_CC_OUTPUT_DIR / f"CC_LLM_Aggregate_Core_Coverage.csv", index=False, float_format='%.1f')

        plt.figure(figsize=(10, max(6, len(llm_agg_core_coverage_df) * 0.5)))
        plot_df_cc_agg_coverage = llm_agg_core_coverage_df.sort_values(by='Coverage_Percent', ascending=True)
        
        # Create ordered colors for the aggregate coverage bar chart
        agg_bar_colors = [LLM_COLORS.get(model, "#333333") for model in plot_df_cc_agg_coverage['Model']]

        bars = plt.barh(plot_df_cc_agg_coverage['Model'], plot_df_cc_agg_coverage['Coverage_Percent'], color=agg_bar_colors)
        plt.xlabel(f'Coverage of Top-{actual_k_core} Core Methods (%)', fontsize=12)
        plt.ylabel('LLM Model', fontsize=12)
        plt.title(f'LLM Coverage of Top-{actual_k_core} Core Methods (Mean of Per-Run %)', fontsize=14)
        plt.xlim(0, 105); plt.grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars:
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2., f'{bar.get_width():.1f}%', ha='left', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(ARTICLE_DIR / "CC_Plot_LLM_Coverage.png"); print(f"Saved CC LLM Coverage plot (Fig cc-bar)."); plt.close()
    else:
        cc_summary_lines.append("Warning: Per-run core coverage data is empty, skipping aggregate table and plot for Figure cc-bar.")

    cc_summary_lines.append("\n--- Kruskal-Wallis Test for Per-Run Core Method Coverage (%) ---")
    print("\n--- Kruskal-Wallis Test for Per-Run Core Method Coverage (%) ---")
    if not per_run_core_coverage_df.empty and 'Core_Coverage_Percent_Run' in per_run_core_coverage_df.columns and \
       per_run_core_coverage_df[MODEL_COL_ALL_METHODS].nunique() >= 2:
        df_for_kw_cc_coverage = per_run_core_coverage_df.dropna(subset=['Core_Coverage_Percent_Run'])
        valid_groups_for_kw_cc = df_for_kw_cc_coverage.groupby(MODEL_COL_ALL_METHODS).filter(lambda x: len(x) >= 2)
        if not valid_groups_for_kw_cc.empty and valid_groups_for_kw_cc[MODEL_COL_ALL_METHODS].nunique() >= 2:
            kw_cc_cov_results = pg.kruskal(data=valid_groups_for_kw_cc, dv='Core_Coverage_Percent_Run', between=MODEL_COL_ALL_METHODS)
            if kw_cc_cov_results is not None and not kw_cc_cov_results.empty:
                cc_summary_lines.append("K-W Results (Per-Run Core Coverage %):\n" + kw_cc_cov_results.round(4).to_string())
                print(kw_cc_cov_results.round(4))
                p_val_kw_cc_cov = kw_cc_cov_results['p-unc'].iloc[0]
                if 'eps-sq' in kw_cc_cov_results.columns and pd.notna(kw_cc_cov_results['eps-sq'].iloc[0]):
                    cc_summary_lines.append(f"  Effect Size (Epsilon-squared): {kw_cc_cov_results['eps-sq'].iloc[0]:.3f}")
                if p_val_kw_cc_cov < P_VALUE_THRESHOLD:
                    cc_summary_lines.append("  K-W significant (Core Coverage). Dunn's post-hoc:")
                    print("  K-W significant (Core Coverage). Dunn's post-hoc:")
                    dunn_cc_cov_res = sp.posthoc_dunn(valid_groups_for_kw_cc, val_col='Core_Coverage_Percent_Run', group_col=MODEL_COL_ALL_METHODS, p_adjust='bonferroni')
                    cc_summary_lines.append("  Dunn's Test (Core Coverage):\n" + dunn_cc_cov_res.round(4).to_string())
                    print(dunn_cc_cov_res.round(4))
                else: cc_summary_lines.append("  K-W not significant for per-run core coverage %.")
            else: cc_summary_lines.append("  K-W for per-run core coverage could not be computed.")
        else: cc_summary_lines.append("  Not enough data/groups for K-W on per-run core coverage after filtering.")
    else: cc_summary_lines.append("  Not enough data or model groups for K-W on per-run core coverage.")

    # --- Plot: Boxplot for Per-Run Core Method Coverage (Figure cc-boxplot) ---
    if not per_run_core_coverage_df.empty and 'Core_Coverage_Percent_Run' in per_run_core_coverage_df.columns:
        print("\nGenerating Boxplot for Per-Run Core Method Coverage (Figure cc-boxplot)...")
        
        models_present_in_data_bp_cc = per_run_core_coverage_df[MODEL_COL_ALL_METHODS].unique()
        plot_order_boxplot_cc = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_data_bp_cc]
        for model in models_present_in_data_bp_cc:
            if model not in plot_order_boxplot_cc: plot_order_boxplot_cc.append(model)
        plot_palette_boxplot_cc = {model: LLM_COLORS.get(model, '#333333') for model in plot_order_boxplot_cc}

        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(
            x=MODEL_COL_ALL_METHODS, y='Core_Coverage_Percent_Run', 
            data=per_run_core_coverage_df, 
            order=plot_order_boxplot_cc, palette=plot_palette_boxplot_cc, 
            showfliers=True 
        )
        plt.title(f'Distribution of Per-Run Coverage of Top-{actual_k_core} Core Methods', fontsize=14)
        plt.xlabel('LLM Model', fontsize=12); plt.ylabel(f'Per-Run Coverage of Top-{actual_k_core} Core Methods (%)', fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(np.arange(0, 101, 10)); plt.ylim(-5, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        
        cc_boxplot_path = ARTICLE_DIR / "CC_Boxplot_PerRun_CoreCoverage.png" 
        plt.savefig(cc_boxplot_path); print(f"Saved CC Per-Run Core Coverage boxplot to {cc_boxplot_path}"); plt.close()
        cc_summary_lines.append(f"\nBoxplot for Per-Run CC Coverage saved to {cc_boxplot_path}")
    else:
        print("Warning: Per-run core coverage data is empty or missing required column, skipping CC boxplot.")
        cc_summary_lines.append("Warning: Per-run core coverage data is empty, skipping CC boxplot.")

    print(f"\nCalculating agreement among LLMs for Top-{actual_k_core} core methods...")
    core_method_agreement_data = []
    num_llms_total = len(llm_unique_raw_method_sets) 
    for core_method_name_raw in core_method_raw_names_set: 
        num_llms_generated_this_method = sum(1 for model_raw_names_set in llm_unique_raw_method_sets.values() if core_method_name_raw in model_raw_names_set)
        agreement_percent = (num_llms_generated_this_method / num_llms_total) * 100 if num_llms_total > 0 else 0
        global_freq_entry = top_k_core_methods_df[top_k_core_methods_df['Core_Method_Name'] == core_method_name_raw]['Global_Frequency'] # Use renamed col
        global_freq = global_freq_entry.iloc[0] if not global_freq_entry.empty else 'N/A'
        core_method_agreement_data.append({
            'Core_Method_Name': core_method_name_raw, 'Global_Frequency': global_freq,
            'Num_LLMs_Generated': num_llms_generated_this_method, 'Agreement_Percent': agreement_percent
        })
    core_method_agreement_df = pd.DataFrame(core_method_agreement_data)
    if not core_method_agreement_df.empty:
        core_method_agreement_df = core_method_agreement_df.sort_values(by='Num_LLMs_Generated', ascending=False)
        cc_summary_lines.append(f"\nAgreement Among LLMs for Top-{actual_k_core} Core Methods:\n" + core_method_agreement_df.to_string(index=False)) # Use to_string
        core_method_agreement_df.to_csv(STATS_CC_OUTPUT_DIR / f"CC_Core_Method_Agreement_Top_{actual_k_core}.csv", index=False, float_format='%.1f')
        plot_df_agreement_cc = core_method_agreement_df.sort_values(by='Num_LLMs_Generated', ascending=True).tail(min(len(core_method_agreement_df), 25))
        
        # Define colors for agreement bar chart based on number of models, not specific model names
        agreement_bar_colors = sns.color_palette("cubehelix_r", n_colors=len(plot_df_agreement_cc))


        plt.figure(figsize=(10, max(6, len(plot_df_agreement_cc) * 0.35)))
        bars = plt.barh(plot_df_agreement_cc['Core_Method_Name'], plot_df_agreement_cc['Num_LLMs_Generated'], color=agreement_bar_colors)
        plt.xlabel('Number of LLMs Generating the Method', fontsize=12); plt.ylabel('Core Method Name', fontsize=12)
        plt.title(f'Agreement for Top-{actual_k_core} Core Methods (Top {len(plot_df_agreement_cc)})', fontsize=14)
        plt.xticks(ticks=np.arange(0, num_llms_total + 2, 1)); plt.grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars: plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,f'{int(bar.get_width())}', ha='left', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(ARTICLE_DIR / "CC_Plot_Method_Agreement.png"); print(f"Saved CC Method Agreement plot."); plt.close()

    if not llm_agg_core_coverage_df.empty and not core_method_agreement_df.empty: 
        print(f"\nGenerating Presence Heatmap for Top-{actual_k_core} Core Methods...")
        heatmap_methods_order = top_k_core_methods_df['Core_Method_Name'].tolist() 
        heatmap_models_order = llm_agg_core_coverage_df['Model'].tolist() 
        presence_matrix_data = []
        for method_name in heatmap_methods_order: 
            method_name_str = str(method_name) 
            row_data = {'Core_Method_Name': method_name_str} # Use the renamed 'Core_Method_Name'
            for llm_name in heatmap_models_order:
                llm_set = llm_unique_raw_method_sets.get(str(llm_name), set())
                row_data[str(llm_name)] = 1 if method_name_str in llm_set else 0
            presence_matrix_data.append(row_data)
        presence_heatmap_df = pd.DataFrame(presence_matrix_data)
        if not presence_heatmap_df.empty:
            presence_heatmap_df = presence_heatmap_df.set_index('Core_Method_Name') 
            valid_heatmap_models_order = [m for m in heatmap_models_order if m in presence_heatmap_df.columns]
            if valid_heatmap_models_order:
                presence_heatmap_df = presence_heatmap_df[valid_heatmap_models_order]
                plt.figure(figsize=(max(10, len(valid_heatmap_models_order) * 0.7), max(8, len(heatmap_methods_order) * 0.25)))
                sns.heatmap(presence_heatmap_df, cmap="Blues", cbar=False, linewidths=.5, linecolor='lightgrey', annot=False)
                plt.title(f'Presence Matrix of Top-{actual_k_core} Core Methods Across LLMs', fontsize=14)
                plt.xlabel('LLM Model (Ordered by Core Method Coverage)', fontsize=12)
                plt.ylabel(f'Top-{actual_k_core} Core Method Names (Ordered by Global Frequency)', fontsize=12)
                plt.xticks(rotation=45, ha="right", fontsize=9); 
                ytick_fontsize = max(5, 10 - int(len(heatmap_methods_order) / 8) ) 
                plt.yticks(fontsize=ytick_fontsize) 
                plt.tight_layout()
                plt.savefig(ARTICLE_DIR / "CC_Plot_Presence_Heatmap.png"); print(f"Saved CC Presence Heatmap."); plt.close()
    else:
        print("Skipping Presence Heatmap due to empty dataframes.")
        cc_summary_lines.append("Skipping Presence Heatmap due to empty dataframes.")

    summary_file_path = STATS_CC_OUTPUT_DIR / "CC_Analysis_Summary_And_Stats.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(cc_summary_lines))
    print(f"\nSaved CC analysis summary (with stats) to {summary_file_path}")
    print("\n--- Core-Method Consensus (CC) Analysis Finished ---")

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError: print("Warning: Seaborn style not found.")
    main_cc_analysis()