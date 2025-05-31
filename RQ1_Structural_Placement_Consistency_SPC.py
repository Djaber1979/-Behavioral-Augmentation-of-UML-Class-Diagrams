#!/usr/bin/env python3
"""
RQ1_Structural_Placement_Consistency_SPC.py

Analyzes Structural Placement Consistency (SPC) for core methods:
1.  Identifies the Top-37 core methods (raw names) from CoreMethods_TopN.csv.
2.  For each core method, determines its dominant class assignment across all diagrams.
3.  For each core method, calculates the placement consistency rate:
    (LLMs placing it in dominant class) / (LLMs generating it at all).
4.  Generates a histogram/density plot for the distribution of these placement consistency rates.
5.  Outputs data to help construct the UML diagram summarizing core methods by dominant class.
6.  Performs One-Sample Wilcoxon test on placement consistency rates.
7.  Performs Spearman correlation between global frequency and placement consistency.

Reads from:
1.  reports/CoreMethods_TopN.csv
2.  reports/Annotation_and_Mapping_Combined.csv

Saves outputs to: reports/article/stats_spc/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Statistical libraries
from scipy import stats
# No specific post-hoc needed for one-sample Wilcoxon or Spearman here

# Configuration
REPORTS_DIR = Path("reports")
CORE_METHODS_INPUT_CSV = REPORTS_DIR / "CoreMethods_TopN.csv"
ALL_METHODS_INPUT_CSV = REPORTS_DIR / "Annotation_and_Mapping_Combined.csv" 

ARTICLE_DIR = REPORTS_DIR / "article"
STATS_SPC_OUTPUT_DIR = ARTICLE_DIR / "stats_spc" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_SPC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_CORE_METHODS: int = 37 

# Column names from input CSVs
MODEL_COL_ALL_METHODS: str = 'Model' 
CLASS_COL_ALL_METHODS: str = 'Class'
RAW_METHOD_NAME_COL_ALL_METHODS: str = 'MethodName' 

CORE_METHOD_NAME_COL_CORE: str = 'MethodName' 
GLOBAL_FREQ_COL_CORE: str = 'GlobalFrequency'

P_VALUE_THRESHOLD: float = 0.05
PERFECT_SCORE_THRESHOLD: float = 99.999

def load_data(filepath: Path, required_cols: list) -> pd.DataFrame | None:
    """Loads CSV data and checks for required columns."""
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: DataFrame from {filepath} is missing required columns: {missing_cols}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def main_spc_analysis():
    print("--- Structural Placement Consistency (SPC) Analysis ---")
    spc_summary_lines = ["--- Structural Placement Consistency (SPC) Analysis ---"]

    # 1. Load CoreMethods_TopN.csv to identify the core method set (RAW NAMES)
    core_methods_df_raw = load_data(CORE_METHODS_INPUT_CSV, 
                                 [CORE_METHOD_NAME_COL_CORE, GLOBAL_FREQ_COL_CORE])
    if core_methods_df_raw is None: 
        print("Failed to load core methods data. Aborting SPC analysis.")
        return

    actual_k_core = min(K_CORE_METHODS, len(core_methods_df_raw))
    if actual_k_core < K_CORE_METHODS:
        print(f"Warning: Requested K_CORE_METHODS={K_CORE_METHODS} but only {len(core_methods_df_raw)} methods found in {CORE_METHODS_INPUT_CSV}.")
    print(f"Using Top-{actual_k_core} core methods (raw names) for SPC, based on GlobalFrequency.")
    
    # Assuming CoreMethods_TopN.csv is already sorted by GlobalFrequency descending. 
    # If not, uncomment: core_methods_df_raw = core_methods_df_raw.sort_values(by=GLOBAL_FREQ_COL_CORE, ascending=False)
    top_k_core_methods_df = core_methods_df_raw.head(actual_k_core).copy() # Use .copy()
    # Ensure the column for merging has a consistent name if it's different from RAW_METHOD_NAME_COL_ALL_METHODS
    # For this script, we assume it's 'MethodName' in CoreMethods_TopN.csv
    top_k_core_methods_df.rename(columns={CORE_METHOD_NAME_COL_CORE: 'Core_Method_Name'}, inplace=True)
    core_method_raw_names_set = set(top_k_core_methods_df['Core_Method_Name'].astype(str))
    
    if not core_method_raw_names_set:
        print("Error: Core method set (raw names) is empty. Cannot proceed."); return

    # 2. Load Annotation_and_Mapping_Combined.csv for all method placements
    all_methods_df = load_data(ALL_METHODS_INPUT_CSV, 
                               [MODEL_COL_ALL_METHODS, RAW_METHOD_NAME_COL_ALL_METHODS, CLASS_COL_ALL_METHODS])
    if all_methods_df is None: 
        print("Failed to load all method details. Aborting SPC analysis."); return

    # Filter all_methods_df to only include instances of the core methods
    # Ensure we are comparing strings to strings for isin
    df_core_method_instances = all_methods_df[
        all_methods_df[RAW_METHOD_NAME_COL_ALL_METHODS].astype(str).isin(list(core_method_raw_names_set)) # Pass list to isin
    ].copy()
    
    if df_core_method_instances.empty:
        print("Error: No instances of core methods found in the all_methods_df. Cannot proceed."); return
        
    df_core_method_instances.dropna(subset=[CLASS_COL_ALL_METHODS], inplace=True)
    df_core_method_instances[CLASS_COL_ALL_METHODS] = df_core_method_instances[CLASS_COL_ALL_METHODS].astype(str)
    # Rename columns for clarity if they differ from the general 'Model', 'MethodName'
    df_core_method_instances.rename(columns={MODEL_COL_ALL_METHODS: 'Model', 
                                             RAW_METHOD_NAME_COL_ALL_METHODS: 'MethodName',
                                             CLASS_COL_ALL_METHODS: 'Class'}, inplace=True)


    # 3. Determine Dominant Class and Calculate Placement Consistency for Each Core Method
    print("\nCalculating dominant class and placement consistency for each core method...")
    placement_consistency_data = []
    core_method_dominant_class_map = {} 

    for core_method_name in core_method_raw_names_set: # Iterate using the set of unique core method names
        instances_of_this_method = df_core_method_instances[
            df_core_method_instances['MethodName'].astype(str) == core_method_name
        ]
        
        if instances_of_this_method.empty:
            placement_consistency_data.append({
                'Core_Method_Name': core_method_name, 'Dominant_Class': 'N/A (Not Generated by any LLM)',
                'Num_LLMs_Generating_Method': 0, 'Num_LLMs_Placing_In_Dominant_Class': 0,
                'Placement_Consistency_Rate (%)': 0.0, 'Dominant_Class_Instance_Count': 0,
                'Total_Instances_of_Method':0
            })
            core_method_dominant_class_map[core_method_name] = 'N/A (Not Generated by any LLM)'
            continue

        class_counts = instances_of_this_method['Class'].value_counts()
        dominant_class = class_counts.index[0] if not class_counts.empty else 'N/A (No Class Info)'
        dominant_class_instance_count = class_counts.iloc[0] if not class_counts.empty else 0
        core_method_dominant_class_map[core_method_name] = dominant_class

        llms_generating_method = set(instances_of_this_method['Model'].unique())
        num_llms_generating_method = len(llms_generating_method)

        llms_placing_in_dominant = set(
            instances_of_this_method[instances_of_this_method['Class'] == dominant_class]['Model'].unique()
        )
        num_llms_placing_in_dominant = len(llms_placing_in_dominant)

        consistency_rate = (num_llms_placing_in_dominant / num_llms_generating_method) * 100 if num_llms_generating_method > 0 else 0.0
        
        placement_consistency_data.append({
            'Core_Method_Name': core_method_name, 'Dominant_Class': dominant_class,
            'Num_LLMs_Generating_Method': num_llms_generating_method,
            'Num_LLMs_Placing_In_Dominant_Class': num_llms_placing_in_dominant,
            'Placement_Consistency_Rate (%)': consistency_rate,
            'Dominant_Class_Instance_Count': dominant_class_instance_count,
            'Total_Instances_of_Method': len(instances_of_this_method)
        })

    placement_consistency_df = pd.DataFrame(placement_consistency_data)
    
    if not placement_consistency_df.empty:
        # Merge with global frequency from top_k_core_methods_df
        # Ensure top_k_core_methods_df has 'Core_Method_Name' and 'Global_Frequency'
        # (It was renamed from CORE_METHOD_NAME_COL_CORE to 'Core_Method_Name' earlier)
        placement_consistency_df = pd.merge(
            placement_consistency_df,
            top_k_core_methods_df[['Core_Method_Name', GLOBAL_FREQ_COL_CORE]], # GLOBAL_FREQ_COL_CORE is 'GlobalFrequency'
            on='Core_Method_Name',
            how='left'
        )
        placement_consistency_df.rename(columns={GLOBAL_FREQ_COL_CORE: 'Global_Frequency'}, inplace=True)


        placement_consistency_df = placement_consistency_df.sort_values(by='Placement_Consistency_Rate (%)', ascending=False)
        spc_summary_lines.append("\nPlacement Consistency per Core Method:\n" + placement_consistency_df.to_string(index=False))
        print("Placement Consistency per Core Method:\n", placement_consistency_df)
        spc_csv_path = STATS_SPC_OUTPUT_DIR / "SPC_Placement_Consistency_Per_Core_Method.csv"
        placement_consistency_df.to_csv(spc_csv_path, index=False, float_format='%.2f')
        print(f"Saved SPC per core method data to {spc_csv_path}")

        avg_placement_rate = placement_consistency_df['Placement_Consistency_Rate (%)'].mean()
        num_100_percent_placement = (placement_consistency_df['Placement_Consistency_Rate (%)'] >= PERFECT_SCORE_THRESHOLD).sum()
        stat_msg = (f"\nAverage Placement Match Rate: {avg_placement_rate:.1f}%\n"
                    f"{num_100_percent_placement} out of {actual_k_core} methods placed in dominant class by all generating models (>= {PERFECT_SCORE_THRESHOLD}% consistency).")
        print(stat_msg); spc_summary_lines.append(stat_msg)

        print("\nGenerating Distribution Plot for Placement Consistency Rates (Figure cc-placement-distribution)...")
        plt.figure(figsize=(8, 6))
        sns.histplot(placement_consistency_df['Placement_Consistency_Rate (%)'].dropna(), bins=10, kde=True, stat="density")
        plt.title(f'Distribution of Placement Consistency Rates\n(Across {actual_k_core} Core Methods)', fontsize=14)
        plt.xlabel('Placement Consistency Rate (%)', fontsize=12); plt.ylabel('Density', fontsize=12)
        plt.xlim(-5, 105); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        spc_dist_plot_path = ARTICLE_DIR / "CC_Plot_Distribution_Placement_MatchRate.png"
        plt.savefig(spc_dist_plot_path); print(f"Saved Placement Consistency distribution plot to {spc_dist_plot_path}"); plt.close()

        # --- Perform Inferential Statistical Test on Placement Consistency Rates ---
        spc_summary_lines.append("\n--- Inferential Statistics on Placement Consistency Rates ---")
        print("\n--- Inferential Statistics on Placement Consistency Rates ---")
        consistency_rates = placement_consistency_df['Placement_Consistency_Rate (%)'].dropna()
        if len(consistency_rates) >= 8: # Wilcoxon typically needs a reasonable number of samples
            target_median_test = 90.0 
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(consistency_rates - target_median_test, alternative='two-sided', zero_method='wilcox')
                msg_test = (f"One-Sample Wilcoxon Signed-Rank Test (vs. median of {target_median_test}%):\n"
                            f"  Statistic={wilcoxon_stat:.3f}, p-value={wilcoxon_p:.4f}")
                print(msg_test); spc_summary_lines.append(msg_test)

                target_median_100 = 100.0
                rates_lt_100 = consistency_rates[consistency_rates < PERFECT_SCORE_THRESHOLD] # Use threshold
                if len(rates_lt_100) >= 8:
                    wilcoxon_stat_lt100, wilcoxon_p_lt100 = stats.wilcoxon(rates_lt_100 - target_median_100, alternative='less', zero_method='wilcox')
                    msg_lt100 = (f"One-Sample Wilcoxon Test (rates < {PERFECT_SCORE_THRESHOLD}% vs. {target_median_100}%; H1: median is less than 100%):\n"
                                 f"  Statistic={wilcoxon_stat_lt100:.3f}, p-value={wilcoxon_p_lt100:.4f} (N={len(rates_lt_100)})")
                    print(msg_lt100); spc_summary_lines.append(msg_lt100)
            except ValueError as e_wilcoxon:
                 msg_wilcoxon_err = f"Wilcoxon test could not be performed: {e_wilcoxon}"
                 print(msg_wilcoxon_err); spc_summary_lines.append(msg_wilcoxon_err)
        else:
            spc_summary_lines.append("Not enough data points (<8) for reliable One-Sample Wilcoxon Signed-Rank Test.")

        if 'Global_Frequency' in placement_consistency_df.columns:
            df_for_corr = placement_consistency_df[['Global_Frequency', 'Placement_Consistency_Rate (%)']].dropna()
            if len(df_for_corr) >= 5:
                spearman_corr, spearman_p = stats.spearmanr(df_for_corr['Global_Frequency'], df_for_corr['Placement_Consistency_Rate (%)'])
                msg_corr = (f"\nSpearman Correlation (Global Frequency vs. Placement Consistency Rate):\n"
                            f"  Correlation Coefficient={spearman_corr:.3f}, p-value={spearman_p:.4f}")
                print(msg_corr); spc_summary_lines.append(msg_corr)
    else:
        print("Warning: Placement Consistency DataFrame is empty. Skipping stats and plots.")
        spc_summary_lines.append("Warning: Placement Consistency DataFrame is empty.")

    # --- Data for Figure~\ref{fig:core-methods-full-uml} ---
    print("\nGenerating data for UML diagram (Dominant Class Assignments)...")
    if not placement_consistency_df.empty:
        # ... (UML data generation logic remains same as previous version) ...
        uml_data = placement_consistency_df[['Core_Method_Name', 'Dominant_Class', 'Placement_Consistency_Rate (%)']].copy()
        uml_data['Is_100_Percent_Consistent'] = uml_data['Placement_Consistency_Rate (%)'] >= PERFECT_SCORE_THRESHOLD
        dominant_class_summary = defaultdict(list)
        for _, row_uml in uml_data.iterrows(): # Renamed loop variable
            dominant_class_summary[row_uml['Dominant_Class']].append({
                'method': row_uml['Core_Method_Name'],
                'consistent_placement': row_uml['Is_100_Percent_Consistent']
            })
        uml_output_lines = ["--- Data for UML Diagram (Figure core-methods-full-uml) ---"]
        print("Data for UML Diagram (Figure core-methods-full-uml):")
        for dom_class, methods_list in dominant_class_summary.items(): # Renamed loop variable
            if str(dom_class).startswith('N/A'): continue
            class_all_methods_consistent = all(m['consistent_placement'] for m in methods_list)
            uml_output_lines.append(f"\nClass: {dom_class} {'(All its core methods 100% consistently placed by generating LLMs)' if class_all_methods_consistent else ''}")
            print(f"\nClass: {dom_class} {'(All its core methods 100% consistently placed by generating LLMs)' if class_all_methods_consistent else ''}")
            for method_detail in methods_list:
                uml_output_lines.append(f"  - Method: {method_detail['method']} {'(100% consistent)' if method_detail['consistent_placement'] else ''}")
                print(f"  - Method: {method_detail['method']} {'(100% consistent)' if method_detail['consistent_placement'] else ''}")
        spc_summary_lines.extend(uml_output_lines)
        with open(STATS_SPC_OUTPUT_DIR / "SPC_Dominant_Class_Assignments_for_UML.txt", 'w', encoding='utf-8') as f_uml:
            f_uml.write("\n".join(uml_output_lines))
        print(f"Saved dominant class assignment data for UML to {STATS_SPC_OUTPUT_DIR / 'SPC_Dominant_Class_Assignments_for_UML.txt'}")
    else:
        print("Cannot generate UML data as placement consistency data is empty.")
        spc_summary_lines.append("Cannot generate UML data as placement consistency data is empty.")

    summary_file_path = STATS_SPC_OUTPUT_DIR / "SPC_Analysis_Summary.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(spc_summary_lines))
    print(f"\nSaved SPC analysis summary to {summary_file_path}")

    print("\n--- Structural Placement Consistency (SPC) Analysis Finished ---")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main_spc_analysis()