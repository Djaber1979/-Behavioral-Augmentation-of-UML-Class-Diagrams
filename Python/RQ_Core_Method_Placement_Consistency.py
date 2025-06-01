#!/usr/bin/env python3
"""
RQ_Core_Method_Placement_Consistency.py

Calculates and visualizes the consistency of class placement for core methods.

The number of core methods (k) is dynamically calculated as:
k = ceil( (total method instances across all diagrams) / (total number of diagrams) )
Total diagrams are determined dynamically from unique Model-Run pairs, fallback to 90.

For each method in the Core Method set (top-k most frequent):
1. Identifies the most common class (Dominant Class) it was placed into across all LLMs.
2. Computes the MatchRate:
   MatchRate_i = (# LLMs assigning method_i to Dominant_Class_i) / (Total LLMs generating method_i) * 100%
3. Calculates the average MatchRate over all top-k core methods.

Generates:
1.  Table: Dominant Class Placement per Core Method (CC_Core_Method_Placement_Consistency.csv)
2.  Table: Grouped Summary Statistics by Dominant Class (CC_Summary_Placement_By_Dominant_Class.csv)
3.  Plot: Bar Chart of MatchRate_Percent per Core Method
4.  Plot: Histogram/Distribution of MatchRate_Percent

Reads from:
1.  reports/CoreMethods_TopN.csv (to identify the core/top N methods)
2.  reports/Annotation_and_Mapping_Combined.csv (for method instances, classes, and models)

Saves outputs to: reports/article/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import math # For math.ceil

# Configuration
REPORTS_DIR = Path("reports")
CORE_METHODS_INPUT_CSV = "CoreMethods_TopN.csv"
ALL_METHODS_INPUT_CSV = "Annotation_and_Mapping_Combined.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

# N_CORE_METHODS will be calculated dynamically

# Column names
MODEL_COL = 'Model'
METHOD_NAME_COL = 'MethodName'
CLASS_COL = 'Class'
CORE_METHOD_NAME_COL = 'MethodName' # From CoreMethods_TopN.csv
GLOBAL_FREQ_COL = 'GlobalFrequency' # From CoreMethods_TopN.csv


def load_data(filepath: Path, required_cols: list) -> pd.DataFrame | None:
    """Loads CSV data and checks for required columns."""
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data from {filepath}")
        if not all(col in df.columns for col in required_cols):
            print(f"Error: DataFrame from {filepath} is missing required columns: {required_cols}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def main():
    print("Starting Core Method Placement Consistency analysis...")

    all_methods_df = load_data(REPORTS_DIR / ALL_METHODS_INPUT_CSV, [MODEL_COL, METHOD_NAME_COL, CLASS_COL])
    if all_methods_df is None: return

    total_method_instances = len(all_methods_df)
    if 'Run' in all_methods_df.columns and MODEL_COL in all_methods_df.columns:
        total_diagrams = all_methods_df.groupby([MODEL_COL, 'Run']).ngroups
        print(f"Dynamically determined total diagrams (unique Model-Run pairs): {total_diagrams}")
        if total_diagrams == 0 :
            print("Warning: Could not determine total diagrams dynamically, defaulting to 90.")
            total_diagrams = 90 
    else:
        print("Warning: 'Run' column not found for dynamic total_diagrams calculation. Defaulting to 90.")
        total_diagrams = 90

    if total_diagrams == 0:
        print("Error: Total number of diagrams is zero. Cannot calculate N_CORE_METHODS.")
        return
        
    n_core_methods_float = total_method_instances / total_diagrams
    N_CORE_METHODS = math.ceil(n_core_methods_float)
    print(f"Total method instances: {total_method_instances}")
    print(f"Total diagrams: {total_diagrams}")
    print(f"Dynamically calculated N_CORE_METHODS (k) = ceil({total_method_instances}/{total_diagrams}) = ceil({n_core_methods_float:.2f}) = {N_CORE_METHODS}")

    core_methods_df_full = load_data(REPORTS_DIR / CORE_METHODS_INPUT_CSV, 
                                [CORE_METHOD_NAME_COL, GLOBAL_FREQ_COL])
    if core_methods_df_full is None: return

    actual_n_core = min(N_CORE_METHODS, len(core_methods_df_full))
    if actual_n_core < N_CORE_METHODS:
        print(f"Warning: Calculated N_CORE_METHODS={N_CORE_METHODS} but only {len(core_methods_df_full)} unique methods found in {CORE_METHODS_INPUT_CSV}.")
        print(f"Using N={actual_n_core} for core set.")
    
    core_method_set_df = core_methods_df_full.head(actual_n_core)
    core_method_names_list = core_method_set_df[CORE_METHOD_NAME_COL].astype(str).tolist()
    
    if not core_method_names_list:
        print("Error: Core method set is empty. Cannot proceed.")
        return
    print(f"Identified {len(core_method_names_list)} core methods for analysis.")

    all_methods_df[METHOD_NAME_COL] = all_methods_df[METHOD_NAME_COL].astype(str)
    core_method_instances_df = all_methods_df[all_methods_df[METHOD_NAME_COL].isin(core_method_names_list)].copy()

    if core_method_instances_df.empty:
        print("Error: No instances of the selected core methods found. Cannot proceed.")
        return

    print("\nCalculating Dominant Class Placement Consistency per Core Method...")
    placement_consistency_data = []
    for core_method_name in core_method_names_list:
        instances_of_this_method = core_method_instances_df[core_method_instances_df[METHOD_NAME_COL] == core_method_name]
        if instances_of_this_method.empty:
            placement_consistency_data.append({
                'Core_Method_Name': core_method_name,
                'Global_Frequency': core_method_set_df.loc[core_method_set_df[CORE_METHOD_NAME_COL] == core_method_name, GLOBAL_FREQ_COL].iloc[0] if core_method_name in core_method_set_df[CORE_METHOD_NAME_COL].values else 'N/A',
                'Dominant_Class': 'N/A (Not Found in Annotations)',
                'Num_Instances_In_Dominant_Class': 0,
                'Total_Instances_Generated_For_Core_Method': 0,
                'Num_LLMs_Generated_Method': 0,
                'Num_LLMs_Matched_Dominant_Class': 0,
                'MatchRate_Percent': np.nan
            })
            continue

        llms_generating_method = instances_of_this_method[MODEL_COL].unique()
        total_llms_generated_method_i = len(llms_generating_method)
        class_counts = instances_of_this_method[CLASS_COL].value_counts()
        dominant_class = class_counts.index[0] if not class_counts.empty else "N/A (No Placements)"
        num_instances_in_dominant_class = class_counts.iloc[0] if not class_counts.empty else 0
        total_instances_generated = len(instances_of_this_method)
        num_llms_matched_dominant_class = 0
        if dominant_class != "N/A (No Placements)" and total_llms_generated_method_i > 0:
            for llm in llms_generating_method:
                if dominant_class in instances_of_this_method[instances_of_this_method[MODEL_COL] == llm][CLASS_COL].unique():
                    num_llms_matched_dominant_class += 1
        
        match_rate_percent = (num_llms_matched_dominant_class / total_llms_generated_method_i) * 100 if total_llms_generated_method_i > 0 else np.nan
        global_freq_entry = core_method_set_df[core_method_set_df[CORE_METHOD_NAME_COL] == core_method_name][GLOBAL_FREQ_COL]
        global_freq = global_freq_entry.iloc[0] if not global_freq_entry.empty else 'N/A'
        placement_consistency_data.append({
            'Core_Method_Name': core_method_name,
            'Global_Frequency': global_freq,
            'Dominant_Class': dominant_class,
            'Num_Instances_In_Dominant_Class': num_instances_in_dominant_class,
            'Total_Instances_Generated_For_Core_Method': total_instances_generated,
            'Num_LLMs_Generated_Method': total_llms_generated_method_i,
            'Num_LLMs_Matched_Dominant_Class': num_llms_matched_dominant_class,
            'MatchRate_Percent': match_rate_percent
        })
    placement_consistency_df = pd.DataFrame(placement_consistency_data)
    
    overall_avg_match_rate = np.nan
    if not placement_consistency_df.empty:
        placement_consistency_df = placement_consistency_df.sort_values(by='MatchRate_Percent', ascending=False, na_position='last')
        placement_output_path = ARTICLE_DIR / "CC_Core_Method_Placement_Consistency.csv"
        placement_consistency_df.to_csv(placement_output_path, index=False, float_format='%.2f')
        print(f"Saved Core Method Placement Consistency table to {placement_output_path}")
        valid_match_rates = placement_consistency_df['MatchRate_Percent'].dropna()
        if not valid_match_rates.empty:
            overall_avg_match_rate = valid_match_rates.mean()
            print(f"\nOverall Average Dominant Class Placement MatchRate (for {len(valid_match_rates)} core methods): {overall_avg_match_rate:.2f}%")
        else:
            print("\nNo valid MatchRates to calculate an overall average.")
    else:
        print("Warning: Core Method Placement Consistency table is empty.")

    # --- New: Generate Grouped Summary Statistics by Dominant Class (Table 3) ---
    if not placement_consistency_df.empty and 'Dominant_Class' in placement_consistency_df.columns:
        print("\nCalculating Grouped Summary Statistics by Dominant Class...")
        # Filter out rows where Dominant_Class might be 'N/A ...' or MatchRate is NaN for meaningful stats
        df_for_class_summary = placement_consistency_df[
            ~placement_consistency_df['Dominant_Class'].str.startswith('N/A', na=False) &
            placement_consistency_df['MatchRate_Percent'].notna()
        ].copy()

        if not df_for_class_summary.empty:
            summary_by_dominant_class = df_for_class_summary.groupby('Dominant_Class')['MatchRate_Percent'].agg(
                num_core_methods_dominated='count',
                avg_match_rate='mean',
                min_match_rate='min',
                median_match_rate='median',
                max_match_rate='max',
                std_match_rate='std'
            ).sort_values(by='num_core_methods_dominated', ascending=False)
            
            # Add a column for the actual core method names that have this class as dominant
            # This can make the table very wide if lists are long.
            # dominant_class_core_methods = df_for_class_summary.groupby('Dominant_Class')['Core_Method_Name'].apply(lambda x: ', '.join(x.astype(str).unique()[:3]) + ('...' if len(x.astype(str).unique()) > 3 else ''))
            # summary_by_dominant_class = summary_by_dominant_class.join(dominant_class_core_methods.rename('Example_Core_Methods'))


            summary_by_class_output_path = ARTICLE_DIR / "CC_Summary_Placement_By_Dominant_Class.csv"
            summary_by_dominant_class.to_csv(summary_by_class_output_path, float_format='%.2f')
            print(f"Saved Summary by Dominant Class table to {summary_by_class_output_path}")
        else:
            print("No valid data for generating summary statistics by dominant class after filtering.")
    else:
        print("Skipping summary statistics by dominant class as placement_consistency_df is empty or missing 'Dominant_Class'.")


    # --- Generate Plots ---
    if not placement_consistency_df.empty and 'MatchRate_Percent' in placement_consistency_df.columns:
        plot_df_match_rate = placement_consistency_df.dropna(subset=['MatchRate_Percent']).copy()
        
        if not plot_df_match_rate.empty:
            print("\nGenerating Plot 1: MatchRate per Core Method...")
            plot_df_match_rate_subset = plot_df_match_rate.sort_values(by='MatchRate_Percent', ascending=True).tail(min(len(plot_df_match_rate), 25))
            if not plot_df_match_rate_subset.empty:
                plt.figure(figsize=(12, max(6, len(plot_df_match_rate_subset) * 0.4)))
                bars = plt.barh(plot_df_match_rate_subset['Core_Method_Name'], 
                                plot_df_match_rate_subset['MatchRate_Percent'],
                                color=sns.color_palette("coolwarm_r", len(plot_df_match_rate_subset)))
                plt.xlabel('Dominant Class Placement MatchRate (%)', fontsize=12)
                plt.ylabel('Core Method Name', fontsize=12)
                plt.title(f'Top-{len(plot_df_match_rate_subset)} Core Methods by Placement Consistency (N={actual_n_core})', fontsize=14)
                plt.xlim(0, 105)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                for i, bar in enumerate(bars):
                    dominant_class_text = str(plot_df_match_rate_subset['Dominant_Class'].iloc[i])
                    max_len_class = 20
                    if len(dominant_class_text) > max_len_class:
                        dominant_class_text = dominant_class_text[:max_len_class-3] + "..."
                    num_matched = int(plot_df_match_rate_subset['Num_LLMs_Matched_Dominant_Class'].iloc[i])
                    num_generated = int(plot_df_match_rate_subset['Num_LLMs_Generated_Method'].iloc[i])
                    annotation_text = f'{bar.get_width():.1f}% (Dom: {dominant_class_text}, {num_matched}/{num_generated} LLMs)'
                    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                             annotation_text, ha='left', va='center', fontsize=7)
                plt.tight_layout(rect=[0, 0, 0.9, 1])
                plt.savefig(ARTICLE_DIR / "CC_Plot_Placement_MatchRate_per_Method.png")
                print(f"Saved Placement MatchRate per Method plot to {ARTICLE_DIR / 'CC_Plot_Placement_MatchRate_per_Method.png'}")
                plt.close()
            else: print("No data for Plot 1 after filtering.")

            valid_match_rates_for_hist = plot_df_match_rate['MatchRate_Percent']
            if not valid_match_rates_for_hist.empty:
                print("\nGenerating Plot 2: Distribution of MatchRate Percentages...")
                plt.figure(figsize=(10, 6))
                sns.histplot(valid_match_rates_for_hist, bins=np.arange(0, 101, 10), kde=True, color="teal")
                plt.xlabel('Dominant Class Placement MatchRate (%)', fontsize=12)
                plt.ylabel('Number of Core Methods', fontsize=12)
                plt.title(f'Distribution of Placement Consistency for {len(valid_match_rates_for_hist)} Core Methods (N={actual_n_core})', fontsize=14)
                plt.xticks(np.arange(0, 101, 10))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                if pd.notna(overall_avg_match_rate):
                     plt.axvline(overall_avg_match_rate, color='red', linestyle='dashed', linewidth=2, 
                                label=f'Avg MatchRate: {overall_avg_match_rate:.1f}%')
                     plt.legend()
                plt.tight_layout()
                plt.savefig(ARTICLE_DIR / "CC_Plot_Distribution_Placement_MatchRate.png")
                print(f"Saved Distribution of Placement MatchRate plot to {ARTICLE_DIR / 'CC_Plot_Distribution_Placement_MatchRate.png'}")
                plt.close()
            else: print("No valid MatchRates for histogram.")
    else:
        print("Skipping plot generation as placement_consistency_df is empty or missing 'MatchRate_Percent'.")

    print("\nCore Method Placement Consistency analysis finished.")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main()