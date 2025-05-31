#!/usr/bin/env python3
"""
RQ1_Annotation_Completeness_AC.py

Assesses Annotation Completeness (AC) by classifying methods based on
the presence of UC and Action tags.
Generates:
1.  A summary table of annotation category counts and percentages per model.
2.  A stacked bar chart showing percentage distribution of annotation categories (Full, UC-only, None).
3.  A grouped bar chart comparing counts of Full vs. None annotations.
4.  Chi-squared test results for comparing distributions across models.

Reads from: reports/Annotation_and_Mapping_Combined.csv
Saves outputs to: reports/article/stats_ac/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
import pingouin as pg # For Cramer's V if needed, or manual calculation

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "Annotation_and_Mapping_Combined.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
STATS_AC_OUTPUT_DIR = ARTICLE_DIR / "stats_ac" # Dedicated folder for AC stats
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_AC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'Model'
UC_REF_COL = 'UC_References' # Column indicating presence of UC tags
UC_ACTION_COL = 'UC_Action'   # Column indicating presence of Action tags

P_VALUE_THRESHOLD = 0.05

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads the input CSV data."""
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False, keep_default_na=False, na_values=['']) # Treat empty strings as NA for bool check
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        required_cols = [MODEL_COL, UC_REF_COL, UC_ACTION_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: DataFrame from {filepath} is missing required columns: {missing_cols}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def categorize_annotation(row) -> str:
    """Categorizes a method's annotation completeness."""
    # Ensure that NA/empty strings correctly evaluate to False for presence
    has_uc = pd.notna(row[UC_REF_COL]) and str(row[UC_REF_COL]).strip() != ""
    has_action = pd.notna(row[UC_ACTION_COL]) and str(row[UC_ACTION_COL]).strip() != ""

    if has_uc and has_action:
        return "Full"
    elif has_uc and not has_action:
        return "UC-only"
    elif not has_uc and has_action:
        return "Action-only" # As per methodology, this should be rare/non-existent
    else: # not has_uc and not has_action
        return "None"

def main_ac_analysis():
    print("--- Starting Annotation Completeness (AC) Analysis ---")
    results_summary_ac_lines = ["--- Annotation Completeness (AC) Analysis ---"]

    input_filepath = REPORTS_DIR / INPUT_CSV_NAME
    df_methods = load_data(input_filepath)

    if df_methods is None or df_methods.empty:
        print("Failed to load or empty method data. Aborting AC analysis.")
        results_summary_ac_lines.append("Failed to load or empty method data. Aborting AC analysis.")
        with open(STATS_AC_OUTPUT_DIR / "AC_Statistical_Summary.txt", 'w', encoding='utf-8') as f: f.write("\n".join(results_summary_ac_lines))
        return
    
    df_methods = df_methods[~df_methods[MODEL_COL].astype(str).str.startswith('Grand')].copy()
    if df_methods.empty:
        print("No model data found after filtering. Aborting AC analysis.")
        return

    print("Categorizing annotations for each method...")
    df_methods['AnnotationCategory'] = df_methods.apply(categorize_annotation, axis=1)
    
    initial_categories_order = ["Full", "UC-only", "Action-only", "None"]
    df_methods['AnnotationCategory'] = pd.Categorical(df_methods['AnnotationCategory'], categories=initial_categories_order, ordered=True)

    annotation_counts_per_model = pd.crosstab(df_methods[MODEL_COL], df_methods['AnnotationCategory'])
    
    plot_categories_order = ["Full", "UC-only", "None"] 

    if 'Action-only' in annotation_counts_per_model.columns:
        if annotation_counts_per_model['Action-only'].sum() == 0:
            annotation_counts_per_model = annotation_counts_per_model.drop(columns=['Action-only'])
            print("Dropped 'Action-only' category as it has zero counts across all models.")
            results_summary_ac_lines.append("Dropped 'Action-only' category due to zero counts.")
        else: 
            action_only_counts_series = annotation_counts_per_model['Action-only']
            warn_msg = f"Warning: 'Action-only' category found with counts:\n{action_only_counts_series[action_only_counts_series > 0].to_string()}"
            print(warn_msg); results_summary_ac_lines.append(warn_msg)
            plot_categories_order = ["Full", "UC-only", "Action-only", "None"] 
    else: 
        print("Confirmed: No methods found in 'Action-only' category.")
        results_summary_ac_lines.append("Confirmed: No methods found in 'Action-only' category.")

    # --- CORRECTED PERCENTAGE CALCULATION ---
    total_methods_per_model_series = annotation_counts_per_model.sum(axis=1)
    annotation_percentages_per_model = pd.DataFrame(0.0, 
                                                    index=annotation_counts_per_model.index, 
                                                    columns=annotation_counts_per_model.columns)
    for model_idx in annotation_counts_per_model.index:
        total_for_model = total_methods_per_model_series.loc[model_idx]
        if total_for_model > 0:
            annotation_percentages_per_model.loc[model_idx] = \
                (annotation_counts_per_model.loc[model_idx] / total_for_model) * 100
    # --- END OF CORRECTION ---
    
    # Ensure all columns in plot_categories_order exist in annotation_percentages_per_model
    # (especially if a category was dropped, or if a model had 0 methods in all categories initially)
    for cat in plot_categories_order:
        if cat not in annotation_percentages_per_model.columns:
            annotation_percentages_per_model[cat] = 0.0


    results_summary_ac_lines.append("\nAnnotation Counts per Model:\n" + annotation_counts_per_model.to_string())
    results_summary_ac_lines.append("\nAnnotation Percentages per Model:\n" + annotation_percentages_per_model.round(2).to_string())
    print("Annotation Counts per Model:\n", annotation_counts_per_model)
    print("Annotation Percentages per Model:\n", annotation_percentages_per_model.round(2))

    ac_counts_path = STATS_AC_OUTPUT_DIR / "AC_Annotation_Counts.csv"
    annotation_counts_per_model.to_csv(ac_counts_path)
    print(f"Saved annotation counts to {ac_counts_path}")
    ac_percentages_path = STATS_AC_OUTPUT_DIR / "AC_Annotation_Percentages.csv"
    annotation_percentages_per_model.to_csv(ac_percentages_path, float_format='%.2f')
    print(f"Saved annotation percentages to {ac_percentages_path}")

    if not annotation_percentages_per_model.empty:
        print("\nGenerating Stacked Bar Chart for Annotation Completeness...")
        existing_plot_cols = [col for col in plot_categories_order if col in annotation_percentages_per_model.columns]
        
        # Ensure the DataFrame for plotting has the columns in the desired order
        plot_df_stacked = annotation_percentages_per_model[existing_plot_cols].copy()

        if not plot_df_stacked.empty:
            plot_df_stacked.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
            plt.title('Annotation Completeness by Model (Percentage)', fontsize=14)
            plt.xlabel('LLM Model', fontsize=12)
            plt.ylabel('Percentage of Methods (%)', fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(np.arange(0, 101, 10))
            plt.legend(title='Annotation Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0,0,0.85,1]) 
            stacked_bar_path = ARTICLE_DIR / "AC_StackedBar_AnnotationCompleteness.png"
            plt.savefig(stacked_bar_path); print(f"Saved stacked bar chart to {stacked_bar_path}"); plt.close()
        else:
            print("Warning: No data for stacked bar chart after filtering columns.")
            results_summary_ac_lines.append("Warning: No data for stacked bar chart of annotation percentages after filtering columns.")
    else:
        print("Warning: Annotation percentages DataFrame is empty. Skipping stacked bar chart.")
        results_summary_ac_lines.append("Warning: Annotation percentages DataFrame is empty.")

    if not annotation_counts_per_model.empty and 'Full' in annotation_counts_per_model.columns and 'None' in annotation_counts_per_model.columns:
        print("\nGenerating Grouped Bar Chart for Full vs. None Annotations...")
        df_full_none_counts = annotation_counts_per_model[['Full', 'None']].copy()
        df_full_none_counts_melted = df_full_none_counts.reset_index().melt(id_vars=MODEL_COL, var_name='Category', value_name='Count')
        
        if not df_full_none_counts_melted.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=MODEL_COL, y='Count', hue='Category', data=df_full_none_counts_melted, palette={'Full': 'green', 'None': 'red'})
            plt.title('Annotation Counts: Full vs. None per Model', fontsize=14)
            plt.xlabel('LLM Model', fontsize=12)
            plt.ylabel('Number of Methods', fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.legend(title='Annotation Category')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            grouped_bar_path = ARTICLE_DIR / "AC_GroupedBar_Full_vs_None.png"
            plt.savefig(grouped_bar_path); print(f"Saved grouped bar chart to {grouped_bar_path}"); plt.close()
        else:
            print("Warning: Data for Full vs. None grouped bar chart is empty after melt.")
            results_summary_ac_lines.append("Warning: Data for Full vs. None grouped bar chart is empty after melt.")
    else:
        print("Warning: No data for Full vs. None grouped bar chart (missing 'Full' or 'None' categories in counts).")
        results_summary_ac_lines.append("Warning: No data for Full vs. None grouped bar chart.")

    results_summary_ac_lines.append("\n--- Statistical Tests for Annotation Completeness ---")
    print("\n--- Statistical Tests for Annotation Completeness ---")
    
    # Use the same `plot_categories_order` for chi2 table consistency
    # but ensure it only uses columns present in annotation_counts_per_model
    chi2_categories = [cat for cat in plot_categories_order if cat in annotation_counts_per_model.columns]
    if not chi2_categories: # If after all filtering, no categories left
        print("No categories left for Chi-squared test after filtering.")
        results_summary_ac_lines.append("No categories left for Chi-squared test after filtering.")
    else:
        contingency_table_for_chi2 = annotation_counts_per_model[chi2_categories].copy()
        contingency_table_for_chi2_cleaned = contingency_table_for_chi2.loc[
            (contingency_table_for_chi2.sum(axis=1) > 0),
            (contingency_table_for_chi2.sum(axis=0) > 0) 
        ]

        if contingency_table_for_chi2_cleaned.shape[0] < 2 or contingency_table_for_chi2_cleaned.shape[1] < 2:
            msg = "Chi-squared test for annotation categories not performed (not enough data rows/cols after cleaning)."
            print(msg); results_summary_ac_lines.append(msg)
        else:
            print("\nPerforming Chi-squared test for annotation category distributions...")
            try:
                chi2_ac, p_ac, dof_ac, expected_ac = stats.chi2_contingency(contingency_table_for_chi2_cleaned)
                results_summary_ac_lines.append(f"Chi-squared Test (Annotation Categories): Chi2={chi2_ac:.3f}, p={p_ac:.4f}, df={dof_ac}")
                print(f"Chi-squared Test (Annotation Categories): Chi2={chi2_ac:.3f}, p={p_ac:.4f}, df={dof_ac}")

                if p_ac < P_VALUE_THRESHOLD:
                    n_ac = contingency_table_for_chi2_cleaned.sum().sum()
                    if n_ac > 0:
                        phi2_ac = chi2_ac / n_ac
                        r_ac, k_ac = contingency_table_for_chi2_cleaned.shape
                        min_dim_ac = min(k_ac - 1, r_ac - 1)
                        if min_dim_ac > 0:
                            cramers_v_ac = np.sqrt(phi2_ac / min_dim_ac)
                            results_summary_ac_lines.append(f"  Cramér's V (Annotation Categories): {cramers_v_ac:.3f}")
                            print(f"  Cramér's V (Annotation Categories): {cramers_v_ac:.3f}")
                        else: results_summary_ac_lines.append("  Cramér's V not well-defined (table dimensions too small).")
            except ValueError as e_chi2_ac:
                msg = f"Chi-squared test for annotation categories failed: {e_chi2_ac}"
                print(msg); results_summary_ac_lines.append(msg)

    summary_file_path = STATS_AC_OUTPUT_DIR / "AC_Statistical_Summary.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_summary_ac_lines))
    print(f"\nSaved AC statistical summary to {summary_file_path}")
    print("\n--- Annotation Completeness (AC) Analysis Finished ---")

# Ensure `if __name__ == "__main__":` block calls main_ac_analysis()
if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main_ac_analysis()

