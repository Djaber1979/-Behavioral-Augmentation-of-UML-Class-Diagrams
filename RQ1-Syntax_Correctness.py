#!/usr/bin/env python3
"""
RQ1_Syntactic_Correctness_SC.py

Analyzes Syntactic Correctness (SC) of generated PlantUML diagrams.
- Generates a grouped bar chart showing successfully rendered vs. errored diagrams per LLM.
- Performs a Chi-squared test to compare error rates across models.

Reads from: reports/article/SyntaxCorrectness_Summary.csv (or similar path)
Saves outputs to: reports/article/stats_sc/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats

# Configuration
REPORTS_DIR = Path("reports")
ARTICLE_DIR = REPORTS_DIR / "article"  # Define ARTICLE_DIR globally
STATS_SC_OUTPUT_DIR = ARTICLE_DIR / "stats_sc" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_SC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV_NAME = ARTICLE_DIR / "SyntaxCorrectness_Summary.csv" 

MODEL_COL = 'LLM' 
CORRECT_COL = 'Correct_Diagrams'
ERRORED_COL = 'Errored_Diagrams'
TOTAL_COL = 'Total_Diagrams'

P_VALUE_THRESHOLD = 0.05

def load_sc_summary_data(filepath: Path) -> pd.DataFrame | None:
    """Loads the syntax correctness summary CSV."""
    if not filepath.is_file():
        print(f"Error: Input SC summary CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        required_cols = [MODEL_COL, CORRECT_COL, ERRORED_COL, TOTAL_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: DataFrame from {filepath} is missing required columns: {missing_cols}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Filter out "Grand Total" row for model-specific analysis
        # CORRECTED LINE:
        df = df[df[MODEL_COL].astype(str).str.lower() != 'grand total'].copy() 
        
        if df.empty:
            print("Error: No model data found after filtering 'Grand Total'.")
            return None
        return df
    except Exception as e:
        print(f"Error loading SC summary CSV from {filepath}: {e}")
        return None
    
def generate_sc_grouped_bar_chart(df_summary: pd.DataFrame, output_path: Path):
    """Generates Figure sc-bar: Grouped bar chart of Correct vs. Errored diagrams."""
    if df_summary.empty or not all(c in df_summary.columns for c in [MODEL_COL, CORRECT_COL, ERRORED_COL]):
        print("Warning: Data for SC grouped bar chart is empty or missing columns. Skipping plot.")
        return

    df_melted = df_summary.melt(id_vars=MODEL_COL, 
                                value_vars=[CORRECT_COL, ERRORED_COL],
                                var_name='Outcome', value_name='Count')

    plt.figure(figsize=(12, 7))
    sns.barplot(x=MODEL_COL, y='Count', hue='Outcome', data=df_melted, 
                palette={CORRECT_COL: 'skyblue', ERRORED_COL: 'salmon'})
    
    plt.title('Syntactic Correctness: Successfully Rendered vs. Syntax Errors per LLM', fontsize=14)
    plt.xlabel('LLM Model', fontsize=12)
    plt.ylabel('Number of Diagrams', fontsize=12)
    # Ensure y-ticks cover the max total (usually 10)
    max_y_val = df_summary[TOTAL_COL].max() if TOTAL_COL in df_summary and not df_summary[TOTAL_COL].empty else 10
    plt.yticks(np.arange(0, max_y_val + 2, 1)) 
    plt.legend(title='Diagram Outcome')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"Saved Syntactic Correctness grouped bar chart to {output_path}")
    except Exception as e:
        print(f"Error saving SC grouped bar chart: {e}")
    plt.close()


def perform_sc_statistical_analysis(df_summary: pd.DataFrame, results_summary_list: list):
    """Performs Chi-squared test for syntactic correctness."""
    print("\n--- Statistical Analysis for Syntactic Correctness (SC) ---")
    results_summary_list.append("\n--- Statistical Analysis for Syntactic Correctness (SC) ---")

    if df_summary.empty or not all(c in df_summary.columns for c in [MODEL_COL, CORRECT_COL, ERRORED_COL]):
        msg = "Warning: Data for SC statistical analysis is empty or missing columns."
        print(msg); results_summary_list.append(msg)
        return

    contingency_table_sc = df_summary.set_index(MODEL_COL)[[CORRECT_COL, ERRORED_COL]]
    contingency_table_sc_cleaned = contingency_table_sc.loc[
        (contingency_table_sc.sum(axis=1) > 0),
        (contingency_table_sc.sum(axis=0) > 0)
    ]

    if contingency_table_sc_cleaned.shape[0] < 2 or contingency_table_sc_cleaned.shape[1] < 2:
        msg = "Chi-squared test for syntactic correctness not performed (not enough data rows/cols after cleaning)."
        print(msg); results_summary_list.append(msg)
    else:
        results_summary_list.append("\nContingency Table for Chi-squared (Syntactic Correctness):\n" + contingency_table_sc_cleaned.to_string())
        print("\nPerforming Chi-squared test for syntactic correctness distributions...")
        try:
            chi2_sc, p_sc, dof_sc, expected_sc = stats.chi2_contingency(contingency_table_sc_cleaned)
            results_summary_list.append(f"Chi-squared Test (Overall Correctness): Chi2={chi2_sc:.3f}, p={p_sc:.4f}, df={dof_sc}")
            print(f"Chi-squared Test (Overall Correctness): Chi2={chi2_sc:.3f}, p={p_sc:.4f}, df={dof_sc}")

            if p_sc < P_VALUE_THRESHOLD:
                n_sc = contingency_table_sc_cleaned.sum().sum()
                if n_sc > 0 :
                    phi2_sc = chi2_sc / n_sc
                    r_sc, k_sc = contingency_table_sc_cleaned.shape
                    min_dim_sc = min(k_sc - 1, r_sc - 1)
                    if min_dim_sc > 0:
                        cramers_v_sc = np.sqrt(phi2_sc / min_dim_sc)
                        results_summary_list.append(f"  Cramér's V (Overall Correctness): {cramers_v_sc:.3f}")
                        print(f"  Cramér's V (Overall Correctness): {cramers_v_sc:.3f}")
                    else:
                        results_summary_list.append("  Cramér's V not well-defined (table dimensions too small).")
                        print("  Cramér's V not well-defined (table dimensions too small).")
        except ValueError as e_chi2_sc:
            msg = f"Chi-squared test for syntactic correctness failed: {e_chi2_sc}"
            print(msg); results_summary_list.append(msg)

def main():
    print("--- Starting Syntactic Correctness (SC) Analysis ---")
    statistical_results_all_sc = []

    df_sc_summary = load_sc_summary_data(INPUT_CSV_NAME)

    if df_sc_summary is None or df_sc_summary.empty:
        print("Failed to load or empty SC summary data. Aborting SC analysis.")
        statistical_results_all_sc.append("Failed to load or empty SC summary data.")
        summary_file_path = STATS_SC_OUTPUT_DIR / "SC_Statistical_Summary.txt"
        with open(summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(statistical_results_all_sc))
        print(f"Saved partial SC statistical summary to {summary_file_path}")
        return

    # --- Generate Grouped Bar Chart (Figure sc-bar) ---
    plot_output_path = ARTICLE_DIR / "SyntaxCorrectness_GroupedBar.png" # Use defined ARTICLE_DIR
    generate_sc_grouped_bar_chart(df_sc_summary, plot_output_path)

    # --- Perform Statistical Analysis ---
    perform_sc_statistical_analysis(df_sc_summary, statistical_results_all_sc)
    
    # --- Save All Statistical Summaries for SC ---
    summary_file_path = STATS_SC_OUTPUT_DIR / "SC_Statistical_Summary.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        for line in statistical_results_all_sc:
            f.write(line + "\n")
    print(f"\nSaved overall SC statistical summary to {summary_file_path}")
        
    print("\n--- Syntactic Correctness (SC) Analysis Finished ---")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main()