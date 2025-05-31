#!/usr/bin/env python3
"""
RQ1_Structural_Fidelity_SF.py

Generates:
1.  A CSV table summarizing per-element preservation rates (%) and a global mean per LLM.
2.  A radar chart for models with partial preservation.
3.  A bar chart for global structural fidelity scores.
4.  Statistical analysis (Repeated Measures ANOVA) for SF scores.

Methodology for Preservation_E:
Preservation_E = (|E_base INTERSECTION E_enriched|) / (|E_base UNION E_enriched|) * 100%
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from pathlib import Path
# Statistical libraries
from scipy import stats
import pingouin as pg 

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "Combined_Struct_Counts_Metrics.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
STATS_SF_OUTPUT_DIR = ARTICLE_DIR / "stats_sf" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_SF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'Model'
RUN_COL = 'Run'     

ELEMENT_TYPES_PREFIX = { 
    "packages": "Pack", "enums": "En", "enum_values": "EV",
    "classes": "Cla", "attributes": "Att", "relationships": "Rel"
}
PERFECT_SCORE_THRESHOLD = 99.999 
P_VALUE_THRESHOLD = 0.05

LLM_COLORS = {
    'Claude3.7': '#E69F00',                     # Orange/Gold
    'Gemini-2.5-Pro-Preview-05-06': '#D55E00',  # Vermillion/Burnt Orange
    'ChatGPT-o3': '#009E73',                    # Bluish Green
    'Qwen3': '#56B4E9',                         # Sky Blue
    'DeepSeek(R1)': '#CC79A7',                  # Reddish Purple/Mauve
    'Grok3': '#0072B2',                         # Darker Blue
    'ChatGPT-4o': '#8C8C00',                    # Dark Yellow/Olive Green
    'Llama4': '#E76F51',                        # Reddish Coral
    'Mistral': '#999999'                        # Grey
}

# Desired order for plots
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


def load_data(filepath: Path) -> pd.DataFrame | None:
    if not filepath.is_file(): print(f"Error: Input CSV not found: {filepath}"); return None
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        required_cols = ['Model', 'Run'] + [f"{p}_preserved" for p in ELEMENT_TYPES_PREFIX.keys()]
        if not all(col in df.columns for col in required_cols):
            print(f"Error: DataFrame missing some required columns. Check {required_cols}")
            return None
        return df
    except Exception as e: print(f"Error loading CSV from {filepath}: {e}"); return None

def calculate_preservation_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating preservation scores for each run...")
    df_scores = df[['Model', 'Run']].copy()
    for prefix, short_name in ELEMENT_TYPES_PREFIX.items():
        preserved_col = f"{prefix}_preserved"
        baseline_col = f"{prefix}_total_baseline"
        added_col = f"{prefix}_added"
        if not all(col in df.columns for col in [preserved_col, baseline_col, added_col]):
            df_scores[f"Preservation_{short_name}"] = np.nan; continue
        numerator = df[preserved_col]
        denominator = df[baseline_col] + df[added_col]
        df_scores[f"Preservation_{short_name}"] = np.where(
            denominator == 0, np.where(numerator == 0, 100.0, np.nan), 
            (numerator / denominator) * 100.0
        )
    return df_scores

def generate_summary_table_csv(df_run_scores: pd.DataFrame, results_summary_list: list) -> pd.DataFrame | None:
    """
    Calculates mean preservation scores per model, a global score, and saves to CSV.
    Returns the full summary DataFrame for use in other plots.
    """
    if df_run_scores.empty or 'Model' not in df_run_scores.columns:
        msg = "Warning: Run scores DataFrame empty or missing 'Model' column. Cannot generate summary table."
        print(msg); results_summary_list.append(msg)
        return None 
    
    preservation_cols_for_agg = [col for col in df_run_scores.columns if col.startswith("Preservation_")]
    if not preservation_cols_for_agg:
        msg = "Warning: No 'Preservation_*' columns in run scores. Cannot generate summary."
        print(msg); results_summary_list.append(msg)
        return None

    summary_df_models = df_run_scores.groupby('Model')[preservation_cols_for_agg].mean()
    
    element_short_names_map = {f"Preservation_{short_name}": short_name for short_name in ELEMENT_TYPES_PREFIX.values()}
    summary_df_models = summary_df_models.rename(columns=element_short_names_map)
    
    element_short_names_list = list(ELEMENT_TYPES_PREFIX.values()) 
    for short_name in element_short_names_list:
        if short_name not in summary_df_models.columns:
            summary_df_models[short_name] = np.nan

    summary_df_models['Global'] = summary_df_models[element_short_names_list].mean(axis=1)
    
    full_summary_for_csv = summary_df_models.copy()
    
    if not full_summary_for_csv.empty:
        gt_element_means = full_summary_for_csv[element_short_names_list].mean()
        gt_global = gt_element_means.mean()
        gt_global_series = pd.Series([gt_global], index=['Global'])
        grand_total_series_with_global = pd.concat([gt_element_means, gt_global_series])
        grand_total_df = grand_total_series_with_global.to_frame(name='Grand Total').T
        grand_total_df.index.name = 'Model'
        full_summary_for_csv = pd.concat([full_summary_for_csv, grand_total_df])
    
    full_summary_for_csv = full_summary_for_csv.reset_index()
    
    # Ensure correct column order for CSV
    final_csv_cols = ['Model'] + element_short_names_list + ['Global']
    final_csv_cols = [col for col in final_csv_cols if col in full_summary_for_csv.columns] # Keep only existing
    full_summary_for_csv = full_summary_for_csv[final_csv_cols]


    csv_path = ARTICLE_DIR / "SF_Preservation_Summary_Table_FULL.csv"
    try:
        if not full_summary_for_csv.empty:
            full_summary_for_csv.to_csv(csv_path, index=False, float_format='%.2f')
            print(f"Saved FULL SF summary table (for CSV) to {csv_path}")
            results_summary_list.append(f"\nFull SF Summary Table (saved to {csv_path}):\n{full_summary_for_csv.round(2).to_string()}")
        else:
            print(f"Full SF summary table is empty. Not saving CSV.")
            results_summary_list.append("Full SF summary table is empty. Not saving CSV.")
    except Exception as e:
        print(f"Error saving FULL SF summary table: {e}")
        
    return full_summary_for_csv

# Ensure these are defined globally in RQ1_Structural_Fidelity_SF.py
# or passed as arguments
# LLM_COLORS = { ... your palette ... }
# LLM_ORDER_FOR_PLOTS = [ ... your model order ... ]
# ELEMENT_TYPES_PREFIX = { "packages": "Pack", ... } (already there)
# PERFECT_SCORE_THRESHOLD (already there)

def generate_radar_chart(df_summary: pd.DataFrame, output_path: Path):
    """
    Generates a radar chart of preservation scores for each LLM.
    Excludes 'Grand Total' and models with 100% on all individual element types.
    Uses custom color palette and model order.
    """
    if df_summary.empty:
        print("Warning: Summary DataFrame empty. Skipping radar chart.")
        return

    # 1. Prepare data: Exclude Grand Total
    df_for_filtering = df_summary[df_summary['Model'] != 'Grand Total'].copy()
    if df_for_filtering.empty:
        print("Warning: No model data (excluding Grand Total) for radar chart filtering.")
        return

    # 2. Identify categories (element types for the axes - 'Pack', 'En', etc.)
    # These are the short names, already columns in df_summary after generate_summary_table_csv
    categories = [short_name for short_name in ELEMENT_TYPES_PREFIX.values() 
                  if short_name in df_for_filtering.columns]
    
    if not categories:
        print("Warning: No element categories (e.g., 'Pack', 'En') found in df_summary columns for radar chart axes.")
        return
    if len(categories) < 3: # Radar plots need at least 3 axes to be meaningful
        print(f"Warning: Need at least 3 categories for radar chart, found {len(categories)}. Skipping.")
        return
        
    # 3. Filter out models with perfect scores on all categories
    perfect_models = []
    models_to_plot_indices = []
    for idx, row in df_for_filtering.iterrows():
        is_all_perfect = True
        for cat_col in categories:
            if cat_col not in row or pd.isna(row[cat_col]) or row[cat_col] < PERFECT_SCORE_THRESHOLD:
                is_all_perfect = False
                break
        if is_all_perfect:
            perfect_models.append(row['Model'])
        else:
            models_to_plot_indices.append(idx) # Store index to use with .loc
            
    if perfect_models:
        print(f"Info (Radar): Excluding models with all element scores >= {PERFECT_SCORE_THRESHOLD}%: {', '.join(perfect_models)}")

    df_plot_filtered = df_for_filtering.loc[models_to_plot_indices]

    if df_plot_filtered.empty:
        print("Warning: No models left to plot on radar chart after filtering perfect scores.")
        return

    # 4. Apply custom order to the filtered models
    models_present_after_filter = df_plot_filtered['Model'].unique()
    plot_order_radar = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_after_filter]
    # Add any remaining models not in LLM_ORDER_FOR_PLOTS (maintains all for plotting if some missed in order list)
    for model in models_present_after_filter:
        if model not in plot_order_radar:
            plot_order_radar.append(model)
    
    # Re-index df_plot to this order for consistent plotting
    df_plot = df_plot_filtered.set_index('Model').reindex(plot_order_radar).reset_index()
    df_plot.dropna(subset=['Model'], inplace=True) # Drop rows if a model in order was not in data

    if df_plot.empty:
        print("Warning: No models left for radar chart after ordering and filtering.")
        return

    # 5. Setup plot
    n_categories_plot = len(categories) # Recalculate based on actual categories found
    angles = [n / float(n_categories_plot) * 2 * pi for n in range(n_categories_plot)]
    angles += angles[:1] # Close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True)) # Adjusted size slightly
    
    num_plot_models = len(df_plot['Model'].unique())
    if num_plot_models == 0: plt.close(); return
    
    # Create palette for plotted models
    plot_palette_radar = [LLM_COLORS.get(model, '#333333') for model in df_plot['Model']]


    # 6. Plot data for each model
    for i, (idx_row, row) in enumerate(df_plot.iterrows()): # Iterate through the ordered df_plot
        model_name = row['Model']
        # Ensure values are numeric and handle potential NaNs for plotting
        values_raw = row[categories].values.astype(float).flatten().tolist()
        # Ensure values align with categories, handle potential missing category columns gracefully
        current_values = []
        for cat_name in categories:
            current_values.append(row.get(cat_name, 0) if pd.notna(row.get(cat_name, 0)) else 0)

        values_to_plot = current_values + current_values[:1] # Close the plot
        
        ax.plot(angles, values_to_plot, linewidth=2, linestyle='solid', label=model_name, color=plot_palette_radar[i])
        ax.fill(angles, values_to_plot, color=plot_palette_radar[i], alpha=0.25)

    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{val}%" for val in np.arange(0, 101, 20)], color="grey", size=9)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 105) 

    plt.title('Per-element Preservation Rates (Models with Scores <100%)', size=14, y=1.12, pad=20)
    # Adjust legend: place below plot, ensure it uses the model names from df_plot
    ax.legend(df_plot['Model'].tolist(), loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=min(3, num_plot_models), fontsize='small')
    
    # fig.tight_layout() # tight_layout can sometimes clip polar plot legends
    
    try:
        plt.savefig(output_path, bbox_inches='tight') # bbox_inches='tight' helps with external legends
        print(f"Saved SF radar chart to {output_path}")
    except Exception as e:
        print(f"Error saving radar chart: {e}")
    plt.close()

def generate_global_score_bar_chart(df_summary: pd.DataFrame, output_path: Path):
    # ... (same as before) ...
    if df_summary.empty or 'Global' not in df_summary.columns:
        print("Warning: Summary DataFrame empty or missing 'Global'. Skipping global score bar chart.")
        return
    df_plot = df_summary[df_summary['Model'] != 'Grand Total'].copy()
    if df_plot.empty: print("Warning: No model data for global score bar chart."); return
    df_plot['Global'] = pd.to_numeric(df_plot['Global'], errors='coerce')
    df_plot = df_plot.dropna(subset=['Global']).sort_values(by='Global', ascending=False)
    if df_plot.empty: print("Warning: No valid Global scores to plot."); return

    plt.figure(figsize=(10, max(4, len(df_plot) * 0.45)))
    barplot = sns.barplot(x='Global', y='Model', data=df_plot, palette='summer', orient='h')
    plt.xlabel('Global Mean Preservation Score (%)', fontsize=12)
    plt.ylabel('LLM Model', fontsize=12)
    plt.title('Global Structural Fidelity (Mean Across Element Types)', fontsize=14)
    plt.xlim(0, 105); plt.grid(axis='x', linestyle='--', alpha=0.7)
    for i, p_bar in enumerate(barplot.patches): 
        value = p_bar.get_width()
        barplot.text(value + 0.5, p_bar.get_y() + p_bar.get_height() / 2,
                     f'{value:.1f}%', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    try: plt.savefig(output_path); print(f"Saved global score bar chart to {output_path}")
    except Exception as e: print(f"Error saving global score bar chart: {e}")
    plt.close()

def perform_sf_statistical_analysis(df_run_scores: pd.DataFrame, results_summary_list: list):
    """Performs Mixed ANOVA for Structural Fidelity scores."""
    print("\n--- Statistical Analysis for Structural Fidelity (SF) ---")
    results_summary_list.append("\n--- Statistical Analysis for Structural Fidelity (SF) ---")

    if df_run_scores.empty or df_run_scores[[col for col in df_run_scores.columns if col.startswith("Preservation_")]].isnull().all().all():
        msg = "SF Run scores DataFrame is empty or all preservation scores are NaN. Skipping Mixed ANOVA."
        print(msg); results_summary_list.append(msg)
        return

    # Prepare data for Mixed ANOVA
    id_vars = ['Model', 'Run']
    value_vars = [col for col in df_run_scores.columns if col.startswith("Preservation_")]
    if not value_vars:
        msg = "No 'Preservation_*' columns found for melting. Skipping Mixed ANOVA."
        print(msg); results_summary_list.append(msg)
        return
        
    df_long = df_run_scores.melt(id_vars=id_vars, value_vars=value_vars,
                                 var_name='ElementType', value_name='PreservationScore')
    df_long['ElementType'] = df_long['ElementType'].str.replace('Preservation_', '')
    df_long.dropna(subset=['PreservationScore'], inplace=True) 

    if df_long.empty or df_long['PreservationScore'].isnull().all():
        msg = "DataFrame for Mixed ANOVA is empty after melting or all scores are NaN. Skipping."
        print(msg); results_summary_list.append(msg)
        return
    
    # Pingouin's mixed_anova expects unique subject identifiers.
    # If 'Run' is just 1-10, it's repeated across models. We need a globally unique subject ID.
    # Let's create one: Model_Run
    df_long['Subject_ID'] = df_long['Model'].astype(str) + "_Run" + df_long['Run'].astype(str)


    if df_long['Model'].nunique() < 1 or df_long['ElementType'].nunique() < 2 : # Need at least 1 group for 'between' and 2 levels for 'within'
        msg = "Not enough groups/levels for Mixed ANOVA (need >=1 model, >=2 element types). Skipping."
        print(msg); results_summary_list.append(msg)
        return
    
    # Check if any group has insufficient data for stats
    group_counts = df_long.groupby(['Model', 'ElementType'])['PreservationScore'].count()
    if group_counts.min() < 2 : # Typically need at least 2 for variance calculations within cells
        # Also check if any model has data for only one element type, or any element type has data for only one model
        if df_long.groupby('Model')['ElementType'].nunique().min() < 2 or \
           df_long.groupby('ElementType')['Model'].nunique().min() < 1 : # (between can have 1 level, but then it's more like a one-way RM anova)
            print("Warning: Data is too sparse for a full mixed ANOVA (e.g., some models might not have scores for all element types, or only one model present).")
            results_summary_list.append("Warning: Mixed ANOVA might be unreliable due to sparse data structure (e.g. not all models have all element types).")
            # Depending on the exact sparsity, pingouin might still run or error out.

    results_summary_list.append("\nMixed ANOVA (Models vs. ElementType Scores):")
    print("\nPerforming Mixed ANOVA (dv='PreservationScore', within='ElementType', subject='Subject_ID', between='Model')...")
    
    try:
        mixed_anova_results = pg.mixed_anova(data=df_long, 
                                             dv='PreservationScore', 
                                             within='ElementType', 
                                             subject='Subject_ID', # Use the globally unique subject ID
                                             between='Model',
                                             effsize="ng2") # Generalized eta-squared
        
        print(mixed_anova_results.round(4))
        results_summary_list.append("Mixed ANOVA Results:\n" + mixed_anova_results.round(4).to_string())

        # Check for significant effects and consider post-hocs
        # Main effect of Model
        if 'Model' in mixed_anova_results['Source'].values:
            model_effect_row = mixed_anova_results[mixed_anova_results['Source'] == 'Model']
            if not model_effect_row.empty:
                model_effect_p = model_effect_row['p-unc'].iloc[0]
                if model_effect_p < P_VALUE_THRESHOLD:
                    results_summary_list.append(f"\nSignificant main effect of Model (p={model_effect_p:.4f}). Consider pairwise post-hocs between models.")
                    print(f"Significant main effect of Model (p={model_effect_p:.4f}).")
                    # Post-hoc for between-subject factor (Model)
                    # Often done using pairwise t-tests or pg.pairwise_gameshowell if variances unequal
                    # posthoc_model = pg.pairwise_tests(data=df_long, dv='PreservationScore', between='Model', subject='Subject_ID', padjust='bonf')
                    # print("\nPost-hoc tests for Model effect (Bonferroni corrected):\n", posthoc_model[posthoc_model['Contrast'] == 'Model'].round(4))
                    # results_summary_list.append("\nPost-hoc tests for Model effect (Bonferroni corrected):\n" + posthoc_model[posthoc_model['Contrast'] == 'Model'].round(4).to_string())


        # Main effect of ElementType
        if 'ElementType' in mixed_anova_results['Source'].values:
            element_effect_row = mixed_anova_results[mixed_anova_results['Source'] == 'ElementType']
            if not element_effect_row.empty:
                element_effect_p = element_effect_row['p-unc'].iloc[0]
                if element_effect_p < P_VALUE_THRESHOLD:
                    results_summary_list.append(f"\nSignificant main effect of ElementType (p={element_effect_p:.4f}). Consider pairwise post-hocs between element types.")
                    print(f"Significant main effect of ElementType (p={element_effect_p:.4f}).")
                    # Post-hoc for within-subject factor (ElementType)
                    # posthoc_element = pg.pairwise_tests(data=df_long, dv='PreservationScore', within='ElementType', subject='Subject_ID', padjust='bonf')
                    # print("\nPost-hoc tests for ElementType effect (Bonferroni corrected):\n", posthoc_element[posthoc_element['Contrast'] == 'ElementType'].round(4))
                    # results_summary_list.append("\nPost-hoc tests for ElementType effect (Bonferroni corrected):\n" + posthoc_element[posthoc_element['Contrast'] == 'ElementType'].round(4).to_string())

        # Interaction effect
        interaction_source_name = f"{MODEL_COL} * ElementType" # Based on pingouin's typical naming
        if interaction_source_name in mixed_anova_results['Source'].values:
            interaction_effect_row = mixed_anova_results[mixed_anova_results['Source'] == interaction_source_name]
            if not interaction_effect_row.empty:
                interaction_effect_p = interaction_effect_row['p-unc'].iloc[0]
                if interaction_effect_p < P_VALUE_THRESHOLD:
                     results_summary_list.append(f"\nSignificant interaction effect {interaction_source_name} (p={interaction_effect_p:.4f}). Further simple effects analysis needed.")
                     print(f"Significant interaction effect {interaction_source_name} (p={interaction_effect_p:.4f}). Further simple effects analysis needed.")

    except Exception as e:
        error_msg = f"Error during Mixed ANOVA: {e}"
        print(error_msg)
        results_summary_list.append(error_msg)
        print("Data passed to mixed_anova (first 5 rows of df_long):\n", df_long.head())
        print("Value counts for Subject_ID:\n", df_long['Subject_ID'].value_counts().head())
        print("Info for df_long:\n")
        df_long.info()
def main():
    print("--- Starting Structural Fidelity (SF) Analysis ---")
    statistical_results_all_sf = [] 

    df_combined_metrics = load_data(REPORTS_DIR / INPUT_CSV_NAME)
    if df_combined_metrics is None: return print("Failed to load data. Exiting SF.")

    df_run_scores = calculate_preservation_scores(df_combined_metrics)
    if df_run_scores.empty: return print("No per-run preservation scores. Exiting SF.")

    # --- Generate Summary Table CSV (Full data) ---
    # The LaTeX table generation is removed from generate_summary_table_csv
    sf_summary_table_full_df = generate_summary_table_csv(df_run_scores, statistical_results_all_sf)
    if sf_summary_table_full_df is None or sf_summary_table_full_df.empty:
        print("SF Summary Table generation failed or produced no data. Some plots might be skipped.")
    
    # --- Generate Plots using sf_summary_table_full_df ---
    if sf_summary_table_full_df is not None and not sf_summary_table_full_df.empty:
        print("\nGenerating SF Radar Chart (models with partial preservation)...")
        generate_radar_chart(sf_summary_table_full_df, ARTICLE_DIR / "SF_RadarChart_Preservation.png")

        print("\nGenerating SF Global Score Bar Chart...")
        generate_global_score_bar_chart(sf_summary_table_full_df, ARTICLE_DIR / "SF_BarChart_GlobalScore.png")
    else:
        print("Skipping Radar and Global Score Bar charts as summary table data is unavailable.")

    perform_sf_statistical_analysis(df_run_scores, statistical_results_all_sf)

    summary_file_path = STATS_SF_OUTPUT_DIR / "SF_Statistical_Summary_Overall.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(statistical_results_all_sf))
    print(f"\nSaved overall SF statistical summary to {summary_file_path}")
        
    print("\n--- Structural Fidelity (SF) Analysis Finished ---")

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError: print("Warning: Seaborn style not found, using Matplotlib default.")
    main()