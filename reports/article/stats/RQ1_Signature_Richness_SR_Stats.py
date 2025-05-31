#!/usr/bin/env python3
"""
RQ1_Signature_Richness_SR_Stats.py

Analyzes multiple dimensions of Signature Richness (SR):
- SR1: Visibility Markers (Table Data, Stats, Bar Chart Plot)
- SR3: Parameter Richness (Table Data, Stats, Bar Chart & Violin Plots)
- SR4: Return Types (Table Data, Stats, Bar Chart Plot)
(SR2 Naming Conventions and SR5 Lexical Coherence to be added later)

Reads from: reports/Annotation_and_Mapping_Combined.csv
Saves outputs to: reports/article/stats/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Statistical libraries
from scipy import stats
import scikit_posthocs as sp
import pingouin as pg

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "Annotation_and_Mapping_Combined.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
STATS_OUTPUT_DIR = ARTICLE_DIR / "stats"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = 'Model'
RUN_COL = 'Run'
SIGNATURE_COL = 'Signature'
METHOD_NAME_COL = 'MethodName'
P_VALUE_THRESHOLD = 0.05

print(f"DEBUG: Script started. ARTICLE_DIR: {ARTICLE_DIR}, STATS_OUTPUT_DIR: {STATS_OUTPUT_DIR}")

def load_data(filepath: Path) -> pd.DataFrame | None:
    print(f"DEBUG: Attempting to load data from {filepath}")
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        required_cols = [MODEL_COL, RUN_COL, SIGNATURE_COL, METHOD_NAME_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: DataFrame from {filepath} is missing required columns: {missing_cols}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def parse_method_signature(signature_str: str) -> dict:
    parsed = {
        'visibility': 'none', 
        'method_name_core': None, 
        'parameters': [],
        'parameter_count': 0, 
        'return_type': 'void'
    }
    if not isinstance(signature_str, str) or pd.isna(signature_str):
        return parsed
    signature_str_copy = str(signature_str)
    vis_match = re.match(r"^\s*([+\-#~])\s*", signature_str_copy)
    if vis_match:
        parsed['visibility'] = vis_match.group(1)
        signature_str_copy = signature_str_copy[vis_match.end():].strip()
    params_match = re.search(r"\((.*?)\)", signature_str_copy)
    name_and_maybe_return_str = signature_str_copy
    after_params_str = ""
    if params_match:
        params_content_str = params_match.group(1).strip()
        name_and_maybe_return_str = signature_str_copy[:params_match.start()].strip()
        after_params_str = signature_str_copy[params_match.end():].strip()
        if params_content_str:
            raw_params = params_content_str.split(',')
            for p_str_full in raw_params:
                p_str = p_str_full.strip()
                if not p_str: continue
                param_name, param_type = None, p_str
                parts = p_str.split(':')
                if len(parts) > 1:
                    potential_name = parts[-2].strip()
                    if len(potential_name.split()) == 1 and not any(c in potential_name for c in '<>,'):
                        param_name = potential_name
                        param_type = parts[-1].strip()
                    else: param_type = p_str
                elif len(p_str.split()) > 1 and not any(c in p_str for c in '<>,'):
                    type_parts = p_str.split()
                    param_type = " ".join(type_parts[:-1])
                    param_name = type_parts[-1]
                else: param_type = p_str
                parsed['parameters'].append({'name': param_name, 'type': param_type})
            parsed['parameter_count'] = len(parsed['parameters'])
    if after_params_str and after_params_str.startswith(':'):
        parsed['return_type'] = after_params_str[1:].strip() or 'void'
    name_parts = name_and_maybe_return_str.split()
    if name_parts:
        parsed['method_name_core'] = name_parts[-1]
        if len(name_parts) > 1 and parsed['return_type'] == 'void':
            potential_return_type = " ".join(name_parts[:-1])
            if potential_return_type and not any(mod in potential_return_type.lower().split() for mod in ['static', 'final', 'abstract', 'public', 'private', 'protected']):
                 parsed['return_type'] = potential_return_type
    if not parsed['method_name_core'] and signature_str_copy:
        parsed['method_name_core'] = signature_str_copy.split('(')[0].strip().split()[-1] if '(' in signature_str_copy else signature_str_copy.strip().split()[-1]
    if not parsed['return_type'] or parsed['return_type'].lower() == 'void':
        parsed['return_type'] = 'void'
    else:
        parsed['return_type'] = re.sub(r"^(final|static|const)\s+", "", parsed['return_type']).strip()
    vis_map = {'+': 'public', '-': 'private', '#': 'protected', '~': 'package'}
    parsed['visibility_category'] = vis_map.get(parsed['visibility'], 'none_or_default')
    return parsed

def extract_features_for_sr(df: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG: Entering extract_features_for_sr")
    if SIGNATURE_COL not in df.columns:
        print(f"Error: '{SIGNATURE_COL}' column not found.")
        return pd.DataFrame()
    if df.empty:
        print("DEBUG: DataFrame is empty at start of extract_features_for_sr.")
        return pd.DataFrame()
    print(f"DEBUG: Parsing {len(df)} signatures...")
    parsed_signatures_series = df[SIGNATURE_COL].apply(parse_method_signature)
    df_out = df.copy()
    df_out['param_count'] = parsed_signatures_series.apply(lambda x: x['parameter_count'])
    df_out['visibility_raw'] = parsed_signatures_series.apply(lambda x: x['visibility']) # Store raw symbol
    df_out['visibility_category'] = parsed_signatures_series.apply(lambda x: x['visibility_category'])
    df_out['return_type_str'] = parsed_signatures_series.apply(lambda x: x['return_type'])
    df_out['is_non_void_return'] = df_out['return_type_str'].apply(
        lambda x: isinstance(x, str) and x.lower().strip() != 'void' and x.strip() != ''
    )
    print("DEBUG: Finished parsing. Columns created: param_count, visibility_raw, visibility_category, return_type_str, is_non_void_return.")
    return df_out

def analyze_parameter_richness(df_features: pd.DataFrame):
    # ... (This function remains unchanged from the previous version) ...
    print("DEBUG: Entering analyze_parameter_richness")
    if df_features.empty or 'param_count' not in df_features.columns:
        print("Warning: Feature DataFrame is empty or missing 'param_count'. Skipping Parameter Richness analysis.")
        return None, None 

    print("\n--- SR3: Parameter Richness Analysis ---")
    results_summary_sr3 = ["--- SR3: Parameter Richness ---"]

    print("DEBUG: Calculating IQR of parameter counts per model...")
    iqr_per_model = df_features.groupby(MODEL_COL)['param_count'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    iqr_df = iqr_per_model.reset_index(name='Param_IQR').dropna(subset=['Param_IQR']) 
    
    if iqr_df.empty: print("Warning: IQR DataFrame is empty. Check param_count data.")
    else:
        print("IQR of Parameter Counts per Model:\n", iqr_df)
        results_summary_sr3.append("\nIQR of Parameter Counts per Model:\n" + iqr_df.to_string())

    print("\nDEBUG: Calculating mean parameter richness for bar chart...")
    mean_params_per_run = df_features.groupby([MODEL_COL, RUN_COL])['param_count'].mean().reset_index()
    if mean_params_per_run.empty:
        print("Warning: mean_params_per_run is empty. Cannot proceed with bar chart stats.")
        if not iqr_df.empty: return iqr_df, "\n".join(results_summary_sr3) + "\nWarning: Mean params per run empty."
        return None, "\n".join(results_summary_sr3)  + "\nWarning: Mean params per run empty."

    model_mean_params_agg = mean_params_per_run.groupby(MODEL_COL)['param_count'].agg(
        Mean_Params='mean', Std_Params='std', N_Runs='count').reset_index()
    
    if model_mean_params_agg.empty: print("Warning: model_mean_params_agg is empty.")
    else:
        model_mean_params_agg['SEM_Params'] = model_mean_params_agg['Std_Params'] / np.sqrt(model_mean_params_agg['N_Runs'])
        model_mean_params_agg['CI_Lower'] = model_mean_params_agg['Mean_Params'] - 1.96 * model_mean_params_agg['SEM_Params']
        model_mean_params_agg['CI_Upper'] = model_mean_params_agg['Mean_Params'] + 1.96 * model_mean_params_agg['SEM_Params']
        model_mean_params_agg.sort_values(by='Mean_Params', ascending=False, inplace=True)
        print("Mean Parameter Richness (per-run avg, then model avg with CI):\n", model_mean_params_agg)
        results_summary_sr3.append("\nMean Parameter Richness (for bar chart):\n" + model_mean_params_agg.to_string())

        plt.figure(figsize=(10, 6))
        plt.bar(model_mean_params_agg[MODEL_COL], model_mean_params_agg['Mean_Params'], 
                       yerr=model_mean_params_agg['SEM_Params'] * 1.96, capsize=5, color=sns.color_palette("coolwarm", len(model_mean_params_agg)))
        plt.xlabel("LLM Model", fontsize=12); plt.ylabel("Mean Number of Parameters per Method", fontsize=12)
        plt.title("Mean Parameter Richness per LLM (95% CI from SEM of Per-Run Averages)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        param_bar_path = ARTICLE_DIR / "SR_BarChart_MeanParamRichness.png"
        plt.savefig(param_bar_path); print(f"Saved Parameter Richness bar chart to {param_bar_path}"); plt.close()

    print("\nDEBUG: Generating Violin Plot for parameter count distributions...")
    if not df_features.empty and 'param_count' in df_features.columns and df_features['param_count'].notna().any():
        plt.figure(figsize=(12, 7))
        order = df_features.groupby(MODEL_COL)['param_count'].median().sort_values().index
        sns.violinplot(x=MODEL_COL, y='param_count', data=df_features, palette="viridis", order=order, cut=0, inner="quartile")
        plt.xlabel("LLM Model", fontsize=12); plt.ylabel("Number of Parameters per Method", fontsize=12)
        plt.title("Parameter Count Distribution per Method and Model", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        param_violin_path = ARTICLE_DIR / "SR_Violin_ParamCountDistribution.png"
        plt.savefig(param_violin_path); print(f"Saved Parameter Count violin plot to {param_violin_path}"); plt.close()
    else: print("Warning: Not enough data for violin plot.")

    results_summary_sr3.append("\n--- Statistical Tests for Parameter Counts ---")
    print("\n--- Statistical Tests for Parameter Counts ---")
    param_counts_by_model_list = [group['param_count'].dropna().values for _, group in df_features.groupby(MODEL_COL)]
    param_counts_by_model_list = [arr for arr in param_counts_by_model_list if len(arr) > 0]

    if len(param_counts_by_model_list) >= 2:
        try:
            levene_stat_params, levene_p_params = stats.levene(*param_counts_by_model_list)
            results_summary_sr3.append(f"Levene's Test (Parameter Counts): Statistic={levene_stat_params:.3f}, p={levene_p_params:.4f}")
            print(f"Levene's Test (Parameter Counts): Statistic={levene_stat_params:.3f}, p={levene_p_params:.4f}")
        except ValueError as e:
            results_summary_sr3.append(f"Levene's Test (Parameter Counts): Could not be computed. Error: {e}")
            print(f"Levene's Test (Parameter Counts): Could not be computed. Error: {e}")
            levene_p_params = 0 

        print("\nPerforming Kruskal-Wallis Test on parameter counts...")
        if all(len(group) > 0 for group in param_counts_by_model_list):
            kw_results_params = pg.kruskal(data=df_features, dv='param_count', between=MODEL_COL)
            if kw_results_params is not None and not kw_results_params.empty:
                print(kw_results_params.round(4))
                results_summary_sr3.append("Kruskal-Wallis Results (Parameter Counts):\n" + kw_results_params.round(4).to_string())
                h_stat_params = kw_results_params['H'].iloc[0]; p_value_kw_params = kw_results_params['p-unc'].iloc[0]
                if 'eps-sq' in kw_results_params.columns and not pd.isna(kw_results_params['eps-sq'].iloc[0]):
                    epsilon_sq_params = kw_results_params['eps-sq'].iloc[0]
                    results_summary_sr3.append(f"Effect Size (Epsilon-squared, ε²): {epsilon_sq_params:.3f}")
                    print(f"Effect Size (Epsilon-squared, ε²): {epsilon_sq_params:.3f}")
                if p_value_kw_params < P_VALUE_THRESHOLD:
                    results_summary_sr3.append("Kruskal-Wallis significant for parameter counts. Dunn's post-hoc:")
                    print("Kruskal-Wallis significant for parameter counts. Dunn's post-hoc:")
                    dunn_results_params = sp.posthoc_dunn(df_features, val_col='param_count', group_col=MODEL_COL, p_adjust='bonferroni')
                    print(dunn_results_params.round(4))
                    results_summary_sr3.append("Dunn's Test Results:\n" + dunn_results_params.round(4).to_string())
                else:
                    results_summary_sr3.append("Kruskal-Wallis not significant for parameter counts.")
                    print("Kruskal-Wallis not significant for parameter counts.")
            else:
                results_summary_sr3.append("Kruskal-Wallis test could not be computed (empty results).")
                print("Kruskal-Wallis test could not be computed (empty results).")
        else:
            results_summary_sr3.append("Not all model groups have data for Kruskal-Wallis parameter count comparison.")
            print("Not all model groups have data for Kruskal-Wallis parameter count comparison.")
    else:
        results_summary_sr3.append("Not enough model groups for Kruskal-Wallis test on parameter counts.")
        print("Not enough model groups for Kruskal-Wallis test on parameter counts.")
        
    return iqr_df, "\n".join(results_summary_sr3)
    # End of analyze_parameter_richness

def analyze_visibility_markers(df_features: pd.DataFrame):
    """Analyzes visibility marker distributions and generates a bar chart."""
    print("\n--- SR1: Visibility Markers Analysis ---")
    results_summary_sr1 = ["--- SR1: Visibility Markers ---"]

    if df_features.empty or 'visibility_raw' not in df_features.columns: # Using visibility_raw for symbol
        print("Warning: Feature DataFrame empty or missing 'visibility_raw'. Skipping Visibility analysis.")
        return None, None, None # Return table data, stats text, plot path

    # Use the raw visibility symbols for counts as per table example (+, --, #, None)
    # If 'visibility_raw' is None (from parser), map to 'None' string for grouping
    df_features['vis_symbol_for_table'] = df_features['visibility_raw'].fillna('None')
    
    visibility_counts = df_features.groupby(MODEL_COL)['vis_symbol_for_table'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    
    # Ensure all expected symbols are columns for the table and plot
    # Your example table uses +, --, #, None
    expected_symbols_for_table = ['+', '-', '#', 'None'] # '~' (package) could be grouped into 'None' or be separate
    
    plot_data_vis = pd.DataFrame(index=visibility_counts.index)
    for symbol in expected_symbols_for_table:
        if symbol in visibility_counts.columns:
            plot_data_vis[symbol] = visibility_counts[symbol]
        else:
            plot_data_vis[symbol] = 0.0 # Add column with zeros if not present
            
    # Prepare table_data for merging later (this is proportions)
    visibility_table_data_for_merge = plot_data_vis.copy()
    visibility_table_data_for_merge.columns = [f"Vis_{col}" for col in visibility_table_data_for_merge.columns]

    print("Visibility Proportions (%):\n", plot_data_vis.round(2))
    results_summary_sr1.append("\nVisibility Proportions (%):\n" + plot_data_vis.round(2).to_string())

    # --- Generate SR1 Bar Chart ---
    df_melted_vis = plot_data_vis.reset_index().melt(id_vars=MODEL_COL, var_name='Marker', value_name='Percentage')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted_vis, x=MODEL_COL, y='Percentage', hue='Marker', palette="muted")
    plt.title('Visibility Marker Distribution by Model', fontsize=14)
    plt.ylabel('Percentage of Methods', fontsize=12)
    plt.xlabel('LLM Model', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 105) # Percentage
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Visibility Marker')
    plt.tight_layout()
    
    vis_barchart_path = ARTICLE_DIR / "SR1_BarChart_VisibilityMarkers.png"
    try:
        plt.savefig(vis_barchart_path, dpi=300)
        print(f"Saved Visibility Marker bar chart to {vis_barchart_path}")
    except Exception as e:
        print(f"Error saving visibility bar chart: {e}")
    plt.close()
    # --- End of SR1 Bar Chart ---


    # Statistical Test: Chi-squared for visibility marker distributions
    # For stats, use the broader categories (public, private, protected, other)
    # from 'visibility_category'
    if 'visibility_category' in df_features.columns:
        contingency_table_vis_cat = pd.crosstab(df_features[MODEL_COL], df_features['visibility_category'])
        contingency_table_vis_cat_cleaned = contingency_table_vis_cat.loc[contingency_table_vis_cat.sum(axis=1) > 0]

        if len(contingency_table_vis_cat_cleaned) >= 2 and len(contingency_table_vis_cat_cleaned.columns) >=2 :
            print("\nPerforming Chi-squared test for visibility category distributions...")
            chi2_vis, p_vis, dof_vis, expected_vis = stats.chi2_contingency(contingency_table_vis_cat_cleaned)
            results_summary_sr1.append(f"\nChi-squared Test (Visibility Categories): Chi2={chi2_vis:.3f}, p={p_vis:.4f}, df={dof_vis}")
            print(f"Chi-squared Test (Visibility Categories): Chi2={chi2_vis:.3f}, p={p_vis:.4f}, df={dof_vis}")

            if p_vis < P_VALUE_THRESHOLD:
                n_vis = contingency_table_vis_cat_cleaned.sum().sum()
                phi2_vis = chi2_vis / n_vis
                k_vis, r_vis = contingency_table_vis_cat_cleaned.shape
                cramers_v_vis = np.sqrt(phi2_vis / min(k_vis - 1, r_vis - 1))
                results_summary_sr1.append(f"Cramér's V (Visibility Categories): {cramers_v_vis:.3f}")
                print(f"Cramér's V (Visibility Categories): {cramers_v_vis:.3f}")
        else:
            results_summary_sr1.append("\nChi-squared test for visibility categories not performed (insufficient data).")
            print("Chi-squared test for visibility categories not performed (insufficient data).")
    else:
        results_summary_sr1.append("\n'visibility_category' column not found for Chi-squared test.")
        print("'visibility_category' column not found for Chi-squared test.")


    return visibility_table_data_for_merge.reset_index(), "\n".join(results_summary_sr1), vis_barchart_path


def analyze_return_types(df_features: pd.DataFrame):
    """Analyzes return type distributions and generates a bar chart."""
    print("\n--- SR4: Return Types Analysis ---")
    results_summary_sr4 = ["--- SR4: Return Types ---"]

    if df_features.empty or 'is_non_void_return' not in df_features.columns:
        print("Warning: Feature DataFrame empty or missing 'is_non_void_return'. Skipping Return Type analysis.")
        return None, None, None

    non_void_proportions = df_features.groupby(MODEL_COL)['is_non_void_return'].mean().mul(100)
    return_type_table_data = non_void_proportions.reset_index(name='Ret_Percent_NonVoid')
    
    # Sort for consistent plotting order (e.g., by proportion)
    return_type_table_data_sorted = return_type_table_data.sort_values(by='Ret_Percent_NonVoid', ascending=False)


    print("Percentage of Non-Void Return Types:\n", return_type_table_data_sorted.round(2))
    results_summary_sr4.append("\nPercentage of Non-Void Return Types:\n" + return_type_table_data_sorted.round(2).to_string())

    # --- Generate SR4 Bar Chart ---
    plt.figure(figsize=(10, 6)) # Adjusted figsize slightly for better label fit potentially
    
    # Use a seaborn palette
    # You can choose 'viridis', 'plasma', 'coolwarm', 'mako', 'rocket', etc.
    # Or categorical ones like 'Set2', 'Paired'
    palette_name = "viridis" 
    
    # If you want each bar to have a *different* color from the palette:
    barplot = sns.barplot(data=return_type_table_data_sorted, 
                          x=MODEL_COL, 
                          y='Ret_Percent_NonVoid', 
                          palette=palette_name) 
    
    # If you wanted all bars to be the *same* color (like 'skyblue' before), but from a palette:
    # single_color = sns.color_palette(palette_name, 1)[0] # Get the first color from the palette
    # barplot = sns.barplot(data=return_type_table_data_sorted, 
    #                       x=MODEL_COL, 
    #                       y='Ret_Percent_NonVoid', 
    #                       color=single_color)

    plt.title('Non-void Return Type Proportion by Model', fontsize=14)
    plt.ylabel('% of Methods with Non-Void Return Types', fontsize=12)
    plt.xlabel('LLM Model', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for p_bar in barplot.patches:
        barplot.annotate(format(p_bar.get_height(), '.1f') + '%', 
                       (p_bar.get_x() + p_bar.get_width() / 2., p_bar.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=8)

    plt.tight_layout()
    
    ret_barchart_path = ARTICLE_DIR / "SR4_BarChart_ReturnTypes.png"
    try:
        plt.savefig(ret_barchart_path, dpi=300)
        print(f"Saved Return Type bar chart to {ret_barchart_path}")
    except Exception as e:
        print(f"Error saving return type bar chart: {e}")
    plt.close()
    # --- End of SR4 Bar Chart ---

    # ... (rest of the statistical test code remains the same) ...
    # Statistical Test: Chi-squared for non-void vs. void distributions
    return_type_counts = df_features.groupby(MODEL_COL)['is_non_void_return'].agg(
        NonVoid_Count=lambda x: x.sum(),
        Void_Count=lambda x: (~x).sum()
    )
    return_type_counts_cleaned = return_type_counts.loc[return_type_counts.sum(axis=1) > 0]

    if len(return_type_counts_cleaned) >= 2 and len(return_type_counts_cleaned.columns) >=2 :
        print("\nPerforming Chi-squared test for return type distributions...")
        try:
            chi2_ret, p_ret, dof_ret, expected_ret = stats.chi2_contingency(return_type_counts_cleaned)
            results_summary_sr4.append(f"\nChi-squared Test (Return Types): Chi2={chi2_ret:.3f}, p={p_ret:.4f}, df={dof_ret}")
            print(f"Chi-squared Test (Return Types): Chi2={chi2_ret:.3f}, p={p_ret:.4f}, df={dof_ret}")

            if p_ret < P_VALUE_THRESHOLD:
                n_ret = return_type_counts_cleaned.sum().sum()
                phi2_ret = chi2_ret / n_ret
                k_ret, r_ret = return_type_counts_cleaned.shape
                cramers_v_ret = np.sqrt(phi2_ret / min(k_ret - 1, r_ret - 1))
                results_summary_sr4.append(f"Cramér's V (Return Types): {cramers_v_ret:.3f}")
                print(f"Cramér's V (Return Types): {cramers_v_ret:.3f}")
        except ValueError as e: 
            results_summary_sr4.append(f"\nChi-squared Test (Return Types) could not be computed: {e}")
            print(f"Chi-squared Test (Return Types) could not be computed: {e}")
    else:
        results_summary_sr4.append("\nChi-squared test for return types not performed (insufficient data).")
        print("Chi-squared test for return types not performed (insufficient data).")

    return return_type_table_data, "\n".join(results_summary_sr4), ret_barchart_path

def main():
    print("--- Starting Signature Richness (SR) Analysis ---")
    print(f"DEBUG: Main function started. Outputs will go to ARTICLE_DIR: {ARTICLE_DIR}, STATS_OUTPUT_DIR: {STATS_OUTPUT_DIR}")
    
    all_methods_data_df = load_data(REPORTS_DIR / INPUT_CSV_NAME)
    if all_methods_data_df is None: return print("Failed to load method data. Exiting SR.")
    print(f"DEBUG: Loaded all_methods_data_df, shape: {all_methods_data_df.shape}")

    df_features = extract_features_for_sr(all_methods_data_df)
    if df_features.empty: return print("Feature extraction failed. Aborting SR.")
    print(f"DEBUG: Extracted features, df_features shape: {df_features.shape}")
    if 'param_count' not in df_features.columns: # Basic check
        print("DEBUG: 'param_count' column is missing after feature extraction. Critical for SR3.")
        # Depending on what's critical, you might want to return here or let other analyses proceed
    if 'visibility_category' not in df_features.columns: # Basic check
        print("DEBUG: 'visibility_category' column is missing after feature extraction. Critical for SR1.")
    if 'is_non_void_return' not in df_features.columns: # Basic check
        print("DEBUG: 'is_non_void_return' column is missing after feature extraction. Critical for SR4.")


    # --- SR3: Parameter Richness ---
    param_iqr_table_data, param_stats_summary_text = analyze_parameter_richness(df_features)
    if param_stats_summary_text:
        with open(STATS_OUTPUT_DIR / "SR3_ParameterRichness_Stats.txt", 'w', encoding='utf-8') as f:
            f.write(param_stats_summary_text)
        print(f"\nSaved SR3 Parameter Richness stats to {STATS_OUTPUT_DIR / 'SR3_ParameterRichness_Stats.txt'}")

    # --- SR1: Visibility Markers ---
    visibility_table_data, visibility_stats_summary_text, _ = analyze_visibility_markers(df_features) # Plot path ignored here
    if visibility_stats_summary_text:
        with open(STATS_OUTPUT_DIR / "SR1_VisibilityMarkers_Stats.txt", 'w', encoding='utf-8') as f:
            f.write(visibility_stats_summary_text)
        print(f"\nSaved SR1 Visibility Markers stats to {STATS_OUTPUT_DIR / 'SR1_VisibilityMarkers_Stats.txt'}")
    
    # --- SR4: Return Types ---
    return_type_table_data, return_type_stats_summary_text, _ = analyze_return_types(df_features) # Plot path ignored here
    if return_type_stats_summary_text:
        with open(STATS_OUTPUT_DIR / "SR4_ReturnTypes_Stats.txt", 'w', encoding='utf-8') as f:
            f.write(return_type_stats_summary_text)
        print(f"\nSaved SR4 Return Types stats to {STATS_OUTPUT_DIR / 'SR4_ReturnTypes_Stats.txt'}")
    
    # --- Combine data for SR Summary Table (CSV Output Only) ---
    print("\n--- Assembling SR Summary Table (CSV Output) ---")
    sr_summary_list = []
    
    # Start with Model column from a reliable source (e.g., param_iqr_table_data if it exists and is not empty)
    # Or, get unique models from df_features if other tables might be empty
    if param_iqr_table_data is not None and not param_iqr_table_data.empty:
        base_df_for_merge = param_iqr_table_data.set_index(MODEL_COL)
        sr_summary_list.append(base_df_for_merge)
    elif visibility_table_data is not None and not visibility_table_data.empty:
        base_df_for_merge = visibility_table_data.set_index(MODEL_COL)
        sr_summary_list.append(base_df_for_merge)
    elif return_type_table_data is not None and not return_type_table_data.empty:
        base_df_for_merge = return_type_table_data.set_index(MODEL_COL)
        sr_summary_list.append(base_df_for_merge)
    elif not df_features.empty: # Fallback to models from df_features
        all_models_df = pd.DataFrame({MODEL_COL: df_features[MODEL_COL].unique()})
        base_df_for_merge = all_models_df.set_index(MODEL_COL)
        # No data to append yet, but base_df_for_merge can be the start for pd.concat if list empty
    else:
        print("No data available to start SR Summary Table. Skipping CSV generation.")
        sr_master_table = pd.DataFrame() # Ensure it's defined

    # Add other data if they weren't the base
    if not (param_iqr_table_data is not None and not param_iqr_table_data.empty and base_df_for_merge is param_iqr_table_data.set_index(MODEL_COL)):
        if param_iqr_table_data is not None and not param_iqr_table_data.empty:
            sr_summary_list.append(param_iqr_table_data.set_index(MODEL_COL))
            
    if not (visibility_table_data is not None and not visibility_table_data.empty and base_df_for_merge is visibility_table_data.set_index(MODEL_COL)):
        if visibility_table_data is not None and not visibility_table_data.empty:
            # For CSV, include all Vis columns explicitly
            sr_summary_list.append(visibility_table_data.set_index(MODEL_COL))

    if not (return_type_table_data is not None and not return_type_table_data.empty and base_df_for_merge is return_type_table_data.set_index(MODEL_COL)):
        if return_type_table_data is not None and not return_type_table_data.empty:
            ret_for_table = return_type_table_data.set_index(MODEL_COL)[['Ret_Percent_NonVoid']]
            ret_for_table.rename(columns={'Ret_Percent_NonVoid': 'Ret.'}, inplace=True)
            sr_summary_list.append(ret_for_table)
            
    if sr_summary_list:
        sr_master_table = pd.concat(sr_summary_list, axis=1).reset_index()
        
        # Define desired order for CSV, including all relevant visibility columns
        desired_cols_order = [MODEL_COL, 'Param_IQR', 'Ret.']
        if visibility_table_data is not None: # Add individual visibility columns for CSV
             # Assuming visibility_table_data returns columns like 'Vis_+', 'Vis_--', 'Vis_#', 'Vis_None'
            vis_cols_from_func = [col for col in visibility_table_data.columns if col.startswith('Vis_') and col != MODEL_COL]
            desired_cols_order.extend(vis_cols_from_func)
            
        # Add 'LexDiv' here when ready: desired_cols_order.append('LexDiv')
            
        final_cols_for_table_csv = [col for col in desired_cols_order if col in sr_master_table.columns]
        sr_master_table_csv = sr_master_table[final_cols_for_table_csv].copy()
        
        # Rename for CSV to match table example where possible
        rename_csv_map = {'Param_IQR':'IQR'} 
        if 'Vis_+' in sr_master_table_csv.columns : rename_csv_map['Vis_+'] = '+'
        if 'Vis_--' in sr_master_table_csv.columns : rename_csv_map['Vis_--'] = '--'
        if 'Vis_#' in sr_master_table_csv.columns : rename_csv_map['Vis_#'] = '#'
        if 'Vis_None' in sr_master_table_csv.columns : rename_csv_map['Vis_None'] = 'None_Vis' # Avoid clash with pandas None

        sr_master_table_csv.rename(columns=rename_csv_map, inplace=True)
        
        sr_table_path = ARTICLE_DIR / "SR_Metrics_Combined_Summary_Table.csv"
        sr_master_table_csv.to_csv(sr_table_path, index=False, float_format='%.2f')
        print(f"Saved Combined SR Metrics Summary Table to {sr_table_path}")
    else:
        print("No data to assemble for SR Summary Table CSV.")

    # <<<< START: REMOVED LATEX TABLE GENERATION BLOCK >>>>
    # print("\n--- Generating LaTeX for SR Table (if data available) ---")
    # if 'sr_master_table' in locals() and not sr_master_table.empty:
    #     df_for_latex = sr_master_table_csv.copy() # Start from the CSV version
    #     # Formatting for LaTeX
    #     for col in df_for_latex.columns:
    #         if col != MODEL_COL:
    #             try: 
    #                 df_for_latex[col] = pd.to_numeric(df_for_latex[col], errors='coerce').apply(
    #                     lambda x: f"{x:.2f}" if pd.notnull(x) else "n/a"
    #                 )
    #             except (ValueError, TypeError): pass 
        
    #     # Construct the '-- / # / None' composite column for LaTeX if individual parts exist
    #     if all(col in df_for_latex.columns for col in ['--', '#', 'None_Vis']):
    #         df_for_latex['-- / # / None'] = df_for_latex.apply(
    #             lambda row: f"{row['--']} / {row['#']} / {row['None_Vis']}", axis=1
    #         )
    #         cols_to_drop_for_latex = ['--', '#', 'None_Vis']
    #         df_for_latex = df_for_latex.drop(columns=cols_to_drop_for_latex, errors='ignore')
    #     else:
    #         print("DEBUG: Could not create composite '-- / # / None' column for LaTeX as individual columns are missing.")
    #         if '-- / # / None' not in df_for_latex.columns: # Add placeholder if not created
    #             df_for_latex['-- / # / None'] = "n/a / n/a / n/a"


    #     final_cols_for_latex_display = [MODEL_COL, 'IQR', 'Ret.', '+', '-- / # / None'] # Add 'LexDiv' here
    #     final_cols_for_latex_display = [col for col in final_cols_for_latex_display if col in df_for_latex.columns]
    #     df_for_latex_display = df_for_latex[final_cols_for_latex_display]
        
    #     latex_headers_display = [col.replace('-- / # / None', '-- / \\# / None') for col in df_for_latex_display.columns]

    #     latex_str = df_for_latex_display.to_latex(index=False, escape=False, na_rep='n/a',
    #                                      header=latex_headers_display,
    #                                      column_format='l' + 'c'*(len(df_for_latex_display.columns)-1) # Basic 'c' format for now
    #                                      ) 
    #     print("\n--- Suggested LaTeX for SR Table ---")
    #     print(latex_str)
    #     with open(STATS_OUTPUT_DIR / "SR_Table_Snippet.tex", "w", encoding="utf-8") as f:
    #         f.write("% Adjust column formats (e.g., S[table-format=1.2] for siunitx) as needed.\n")
    #         f.write(latex_str)
    #     print(f"Saved LaTeX table snippet to {STATS_OUTPUT_DIR / 'SR_Table_Snippet.tex'}")
    # else:
    #     print("SR Master Table for LaTeX is empty or not created, skipping LaTeX generation.")
    # <<<< END: REMOVED LATEX TABLE GENERATION BLOCK >>>>

    print("\n--- Signature Richness (SR) Analysis (Parts 1,3,4) Finished ---")

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError: print("Warning: Seaborn style not found, using Matplotlib default.")
    main()