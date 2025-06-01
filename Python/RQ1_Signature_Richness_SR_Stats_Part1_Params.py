#!/usr/bin/env python3
"""
RQ1_Signature_Richness_SR_Stats_Part1_Params.py
... (rest of the docstring) ...
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
        # print("DEBUG: df.head() after loading:\n", df.head())
        # print("DEBUG: df.info() after loading:")
        # df.info()
        
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
    # print(f"DEBUG_PARSE: Input signature: '{signature_str}'") # Uncomment for very verbose parsing debug
    parsed = {
        'visibility': None, 'method_name_core': None, 'parameters': [],
        'parameter_count': 0, 'return_type': 'void' # Default if not found or empty
    }
    if not isinstance(signature_str, str) or pd.isna(signature_str):
        # print(f"DEBUG_PARSE: Invalid input type or NaN: '{signature_str}'")
        return parsed

    signature_str_copy = str(signature_str).strip() 

    # 1. Visibility
    vis_match = re.match(r"^\s*([+\-#~])\s*", signature_str_copy)
    if vis_match:
        parsed['visibility'] = vis_match.group(1)
        signature_str_copy = signature_str_copy[vis_match.end():].strip()
        # print(f"DEBUG_PARSE: Found visibility '{parsed['visibility']}', remaining: '{signature_str_copy}'")

    # Isolate parameters string and parts before/after
    params_content_str = None
    name_and_return_part = signature_str_copy
    after_params_part = ""

    # Look for the main parentheses for parameters
    # Handle potential nested parentheses in types by finding the matching one for the method
    open_paren_index = signature_str_copy.find('(')
    if open_paren_index != -1:
        balance = 0
        closed_paren_index = -1
        for i in range(open_paren_index, len(signature_str_copy)):
            if signature_str_copy[i] == '(':
                balance += 1
            elif signature_str_copy[i] == ')':
                balance -= 1
                if balance == 0:
                    closed_paren_index = i
                    break
        
        if closed_paren_index != -1:
            params_content_str = signature_str_copy[open_paren_index+1:closed_paren_index].strip()
            name_and_return_part = signature_str_copy[:open_paren_index].strip()
            after_params_part = signature_str_copy[closed_paren_index+1:].strip()
            # print(f"DEBUG_PARSE: Params content: '{params_content_str}', Before: '{name_and_return_part}', After: '{after_params_part}'")

    # 2. Parameters
    if params_content_str: # Only if params_content_str is not None and not empty
        # Smart split for parameters, respecting generics like Map<String, Type>
        # This regex splits by comma, but not commas inside < >
        raw_params = re.split(r",(?![^<]*>)", params_content_str)
        # print(f"DEBUG_PARSE: Raw params after split: {raw_params}")
        
        for p_full_str in raw_params:
            p_str = p_full_str.strip()
            if not p_str: continue

            param_name, param_type = None, p_str 
            
            # Try to find last space before a potential name
            # e.g. "final String name", "List<String> items"
            # This assumes type might have spaces, name is usually last word before colon or end
            parts = p_str.split()
            if len(parts) > 1:
                # Check if there's a ':', e.g., "name : Type" or "name:Type"
                colon_split = p_str.rsplit(':', 1)
                if len(colon_split) == 2 and colon_split[0].strip() and not any(c in colon_split[0] for c in '<>,'):
                    # Likely "name : Type"
                    param_name = colon_split[0].strip()
                    param_type = colon_split[1].strip()
                elif not any(c in parts[-1] for c in '<>,:'): # If last word looks like a simple name
                    param_name = parts[-1]
                    param_type = " ".join(parts[:-1])
                else: # Assume all is type
                    param_type = p_str
            else: # Only one word, assume it's a type
                param_type = p_str
            
            parsed['parameters'].append({'name': param_name, 'type': param_type})
        parsed['parameter_count'] = len(parsed['parameters'])
        # print(f"DEBUG_PARSE: Parsed parameters: {parsed['parameters']}, Count: {parsed['parameter_count']}")


    # 3. Return Type
    if after_params_part and after_params_part.startswith(':'):
        parsed['return_type'] = after_params_part[1:].strip() or 'void'
        # print(f"DEBUG_PARSE: Return type from after params: '{parsed['return_type']}'")

    # 4. Method Name and potential prefixed Return Type
    name_parts = name_and_return_part.split()
    if name_parts:
        parsed['method_name_core'] = name_parts[-1]
        if len(name_parts) > 1 and (parsed['return_type'] == 'void' or not parsed['return_type']): # Only if not already found after params
            potential_return_type = " ".join(name_parts[:-1])
            if potential_return_type and not any(mod in potential_return_type.lower().split() for mod in ['static', 'final', 'abstract', 'public', 'private', 'protected', 'synchronized', 'native']):
                 parsed['return_type'] = potential_return_type
        # print(f"DEBUG_PARSE: Method name core: '{parsed['method_name_core']}', Potential prefixed return: '{parsed['return_type']}'")


    # Fallback for method name if still not found
    if not parsed['method_name_core'] and signature_str_copy: 
        main_part = signature_str_copy.split('(')[0].strip()
        if main_part:
            parsed['method_name_core'] = main_part.split()[-1]
            # print(f"DEBUG_PARSE: Fallback method name: '{parsed['method_name_core']}'")

    # Clean up return type
    if not parsed['return_type'] or parsed['return_type'].lower() == 'void' or not parsed['return_type'].strip():
        parsed['return_type'] = 'void'
    else:
        # Remove leading/trailing modifiers from return type
        parsed['return_type'] = re.sub(r"^(final|static|const|public|private|protected|synchronized|native)\s+", "", parsed['return_type']).strip()
        if not parsed['return_type']: parsed['return_type'] = 'void' # If it became empty

    # print(f"DEBUG_PARSE: Final parsed: {parsed}")
    return parsed

def extract_features_for_sr(df: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG: Entering extract_features_for_sr")
    if SIGNATURE_COL not in df.columns:
        print(f"Error: '{SIGNATURE_COL}' column not found in DataFrame for feature extraction.")
        return pd.DataFrame()
    if df.empty:
        print("DEBUG: DataFrame is empty at start of extract_features_for_sr.")
        return pd.DataFrame()

    print(f"DEBUG: Parsing {len(df)} signatures...")
    # Test parser on a few examples
    # sample_signatures_test = df[SIGNATURE_COL].dropna().sample(min(5, len(df[SIGNATURE_COL].dropna()))).tolist()
    # print("DEBUG: Sample signatures for parser testing:")
    # for sig_test in sample_signatures_test:
    #     print(f"  Original: '{sig_test}' -> Parsed: {parse_method_signature(sig_test)}")

    parsed_signatures_series = df[SIGNATURE_COL].apply(parse_method_signature)
    
    df_out = df.copy()
    df_out['param_count'] = parsed_signatures_series.apply(lambda x: x['parameter_count'])
    
    print(f"DEBUG: Finished parsing. Param_count column created. Nulls in param_count: {df_out['param_count'].isnull().sum()}")
    # print("DEBUG: df_out.head() in extract_features_for_sr:\n", df_out[[SIGNATURE_COL, 'param_count']].head())
    # print("DEBUG: Param_count describe:\n", df_out['param_count'].describe())
    return df_out

def analyze_parameter_richness(df_features: pd.DataFrame):
    print("DEBUG: Entering analyze_parameter_richness")
    if df_features.empty or 'param_count' not in df_features.columns:
        print("Warning: Feature DataFrame is empty or missing 'param_count'. Skipping Parameter Richness analysis.")
        return None, None 

    print("\n--- SR3: Parameter Richness Analysis ---")
    results_summary_sr3 = ["--- SR3: Parameter Richness ---"]

    print("DEBUG: Calculating IQR of parameter counts per model...")
    iqr_per_model = df_features.groupby(MODEL_COL)['param_count'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25) if len(x.dropna()) > 0 else np.nan)
    iqr_df = iqr_per_model.reset_index(name='Param_IQR').dropna(subset=['Param_IQR']) # Drop models if IQR is NaN
    
    if iqr_df.empty:
        print("Warning: IQR DataFrame is empty. Check param_count data.")
    else:
        print("IQR of Parameter Counts per Model:\n", iqr_df)
        results_summary_sr3.append("\nIQR of Parameter Counts per Model:\n" + iqr_df.to_string())

    print("\nDEBUG: Calculating mean parameter richness for bar chart...")
    mean_params_per_run = df_features.groupby([MODEL_COL, RUN_COL])['param_count'].mean().reset_index()
    if mean_params_per_run.empty:
        print("Warning: mean_params_per_run is empty. Cannot proceed with bar chart stats.")
        # Try to generate plots even if some stats are empty
        if not iqr_df.empty: # if at least IQR was computed
             return iqr_df, "\n".join(results_summary_sr3) + "\nWarning: Mean params per run empty."
        return None, "\n".join(results_summary_sr3)  + "\nWarning: Mean params per run empty."


    model_mean_params_agg = mean_params_per_run.groupby(MODEL_COL)['param_count'].agg(
        Mean_Params='mean', Std_Params='std', N_Runs='count'
    ).reset_index()
    
    if model_mean_params_agg.empty:
        print("Warning: model_mean_params_agg is empty.")
    else:
        model_mean_params_agg['SEM_Params'] = model_mean_params_agg['Std_Params'] / np.sqrt(model_mean_params_agg['N_Runs'])
        model_mean_params_agg['CI_Lower'] = model_mean_params_agg['Mean_Params'] - 1.96 * model_mean_params_agg['SEM_Params']
        model_mean_params_agg['CI_Upper'] = model_mean_params_agg['Mean_Params'] + 1.96 * model_mean_params_agg['SEM_Params']
        model_mean_params_agg.sort_values(by='Mean_Params', ascending=False, inplace=True)
        print("Mean Parameter Richness (per-run avg, then model avg with CI):\n", model_mean_params_agg)
        results_summary_sr3.append("\nMean Parameter Richness (for bar chart):\n" + model_mean_params_agg.to_string())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_mean_params_agg[MODEL_COL], model_mean_params_agg['Mean_Params'], 
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
    else:
        print("Warning: Not enough data for violin plot.")

    results_summary_sr3.append("\n--- Statistical Tests for Parameter Counts ---")
    print("\n--- Statistical Tests for Parameter Counts ---")
    param_counts_by_model_list = [group['param_count'].dropna().values for _, group in df_features.groupby(MODEL_COL)]
    # Filter out any empty arrays that might result if a model had no methods or all param_counts were NaN
    param_counts_by_model_list = [arr for arr in param_counts_by_model_list if len(arr) > 0]

    if len(param_counts_by_model_list) >= 2:
        try:
            levene_stat_params, levene_p_params = stats.levene(*param_counts_by_model_list)
            results_summary_sr3.append(f"Levene's Test (Parameter Counts): Statistic={levene_stat_params:.3f}, p={levene_p_params:.4f}")
            print(f"Levene's Test (Parameter Counts): Statistic={levene_stat_params:.3f}, p={levene_p_params:.4f}")
        except ValueError as e:
            results_summary_sr3.append(f"Levene's Test (Parameter Counts): Could not be computed. Error: {e}")
            print(f"Levene's Test (Parameter Counts): Could not be computed. Error: {e}")
            levene_p_params = 0 # Assume variances not homogeneous if test fails

        print("\nPerforming Kruskal-Wallis Test on parameter counts...")
        # Check if all groups for Kruskal-Wallis have data
        if all(len(group) > 0 for group in param_counts_by_model_list):
            kw_results_params = pg.kruskal(data=df_features, dv='param_count', between=MODEL_COL) # pg.kruskal handles NaN internally
            if kw_results_params is not None and not kw_results_params.empty:
                print(kw_results_params.round(4))
                results_summary_sr3.append("Kruskal-Wallis Results (Parameter Counts):\n" + kw_results_params.round(4).to_string())
                h_stat_params = kw_results_params['H'].iloc[0]; p_value_kw_params = kw_results_params['p-unc'].iloc[0]
                if 'eps-sq' in kw_results_params.columns:
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


def main():
    print("--- Starting Signature Richness (SR) Analysis ---")
    print(f"DEBUG: Main function started. Outputs will go to ARTICLE_DIR: {ARTICLE_DIR}, STATS_OUTPUT_DIR: {STATS_OUTPUT_DIR}")
    
    all_methods_data_df = load_data(REPORTS_DIR / INPUT_CSV_NAME)
    if all_methods_data_df is None:
        print("Failed to load method data. Exiting SR analysis.")
        return
    print(f"DEBUG: Loaded all_methods_data_df, shape: {all_methods_data_df.shape}")

    df_features = extract_features_for_sr(all_methods_data_df)
    if df_features.empty:
        print("Feature extraction failed or resulted in empty data. Aborting SR analysis.")
        return
    print(f"DEBUG: Extracted features, df_features shape: {df_features.shape}")
    if 'param_count' not in df_features.columns:
        print("DEBUG: 'param_count' column is missing after feature extraction.")
        return
    print(f"DEBUG: param_count describe after extraction:\n{df_features['param_count'].describe()}")


    param_iqr_table_data, param_stats_summary_text = analyze_parameter_richness(df_features)
    print(f"DEBUG: Back in main from analyze_parameter_richness. iqr_table_data is None: {param_iqr_table_data is None}")
    if param_iqr_table_data is not None:
         print(f"DEBUG: iqr_table_data shape: {param_iqr_table_data.shape}")


    if param_stats_summary_text:
        sr3_summary_file_path = STATS_OUTPUT_DIR / "SR3_ParameterRichness_Statistical_Summary.txt"
        try:
            with open(sr3_summary_file_path, 'w', encoding='utf-8') as f:
                f.write(param_stats_summary_text)
            print(f"\nSaved SR3 Parameter Richness statistical summary to {sr3_summary_file_path}")
        except Exception as e:
            print(f"Error saving SR3 summary text file: {e}")
    else:
        print("DEBUG: No statistical summary text generated for SR3.")

    if param_iqr_table_data is not None and not param_iqr_table_data.empty:
        sr_summary_table = param_iqr_table_data.set_index(MODEL_COL)
        partial_sr_table_path = ARTICLE_DIR / "SR_Metrics_Partial_Summary_ParamsIQR.csv" # More specific name
        if not sr_summary_table.empty:
            sr_summary_table.to_csv(partial_sr_table_path, float_format='%.2f')
            print(f"Saved PARTIAL SR metrics table (with Param_IQR) to {partial_sr_table_path}")
        else:
            print("Partial SR metrics table (Param_IQR) is empty, not saving.")
    else:
        print("No IQR data generated for partial SR metrics table.")

    print("\n--- Signature Richness (SR) Analysis (Part 1 - Params) Finished ---")


if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main()