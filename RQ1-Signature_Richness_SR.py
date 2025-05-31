#!/usr/bin/env python3
"""
RQ1-Signature_Richness_SR.py

Generates reports and visualizations for signature richness metrics.
Reads from:
    - reports/Combined_Struct_Counts_Metrics.csv (for pre-calculated ParamRichness, ReturnTypeCompleteness, Redundancy)
    - Raw JSONs via MetricCache (for Naming Adherence, Lexical Diversity proxy, Visibility Markers)
Saves outputs to: reports/article/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter
from Levenshtein import distance as levenshtein_distance
from scipy.stats import iqr

# --- Import shared pipeline components from main3.py ---
try:
    from main3 import MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME, GOLD_STANDARD_MAP_FNAME, safe_divide
except ImportError:
    print("Error: Could not import from main3.py. Make sure it's in the same directory or PYTHONPATH.")
    print("Attempting to define a fallback safe_divide if main3.py import fails for testing purposes.")
    # Fallback safe_divide if main3.py is not available (for isolated testing)
    def safe_divide(n, d, default=np.nan):
        if isinstance(d, pd.Series):
            mask_invalid_d = (d == 0) | d.isnull()
            if isinstance(n, pd.Series):
                result = pd.Series(default, index=n.index.union(d.index), dtype=float)
                n_aligned, d_aligned = n.align(d, join='left', copy=False)
                with np.errstate(divide='ignore', invalid='ignore'):
                    division_result = n_aligned / d_aligned
                result[~mask_invalid_d] = division_result[~mask_invalid_d]
            else: 
                result = pd.Series(default, index=d.index, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    division_result = n / d
                result[~mask_invalid_d] = division_result[~mask_invalid_d]
            return result
        else: 
            if d == 0 or d is None or (isinstance(d, float) and np.isnan(d)): return default # Use np.isnan for float
            try: r = n / d
            except ZeroDivisionError: return default
            except TypeError: return default
            return default if isinstance(r, float) and np.isnan(r) else r


# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_COMBINED_METRICS = REPORTS_DIR / "Combined_Struct_Counts_Metrics.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions for New Metrics ---

def is_camel_case(name: str) -> bool:
    """Checks if a method name is likely camelCase (starts lowercase, no underscores)."""
    if not name or not isinstance(name, str) or not name[0].islower(): # Added isinstance check
        return False
    return bool(re.match(r"^[a-z]+[a-zA-Z0-9]*$", name))

def get_visibility_marker(visibility_json_field: str | None, signature_str: str | None) -> str:
    """
    Determines the visibility marker.
    Prioritizes the dedicated 'visibility_json_field' from the JSON.
    Falls back to parsing the 'signature_str' if the dedicated field is not informative.
    """
    # Priority 1: Use the dedicated visibility field if present and valid
    if isinstance(visibility_json_field, str):
        vis_cleaned = visibility_json_field.strip()
        if vis_cleaned == "+": return "+"
        if vis_cleaned == "-": return "-"
        if vis_cleaned == "#": return "#"
        # Add other direct mappings if your JSON uses "public", "private", etc. in this field
        if vis_cleaned.lower() == "public": return "+"
        if vis_cleaned.lower() == "private": return "-"
        if vis_cleaned.lower() == "protected": return "#"

    # Priority 2: Fallback to parsing the signature string
    if isinstance(signature_str, str):
        sig_norm = signature_str.strip() # Normalize: remove leading/trailing whitespace
        # Check for direct symbols first
        if sig_norm.startswith("+"): return "+" # Handles "+ methodName"
        if sig_norm.startswith("-"): return "-" # Handles "- methodName"
        if sig_norm.startswith("#"): return "#" # Handles "# methodName"
        
        # Then check for keywords (case-insensitive)
        sig_lower = sig_norm.lower()
        if sig_lower.startswith("public "): return "+"
        if sig_lower.startswith("private "): return "-"
        if sig_lower.startswith("protected "): return "#"
        
    return "None"

    """Extracts visibility marker from a signature string."""
    if not isinstance(signature, str): return "None" # Default if not a string
    sig_lower = signature.lower().strip() # Strip whitespace
    if sig_lower.startswith("public "): return "+"
    if sig_lower.startswith("private "): return "-" # Using hyphen-minus
    if sig_lower.startswith("protected "): return "#"
    return "None" # Default if no explicit marker


def calculate_normalized_levenshtein(s1: str, s2: str) -> float:
    """Calculates normalized Levenshtein distance between two strings."""
    if not isinstance(s1, str): s1 = str(s1) # Ensure strings
    if not isinstance(s2, str): s2 = str(s2)

    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 1.0 
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 0.0 # Should be caught by previous checks
    return levenshtein_distance(s1, s2) / max_len

def calculate_lexical_diversity_model(method_names: list[str]) -> float | None:
    """
    Calculates mean normalized Levenshtein distance among a list of method names.
    """
    if not method_names or len(method_names) < 2:
        return None 

    unique_names = sorted(list(set(str(name) for name in method_names if name))) # Ensure names are strings and not None/empty
    if len(unique_names) < 2:
        return 0.0 

    distances = []
    # Limit pairs for performance if many unique names - e.g. max 500 unique names for N^2 comparison
    # For a more robust solution on very large datasets, consider sampling.
    limit_unique_names = 500 
    if len(unique_names) > limit_unique_names:
        print(f"    Limiting LexDiv calculation to {limit_unique_names} unique names (out of {len(unique_names)}) for performance.")
        # Potentially sample unique_names, or just take the first N
        # For simplicity, let's take a slice, but random sampling would be better for representativeness
        unique_names_sample = unique_names[:limit_unique_names]
    else:
        unique_names_sample = unique_names

    for i in range(len(unique_names_sample)):
        for j in range(i + 1, len(unique_names_sample)):
            distances.append(calculate_normalized_levenshtein(unique_names_sample[i], unique_names_sample[j]))
            
    return np.mean(distances) if distances else 0.0


# --- Main Data Processing and Table Generation ---
def generate_sr_metrics(cache: MetricCache, df_combined_metrics: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates all SR metrics per model.
    """
    all_model_sr_data = []

    models_from_cache = []
    if not cache.global_details_df.empty and 'model' in cache.global_details_df.columns:
        models_from_cache = sorted(cache.global_details_df['model'].unique())
    
    models_from_combined = []
    if not df_combined_metrics.empty and 'Model' in df_combined_metrics.columns:
        models_from_combined = sorted(df_combined_metrics['Model'].unique())

    models = sorted(list(set(models_from_cache) | set(models_from_combined)))

    if not models:
        print("No models found to process for SR metrics.")
        return None

    for model_name in models:
        model_data = {}
        model_data['Model'] = model_name

        model_combined_metrics = df_combined_metrics[df_combined_metrics['Model'] == model_name] if not df_combined_metrics.empty else pd.DataFrame()
        
        if not model_combined_metrics.empty and 'ReturnTypeCompleteness' in model_combined_metrics.columns:
            ret_type_completeness_series = model_combined_metrics['ReturnTypeCompleteness']
            model_data['Ret. (%)'] = ret_type_completeness_series.mean() if not ret_type_completeness_series.empty else np.nan
        else:
            model_data['Ret. (%)'] = np.nan

        model_methods_df = pd.DataFrame()
        if not cache.global_details_df.empty and 'model' in cache.global_details_df.columns:
            model_methods_df = cache.global_details_df[cache.global_details_df['model'] == model_name]

        if not model_methods_df.empty:
            # Naming Adherence (camelCase %)
            all_method_names = model_methods_df['name'].dropna().tolist() # Used for camelCase and LexDiv
            camel_case_count = sum(1 for name in all_method_names if is_camel_case(name))
            model_data['camelCase (%)'] = safe_divide(camel_case_count * 100.0, len(all_method_names), 0.0)

            # Visibility Marker Usage
            # Ensure 'visibility_json' and 'signature' columns exist
            if 'visibility_json' not in model_methods_df.columns:
                print(f"Warning: 'visibility_json' column missing in data for model {model_name}. Adding default.")
                model_methods_df['visibility_json'] = None 
            if 'signature' not in model_methods_df.columns:
                print(f"Warning: 'signature' column missing in data for model {model_name}. Adding default.")
                model_methods_df['signature'] = None
            
            # --- CORRECTED CALL TO get_visibility_marker ---
            visibility_counts = Counter(
                get_visibility_marker(row['visibility_json'], row['signature'])
                for _, row in model_methods_df.iterrows() 
            )
            # --- END OF CORRECTION ---
            
            total_methods_for_visibility = len(model_methods_df)
            
            model_data['+'] = safe_divide(visibility_counts.get('+', 0) * 100.0, total_methods_for_visibility, 0.0)
            model_data['-'] = safe_divide(visibility_counts.get('-', 0) * 100.0, total_methods_for_visibility, 0.0)
            model_data['#'] = safe_divide(visibility_counts.get('#', 0) * 100.0, total_methods_for_visibility, 0.0)
            model_data['None_Visibility'] = safe_divide(visibility_counts.get('None', 0) * 100.0, total_methods_for_visibility, 0.0)

            # Lexical Diversity
            print(f"Calculating lexical diversity for {model_name} (unique names: {len(set(all_method_names))})...")
            lex_div_val = calculate_lexical_diversity_model(all_method_names)
            model_data['LexDiv'] = lex_div_val if lex_div_val is not None else np.nan
            if pd.notna(model_data['LexDiv']): print(f"  LexDiv for {model_name}: {model_data['LexDiv']:.3f}")
            else: print(f"  LexDiv for {model_name}: Not calculated or N/A")

            # Parameter Richness related metrics (IQR)
            param_counts_per_method_model = []
            model_files_for_params = model_methods_df['file'].unique()
            for file_name in model_files_for_params:
                if file_name in cache.json_data:
                    json_content = cache.json_data[file_name]
                    if cache.file_info.get(file_name, {}).get('model') == model_name:
                        for cls_item in json_content.get('classes', []):
                            for method_item in cls_item.get('methods', []):
                                if method_item.get('name'):
                                    param_counts_per_method_model.append(len(method_item.get('parameters', [])))
            model_data['IQR_ParamRichness'] = iqr(param_counts_per_method_model) if param_counts_per_method_model else np.nan
        else:
            model_data['camelCase (%)'] = np.nan
            model_data['+'] = np.nan
            model_data['-'] = np.nan
            model_data['#'] = np.nan
            model_data['None_Visibility'] = np.nan
            model_data['LexDiv'] = np.nan
            model_data['IQR_ParamRichness'] = np.nan

        all_model_sr_data.append(model_data)

    if not all_model_sr_data:
        print("No SR data generated.")
        return None
        
    sr_summary_df = pd.DataFrame(all_model_sr_data)
    column_order = ['Model', 'IQR_ParamRichness', 'Ret. (%)', 'camelCase (%)', 'LexDiv', 
                    '+', '-', '#', 'None_Visibility'] 
    for col in column_order:
        if col not in sr_summary_df.columns: sr_summary_df[col] = np.nan
            
    return sr_summary_df[column_order]

# --- Plotting Functions ---
def plot_param_richness_barchart(df_combined_metrics: pd.DataFrame, output_path: Path):
    if df_combined_metrics.empty or 'Model' not in df_combined_metrics.columns or 'ParamRichness' not in df_combined_metrics.columns:
        print("Skipping ParamRichness barchart: DataFrame empty or missing Model/ParamRichness column.")
        return

    model_stats = df_combined_metrics.groupby('Model')['ParamRichness'].agg(['mean', 'std', 'count']).copy() # Use .copy()
    model_stats['sem'] = model_stats['std'] / np.sqrt(model_stats['count'])
    model_stats.sort_values(by='mean', ascending=False, inplace=True)

    plt.figure(figsize=(12, 7))
    plt.bar(model_stats.index, model_stats['mean'], yerr=model_stats['sem'] * 1.96, capsize=5, color=sns.color_palette("coolwarm", len(model_stats)))
    plt.xlabel("LLM Model", fontsize=12)
    plt.ylabel("Mean Parameter Richness per Method", fontsize=12) # Clarified label
    plt.title("Mean Parameter Richness (Error Bars: Approx. 95% CI from SEM of per-run/file averages)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try: plt.savefig(output_path); print(f"Saved ParamRichness bar chart to {output_path}")
    except Exception as e: print(f"Error saving ParamRichness bar chart: {e}")
    plt.close()

def plot_param_count_violin(cache: MetricCache, output_path: Path):
    if cache.global_details_df.empty:
        print("Skipping violin plot: global_details_df is empty in cache.")
        return

    param_counts_data = []
    models = sorted(cache.global_details_df['model'].unique()) if 'model' in cache.global_details_df else []
    for model_name in models:
        # Iterate through raw JSON data associated with this model
        # Get files for this model from cache.file_info
        model_files = [f_name for f_name, info in cache.file_info.items() if info['model'] == model_name]

        for file_name in model_files:
            if file_name in cache.json_data:
                json_content = cache.json_data[file_name]
                for cls_item in json_content.get('classes', []):
                    # SR script uses cache with class_names_list=[], so all classes are in scope
                    for method_item in cls_item.get('methods', []):
                        if method_item.get('name'):
                            param_counts_data.append({'Model': model_name, 'ParamCount': len(method_item.get('parameters', []))})
    if not param_counts_data:
        print("No parameter count data for violin plot. Skipping.")
        return

    violin_df = pd.DataFrame(param_counts_data)
    if violin_df.empty:
        print("Violin DataFrame is empty. Skipping plot.")
        return

    plt.figure(figsize=(14, 8))
    # Sort models by median parameter count for ordered violin plot
    order = violin_df.groupby('Model')['ParamCount'].median().sort_values(ascending=False).index
    sns.violinplot(x='Model', y='ParamCount', data=violin_df, order=order, palette="muted", inner="quartile", cut=0, scale="width")
    plt.xlabel("LLM Model", fontsize=12)
    plt.ylabel("Number of Parameters per Method", fontsize=12)
    plt.title("Parameter Count Distribution per Model", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    try: plt.savefig(output_path); print(f"Saved parameter count violin plot to {output_path}")
    except Exception as e: print(f"Error saving violin plot: {e}")
    plt.close()

# In RQ1-Signature_Richness_SR.py

# ... (other functions remain the same) ...

def plot_lexdiv_vs_redundancy_pointcloud(sr_df: pd.DataFrame, df_combined_metrics: pd.DataFrame, output_path: Path) -> str:
    """
    Point cloud: Lexical diversity (X-axis) vs. redundancy (Y-axis) for each LLM.
    Omits the legend from the plot and sets specific Y-axis ticks.
    Returns a string with LaTeX code for a potential manual legend.
    """
    latex_legend_str = "\\% LaTeX code for legend items (example, adjust as needed):\n"
    latex_legend_str += "\\% \\begin{itemize}\n"

    if sr_df.empty or 'Model' not in sr_df.columns or 'LexDiv' not in sr_df.columns:
        print("Skipping LexDiv vs Redundancy plot: SR DataFrame empty or missing Model/LexDiv.")
        return latex_legend_str + "\\% No data for legend.\n\\% \\end{itemize}\n"
    if df_combined_metrics.empty or 'Model' not in df_combined_metrics.columns or 'Redundancy' not in df_combined_metrics.columns:
        print("Skipping LexDiv vs Redundancy plot: Combined Metrics DataFrame empty or missing Model/Redundancy.")
        return latex_legend_str + "\\% No data for legend.\n\\% \\end{itemize}\n"

    model_redundancy = df_combined_metrics.groupby('Model')['Redundancy'].mean().rename("MeanRedundancy")
    plot_df = sr_df.merge(model_redundancy, on="Model", how="inner")

    if plot_df.empty or 'LexDiv' not in plot_df.columns or 'MeanRedundancy' not in plot_df.columns:
         print("Skipping LexDiv vs Redundancy plot: Merged DataFrame for plot is empty or missing columns.")
         return latex_legend_str + "\\% No data for legend.\n\\% \\end{itemize}\n"
    
    plot_df = plot_df.dropna(subset=['LexDiv', 'MeanRedundancy'])
    if plot_df.empty:
        print("No valid data points for LexDiv vs Redundancy plot after dropping NaNs. Skipping.")
        return latex_legend_str + "\\% No data for legend.\n\\% \\end{itemize}\n"

    plt.figure(figsize=(10, 8))
    
    num_models = plot_df['Model'].nunique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    palette = sns.color_palette("viridis", n_colors=num_models) if num_models > 0 else None

    sns.scatterplot(
        x='LexDiv', 
        y='MeanRedundancy', 
        hue='Model', 
        style='Model',
        markers=[markers[i % len(markers)] for i in range(num_models)] if num_models > 0 else False,
        data=plot_df, 
        s=120, 
        palette=palette, 
        legend=False
    )
    
    for i in range(plot_df.shape[0]):
        model_name = plot_df['Model'].iloc[i]
        plt.text(
            plot_df['LexDiv'].iloc[i] + 0.001 * plot_df['LexDiv'].max(skipna=True), # skipna for max
            plot_df['MeanRedundancy'].iloc[i], 
            model_name, 
            fontdict={'size': 9}
        )
        marker_index = plot_df['Model'].astype('category').cat.codes.iloc[i]
        latex_legend_str += f"\\%   \\item[{markers[marker_index % len(markers)]}] {model_name.replace('_', '\\_')}\n"

    plt.xlabel("Mean Normalized Levenshtein Distance (Lexical Diversity)", fontsize=12)
    plt.ylabel("Mean Redundancy (Avg Methods per Run / Avg Unique Names per Run)", fontsize=12)
    plt.title("Lexical Diversity vs. Redundancy per LLM", fontsize=14)
    
    # --- SET SPECIFIC Y-AXIS TICKS ---
    y_ticks_custom = [1.0, 1.05, 1.10]
    plt.yticks(y_ticks_custom)
    # Optionally, set Y-axis limits if your data might go significantly beyond these ticks
    # For example, if min redundancy is below 1 or max is well above 1.15
    min_y_data = plot_df['MeanRedundancy'].min(skipna=True)
    max_y_data = plot_df['MeanRedundancy'].max(skipna=True)
    # Set Y limits to include your custom ticks and a bit of padding if necessary
    # Ensure the limits encompass the custom ticks.
    y_lim_lower = min(y_ticks_custom[0] - 0.02, min_y_data - 0.02) if pd.notna(min_y_data) else y_ticks_custom[0] - 0.02
    y_lim_upper = max(y_ticks_custom[-1] + 0.02, max_y_data + 0.02) if pd.notna(max_y_data) else y_ticks_custom[-1] + 0.02
    if pd.notna(y_lim_lower) and pd.notna(y_lim_upper) and y_lim_lower < y_lim_upper:
        plt.ylim(y_lim_lower, y_lim_upper)
    else: # Fallback if limits are problematic
        print("Warning: Could not determine appropriate Y-axis limits automatically. Using Matplotlib defaults or just custom ticks.")


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() 
    
    try:
        plt.savefig(output_path)
        print(f"Saved LexDiv vs Redundancy point cloud to {output_path}")
    except Exception as e: 
        print(f"Error saving point cloud: {e}")
    plt.close()

    latex_legend_str += "\\% \\end{itemize}\n"
    return latex_legend_str

# --- Main Execution ---
def main():
    print("Starting RQ1 - Signature Richness (SR) analysis...")
    
    df_combined = None
    if INPUT_CSV_COMBINED_METRICS.is_file():
        try:
            df_combined = pd.read_csv(INPUT_CSV_COMBINED_METRICS)
            print(f"Successfully loaded data from {INPUT_CSV_COMBINED_METRICS}")
        except Exception as e:
            print(f"Error loading CSV from {INPUT_CSV_COMBINED_METRICS}: {e}")
            df_combined = pd.DataFrame() # Use empty DataFrame if loading fails but file exists
    else:
        print(f"Error: {INPUT_CSV_COMBINED_METRICS} not found. Some SR metrics might be incomplete.")
        df_combined = pd.DataFrame() # Initialize empty if file not found

    print("Initializing MetricCache to access raw JSON data for detailed SR metrics...")
    try:
        # Initialize MetricCache with empty CLASS_NAMES to access all methods for SR specific metrics
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=[])
        print("MetricCache initialized for SR.")
    except NameError as e: # Catch if JSON_INPUT_DIR etc. not defined due to main3 import issue
        print(f"NameError during MetricCache initialization (likely main3.py components not imported): {e}")
        print("SR analysis requiring raw JSON data will be skipped.")
        cache = None # Ensure cache is None if init fails
    except FileNotFoundError as e:
        print(f"FileNotFoundError during MetricCache initialization (e.g., baseline file): {e}")
        cache = None
    except Exception as e:
        print(f"An unexpected error occurred during MetricCache initialization: {e}")
        cache = None


    # 1. Generate SR Metrics Table
    print("\nGenerating SR metrics table...")
    sr_summary_df = None
    if cache is not None: # Proceed only if cache was initialized
        sr_summary_df = generate_sr_metrics(cache, df_combined if df_combined is not None else pd.DataFrame())
    
    if sr_summary_df is not None and not sr_summary_df.empty:
        sr_table_output_path = ARTICLE_DIR / "SR_Signature_Richness_Summary.csv"
        try:
            sr_summary_df.to_csv(sr_table_output_path, index=False, float_format='%.2f')
            print(f"Saved SR summary table to {sr_table_output_path}")
        except Exception as e: print(f"Error saving SR summary table: {e}")
    else:
        print("SR metrics table generation failed or produced no data.")

    # 2. Generate Bar Chart for Parameter Richness
    print("\nGenerating bar chart for Mean Parameter Richness...")
    if df_combined is not None and not df_combined.empty:
        plot_param_richness_barchart(df_combined, ARTICLE_DIR / "SR_BarChart_MeanParamRichness.png")
    else:
        print("Skipping Mean Parameter Richness bar chart as combined metrics data is unavailable.")
    
    # 3. Generate Violin Plot for Parameter Count Distribution
    print("\nGenerating violin plot for Parameter Count Distribution...")
    if cache is not None:
        plot_param_count_violin(cache, ARTICLE_DIR / "SR_Violin_ParamCountDistribution.png")
    else:
        print("Skipping Parameter Count Distribution violin plot as MetricCache failed to initialize.")

    # 4. Generate Point Cloud for Lexical Diversity vs. Redundancy
    print("\nGenerating point cloud for Lexical Diversity vs. Redundancy...")
    if sr_summary_df is not None and not sr_summary_df.empty and \
       df_combined is not None and not df_combined.empty:
        latex_legend = plot_lexdiv_vs_redundancy_pointcloud(
            sr_summary_df, 
            df_combined, 
            ARTICLE_DIR / "SR_PointCloud_LexDiv_vs_Redundancy.png"
        )
        print("\nLaTeX code snippet for manual legend (customize as needed):")
        print(latex_legend)

    else:
        print("Skipping LexDiv vs Redundancy plot as SR summary or combined metrics data is not available.")
        
    print("\nRQ1 - Signature Richness (SR) analysis finished.")

if __name__ == "__main__":
    # Attempt to set a style, fall back if seaborn is problematic on some setups
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        print("Seaborn style 'seaborn-v0_8-whitegrid' not found or error applying. Using default Matplotlib style.")
        pass # Use Matplotlib default style
    main()