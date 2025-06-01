#!/usr/bin/env python3
"""
RQ1_Method_Quantity_MQ_Stats.py
# ... (rest of docstring as before) ...
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp 
import pingouin as pg       

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "Combined_Struct_Counts_Metrics.csv"
ARTICLE_DIR = REPORTS_DIR / "article" 
STATS_MQ_OUTPUT_DIR = ARTICLE_DIR / "stats_mq" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_MQ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_METHOD_COUNT_COL = 'Total_Methods_Generated_Per_Run' 
P_VALUE_THRESHOLD = 0.05 

# --- Custom Color Palette and Order ---
# IMPORTANT: Keys in LLM_COLORS and names in LLM_ORDER_FOR_PLOTS 
# MUST EXACTLY MATCH the unique names in your DataFrame's 'Model' column.
# Please verify these against your data. I'm using names based on previous CSV examples.

LLM_COLORS = {
    'Claude3.7': '#E69F00',                     # Orange/Gold
    'Gemini-2.5-Pro-Preview-05-06': '#D55E00',  # Vermillion/Burnt Orange (Matches your example data)
    'ChatGPT-o3': '#009E73',                    # Bluish Green
    'Qwen3': '#56B4E9',                         # Sky Blue
    'DeepSeek(R1)': '#CC79A7',                  # Reddish Purple/Mauve (Matches your example data)
    'Grok3': '#0072B2',                         # Darker Blue
    'ChatGPT-4o': '#8C8C00',                    # Dark Yellow/Olive Green
    'Llama4': '#E76F51',                        # Reddish Coral
    'Mistral': '#999999'                        # Grey (Assuming 'Mistral' is the name in data, not 'Mistral 8x7B')
}

# Desired order for plots
LLM_ORDER_FOR_PLOTS = [
    'Claude3.7',
    'Gemini-2.5-Pro-Preview-05-06', # Verify this exact name from your data
    'ChatGPT-o3',
    'Qwen3',
    'DeepSeek(R1)',                 # Verify this exact name
    'Grok3',
    'ChatGPT-4o',
    'Llama4',
    'Mistral'                       # Verify this exact name
]
# --- End Custom Color Palette and Order ---


def load_and_prepare_data(filepath: Path) -> pd.DataFrame | None:
    # ... (same as your last working version) ...
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        required_cols = ['Model', 'Run', 'Target_Class_Methods', 'Extra_Methods_Count']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: DataFrame missing one or more required columns from: {required_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        df[PRIMARY_METHOD_COUNT_COL] = df['Target_Class_Methods'].fillna(0) + df['Extra_Methods_Count'].fillna(0)
        print(f"Created '{PRIMARY_METHOD_COUNT_COL}' by summing 'Target_Class_Methods' and 'Extra_Methods_Count'.")
        
        df_mq = df[~df['Model'].astype(str).str.lower().str.startswith('grand total')].copy()
        df_mq = df_mq[['Model', PRIMARY_METHOD_COUNT_COL]].copy() 
        
        df_mq[PRIMARY_METHOD_COUNT_COL] = pd.to_numeric(df_mq[PRIMARY_METHOD_COUNT_COL], errors='coerce')
        df_mq.dropna(subset=[PRIMARY_METHOD_COUNT_COL], inplace=True)

        if df_mq.empty:
            print("Error: DataFrame is empty after preparation for MQ analysis.")
            return None
        
        # Ensure model names in data match the keys in LLM_COLORS and LLM_ORDER_FOR_PLOTS
        unique_models_in_data = df_mq['Model'].unique()
        for model_name in unique_models_in_data:
            if model_name not in LLM_COLORS:
                print(f"Warning: Model '{model_name}' from data not found in LLM_COLORS keys. Plot colors might be inconsistent.")
            if model_name not in LLM_ORDER_FOR_PLOTS:
                 print(f"Warning: Model '{model_name}' from data not found in LLM_ORDER_FOR_PLOTS. Plot order might be affected.")


        if df_mq['Model'].nunique() < 2:
            print("Error: Less than 2 models available for comparison. Statistical tests require at least 2 groups.")
            return None
        return df_mq
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def cohens_d_from_groups(group1_data, group2_data):
    # ... (same as your last working version) ...
    n1, n2 = len(group1_data), len(group2_data)
    if n1 < 2 or n2 < 2: return np.nan 
    mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
    std1, std2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
    if std1 == 0 and std2 == 0 and mean1 == mean2: return 0.0
    if std1 == 0 and std2 == 0 and mean1 != mean2: return np.inf 
    s_pooled_numerator = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2)
    s_pooled_denominator = (n1 + n2 - 2)
    if s_pooled_denominator == 0: return np.nan 
    pooled_std = np.sqrt(s_pooled_numerator / s_pooled_denominator)
    if pooled_std == 0: return np.inf if (mean1 - mean2) != 0 else 0.0
    d = (mean1 - mean2) / pooled_std
    return d

def main_mq_stats():
    print(f"--- Starting MQ Statistical Analysis ({PRIMARY_METHOD_COUNT_COL}) ---")
    results_summary = [f"--- MQ Statistical Analysis ({PRIMARY_METHOD_COUNT_COL}) ---"]

    mq_data_df = load_and_prepare_data(REPORTS_DIR / INPUT_CSV_NAME)

    if mq_data_df is None or mq_data_df.empty:
        print("MQ data preparation failed. Aborting MQ stats.")
        results_summary.append("MQ data preparation failed. Aborting MQ stats.")
        summary_file_path = STATS_MQ_OUTPUT_DIR / f"MQ_Stats_Summary_{PRIMARY_METHOD_COUNT_COL}.txt"
        with open(summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(results_summary))
        return

    desc_stats = mq_data_df.groupby('Model')[PRIMARY_METHOD_COUNT_COL].describe()
    results_summary.append("\nDescriptive Statistics for Method Quantity:\n" + desc_stats.to_string())
    print("\nDescriptive Statistics for Method Quantity:\n", desc_stats)

    results_summary.append("\n--- Assumption Checks ---")
    print("\n--- Assumption Checks ---")
    all_groups_normal = True
    normality_details = []
    for model_name, group_data in mq_data_df.groupby('Model'):
        count_data = group_data[PRIMARY_METHOD_COUNT_COL].dropna()
        if len(count_data) >= 3:
            stat_sw, p_sw = stats.shapiro(count_data)
            is_normal = p_sw > P_VALUE_THRESHOLD
            normality_details.append(f"  Model {model_name} (N={len(count_data)}): W={stat_sw:.3f}, p={p_sw:.4f} {'(Normal)' if is_normal else '(Not Normal)'}")
            if not is_normal: all_groups_normal = False
        else:
            normality_details.append(f"  Model {model_name}: Not enough data for Shapiro-Wilk (N={len(count_data)}). Assuming not normal."); all_groups_normal = False
    results_summary.extend(["Shapiro-Wilk Normality Test (p > 0.05 suggests normality):"] + normality_details)
    print("\n".join(["Shapiro-Wilk Normality Test (p > 0.05 suggests normality):"] + normality_details))
    
    model_groups_for_levene = [g[PRIMARY_METHOD_COUNT_COL].dropna().values for _, g in mq_data_df.groupby('Model') if len(g[PRIMARY_METHOD_COUNT_COL].dropna()) > 1]
    variances_homogeneous = False
    if len(model_groups_for_levene) >= 2 :
        valid_groups_for_levene = [g for g in model_groups_for_levene if len(g) > 0]
        if len(valid_groups_for_levene) >=2:
            levene_stat, levene_p = stats.levene(*valid_groups_for_levene)
            variances_homogeneous = levene_p > P_VALUE_THRESHOLD
            results_summary.append(f"\nLevene's Test: Statistic={levene_stat:.3f}, p={levene_p:.4f} {'(Homogeneous)' if variances_homogeneous else '(Not Homogeneous)'}")
            print(f"\nLevene's Test: Statistic={levene_stat:.3f}, p={levene_p:.4f} {'(Homogeneous)' if variances_homogeneous else '(Not Homogeneous)'}")
        else:
            results_summary.append("\nNot enough valid groups for Levene's. Assuming not homogeneous."); print("\nNot enough valid groups for Levene's. Assuming not homogeneous.")
    else:
        results_summary.append("\nNot enough groups for Levene's. Assuming not homogeneous."); print("\nNot enough groups for Levene's. Assuming not homogeneous.")

    use_parametric = all_groups_normal and variances_homogeneous
    results_summary.append(f"Decision: Use {'Parametric (ANOVA)' if use_parametric else 'Non-Parametric (Kruskal-Wallis)'} based on assumptions.")
    print(f"Decision: Use {'Parametric (ANOVA)' if use_parametric else 'Non-Parametric (Kruskal-Wallis)'} based on assumptions.")

    results_summary.append("\n--- Omnibus Test ---"); print("\n--- Omnibus Test ---")
    omnibus_p_value, significant_omnibus = 1.0, False
    eta_sq_effect, epsilon_sq_effect = 'N/A', 'N/A'

    if use_parametric:
        aov = pg.anova(data=mq_data_df, dv=PRIMARY_METHOD_COUNT_COL, between='Model', detailed=True)
        results_summary.append("ANOVA Results:\n" + aov.round(4).to_string()); print(aov.round(4))
        model_effect_row = aov[aov['Source'] == 'Model']
        if not model_effect_row.empty:
            f_value, omnibus_p_value, eta_sq_effect = model_effect_row[['F', 'p-unc', 'eta-sq']].iloc[0]
        else: f_value, omnibus_p_value, eta_sq_effect = np.nan, 1.0, np.nan
        results_summary.append(f"ANOVA: F={f_value:.3f}, p={omnibus_p_value:.4f}, Eta-squared (η²)={eta_sq_effect:.3f}")
        print(f"ANOVA: F={f_value:.3f}, p={omnibus_p_value:.4f}, Eta-squared (η²)={eta_sq_effect:.3f}")
        if omnibus_p_value < P_VALUE_THRESHOLD: significant_omnibus = True
    else:
        kw_results_pg = pg.kruskal(data=mq_data_df, dv=PRIMARY_METHOD_COUNT_COL, between='Model')
        results_summary.append("Kruskal-Wallis Results:\n" + kw_results_pg.round(4).to_string()); print(kw_results_pg.round(4))
        if not kw_results_pg.empty:
            h_stat, omnibus_p_value = kw_results_pg[['H', 'p-unc']].iloc[0]
            epsilon_sq_effect = kw_results_pg['eps-sq'].iloc[0] if 'eps-sq' in kw_results_pg.columns and pd.notna(kw_results_pg['eps-sq'].iloc[0]) else 'N/A'
        else: h_stat, omnibus_p_value, epsilon_sq_effect = np.nan, 1.0, 'N/A'
        results_summary.append(f"Kruskal-Wallis: H={h_stat:.3f}, p={omnibus_p_value:.4f}, Epsilon-squared (ε²)={epsilon_sq_effect if isinstance(epsilon_sq_effect, str) else f'{epsilon_sq_effect:.3f}'}")
        print(f"Kruskal-Wallis: H={h_stat:.3f}, p={omnibus_p_value:.4f}, Epsilon-squared (ε²)={epsilon_sq_effect if isinstance(epsilon_sq_effect, str) else f'{epsilon_sq_effect:.3f}'}")
        if omnibus_p_value < P_VALUE_THRESHOLD: significant_omnibus = True

    if significant_omnibus:
        results_summary.append("\n--- Post-Hoc Tests (Bonferroni Corrected) ---")
        print("\n--- Post-Hoc Tests (Bonferroni Corrected) ---")
        if use_parametric:
            tukey_results = pairwise_tukeyhsd(mq_data_df[PRIMARY_METHOD_COUNT_COL], mq_data_df['Model'], alpha=P_VALUE_THRESHOLD)
            results_summary.append("Tukey HSD Results:\n" + str(tukey_results)) ; print(tukey_results)
            tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
            significant_pairs_tukey = tukey_df[tukey_df['reject'] == True]
            if not significant_pairs_tukey.empty:
                results_summary.append("\nCohen's d for significant Tukey HSD pairs:")
                print("\nCohen's d for significant Tukey HSD pairs:")
                for _, row in significant_pairs_tukey.iterrows():
                    g1d = mq_data_df[mq_data_df['Model'] == row['group1']][PRIMARY_METHOD_COUNT_COL].dropna()
                    g2d = mq_data_df[mq_data_df['Model'] == row['group2']][PRIMARY_METHOD_COUNT_COL].dropna()
                    d = cohens_d_from_groups(g1d, g2d)
                    results_summary.append(f"  {row['group1']} vs {row['group2']}: Cohen's d = {d:.3f}"); print(f"  {row['group1']} vs {row['group2']}: Cohen's d = {d:.3f}")
        else: 
            dunn_results = sp.posthoc_dunn(mq_data_df, val_col=PRIMARY_METHOD_COUNT_COL, group_col='Model', p_adjust='bonferroni')
            results_summary.append("Dunn's Test Results:\n" + dunn_results.round(4).to_string()); print(dunn_results.round(4))
            results_summary.append("\nRank-Biserial Correlation for significant Dunn's pairs (from pairwise Mann-Whitney U):")
            print("\nRank-Biserial Correlation for significant Dunn's pairs (from pairwise Mann-Whitney U):")
            
            models_in_data = mq_data_df['Model'].unique()
            for i in range(len(models_in_data)):
                for j in range(i + 1, len(models_in_data)):
                    model1_name = models_in_data[i] 
                    model2_name = models_in_data[j]
                    p_dunn_val = np.nan
                    if dunn_results is not None:
                        if model1_name in dunn_results.index and model2_name in dunn_results.columns: p_dunn_val = dunn_results.loc[model1_name, model2_name]
                        elif model2_name in dunn_results.index and model1_name in dunn_results.columns: p_dunn_val = dunn_results.loc[model2_name, model1_name]
                    if pd.notna(p_dunn_val) and p_dunn_val < P_VALUE_THRESHOLD:
                        g1d = mq_data_df[mq_data_df['Model'] == model1_name][PRIMARY_METHOD_COUNT_COL].dropna()
                        g2d = mq_data_df[mq_data_df['Model'] == model2_name][PRIMARY_METHOD_COUNT_COL].dropna()
                        n1, n2 = len(g1d), len(g2d)
                        if n1 > 0 and n2 > 0: 
                            mwu_res = pg.mwu(g1d, g2d, alternative='two-sided')
                            rbc_val_pg = mwu_res['r_RB'].iloc[0] if 'r_RB' in mwu_res.columns and not mwu_res['r_RB'].empty and pd.notna(mwu_res['r_RB'].iloc[0]) else None
                            rbc_display = "N/A"
                            if rbc_val_pg is not None: rbc_display = f"{rbc_val_pg:.3f}"
                            else: 
                                u_val = mwu_res['U-val'].iloc[0] if 'U-val' in mwu_res.columns and not mwu_res['U-val'].empty and pd.notna(mwu_res['U-val'].iloc[0]) else None
                                if u_val is not None and n1 > 0 and n2 > 0:
                                    try: rbc_manual = 1 - (2 * u_val) / (n1 * n2); rbc_display = f"{rbc_manual:.3f} (manual)"
                                    except ZeroDivisionError: pass 
                                    except Exception: pass 
                            results_summary.append(f"  {model1_name} vs {model2_name} (pDunn={p_dunn_val:.4f}): r_rb = {rbc_display}"); print(f"  {model1_name} vs {model2_name} (pDunn={p_dunn_val:.4f}): r_rb = {rbc_display}")
                        else: print(f"  Skipping MWU/RBC for {model1_name} vs {model2_name} due to empty group(s).")
    else:
        results_summary.append("\nOmnibus test not significant. No post-hoc tests performed.")
        print("\nOmnibus test not significant. No post-hoc tests performed.")

    print("\n--- Generating Box Plot ---")
    plt.figure(figsize=(12, 8)) 
    
    # Filter LLM_ORDER_FOR_PLOTS to only include models present in mq_data_df
    # and ensure LLM_COLORS has these models
    models_present_in_data = mq_data_df['Model'].unique()
    plot_order = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_data]
    
    # Create a palette dictionary for only the models being plotted, in their specified order
    plot_palette = {model: LLM_COLORS.get(model, '#000000') for model in plot_order} # Default to black if missing

    ax = sns.boxplot(x='Model', y=PRIMARY_METHOD_COUNT_COL, data=mq_data_df, 
                     order=plot_order, palette=plot_palette, showfliers=True) # Use plot_order and plot_palette
    # Overlay stripplot, also using the order and a consistent color or individual colors
    # To ensure stripplot points align with boxplot colors:
    strip_colors_ordered = [plot_palette[model] for model in plot_order]

    # Create a new DataFrame for stripplot to ensure correct color mapping with hue
    # Or, iterate and plot strip for each model if direct palette mapping is tricky with hue
    # For simplicity, let's use a single darker color for all points.
    sns.stripplot(x='Model', y=PRIMARY_METHOD_COUNT_COL, data=mq_data_df, 
                  order=plot_order, color=".25", size=4, jitter=0.15, ax=ax, alpha=0.6)
    
    plot_title = f'Method Quantity ({PRIMARY_METHOD_COUNT_COL}) per LLM'
    if significant_omnibus:
        plot_title += f'\nOverall significant difference (p={omnibus_p_value:.4f})'
        if use_parametric and eta_sq_effect != 'N/A' and not pd.isna(eta_sq_effect): plot_title += f"; ANOVA, η²={eta_sq_effect:.3f}"
        elif not use_parametric and epsilon_sq_effect != 'N/A' and not pd.isna(epsilon_sq_effect): plot_title += f"; Kruskal-Wallis, ε²={epsilon_sq_effect if isinstance(epsilon_sq_effect, str) else f'{epsilon_sq_effect:.3f}'}"
    
    plt.title(plot_title, fontsize=14, pad=20) 
    plt.xlabel("LLM Model", fontsize=12); plt.ylabel(f"Methods per Run ({PRIMARY_METHOD_COUNT_COL})", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plot_output_path = ARTICLE_DIR / f"MQ_Boxplot_{PRIMARY_METHOD_COUNT_COL}.png" # Matches figure label
    plt.savefig(plot_output_path); print(f"Saved MQ box plot to {plot_output_path}"); plt.close()
    results_summary.append(f"\nBox plot saved to: {plot_output_path}")

    summary_file_path = STATS_MQ_OUTPUT_DIR / f"MQ_Stats_Summary_{PRIMARY_METHOD_COUNT_COL}.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f: f.write("\n".join(results_summary))
    print(f"\nSaved statistical summary to {summary_file_path}")
    print("\n--- MQ Statistical Analysis Finished ---")

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError: print("Warning: Seaborn style not found.")
    main_mq_stats()