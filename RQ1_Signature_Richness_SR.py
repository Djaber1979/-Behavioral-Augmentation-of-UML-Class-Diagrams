#!/usr/bin/env python3
"""
RQ1_Signature_Richness_SR.py

Analyzes multiple dimensions of Signature Richness (SR):
- SR1: Visibility Markers
- SR2: Naming Conventions (camelCase check)
- SR3: Parameter Richness (IQR, Means, Distributions, Stats)
- SR4: Return Types (Proportions, Stats)
- SR5: Lexical Coherence (LexDiv, Redundancy, Scatter Plot, Stats)

Generates Table sr-metrics, and Figures sr-param-bar, sr-param-violin,
SR4_BarChart_ReturnTypes, and sr-lexdiv-scatter.

Reads raw method data via MetricCache (from main3.py components).
Saves outputs to: reports/article/stats_sr/ and reports/article/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import Levenshtein # pip install python-Levenshtein
from scipy.stats import iqr 

# Statistical libraries
from scipy import stats as scipy_stats 
import scikit_posthocs as sp
import pingouin as pg

# --- Import shared pipeline components from main3.py ---
try:
    from main3 import MetricCache, JSON_INPUT_DIR, BASELINE_JSON_FNAME
    METRIC_CACHE_AVAILABLE = True
    print("Successfully imported MetricCache and config from main3.py.")
except ImportError as e:
    print(f"Error importing from main3.py: {e}. SR analysis may be limited or fail.")
    MetricCache = None; METRIC_CACHE_AVAILABLE = False
    JSON_INPUT_DIR = Path("JSON"); BASELINE_JSON_FNAME = "methodless.json" 
except NameError as e_name: 
    print(f"NameError during import from main3.py: {e_name}")
    MetricCache = None; METRIC_CACHE_AVAILABLE = False
    JSON_INPUT_DIR = Path("JSON"); BASELINE_JSON_FNAME = "methodless.json"

# Configuration
REPORTS_DIR: Path = Path("reports")
ARTICLE_DIR: Path = REPORTS_DIR / "article"
STATS_SR_OUTPUT_DIR: Path = ARTICLE_DIR / "stats_sr" 
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
STATS_SR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL: str = 'model' 
RUN_COL: str = 'run'     
RAW_METHOD_NAME_COL: str = 'name' 
METHOD_SIGNATURE_COL: str = 'signature' 
VISIBILITY_JSON_COL: str = 'visibility_json' 

P_VALUE_THRESHOLD: float = 0.05
PERFECT_SCORE_THRESHOLD: float = 99.999

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


# --- Helper Functions ---
def parse_method_signature(signature_str: str, raw_method_name: str | None = None) -> dict:
    parsed = {
        'visibility_symbol': None, 'visibility_category': 'none_or_default',
        'method_name_core': None, 'parameters': [],
        'parameter_count': 0, 'return_type': 'void'
    }
    if not isinstance(signature_str, str) or pd.isna(signature_str):
        if isinstance(raw_method_name, str): 
            parsed['method_name_core'] = raw_method_name
        return parsed

    signature_str_copy = str(signature_str).strip()
    vis_match = re.match(r"^\s*([+\-#~])\s*", signature_str_copy)
    vis_map = {'+': 'public', '-': 'private', '#': 'protected', '~': 'package'}
    if vis_match:
        parsed['visibility_symbol'] = vis_match.group(1)
        parsed['visibility_category'] = vis_map.get(parsed['visibility_symbol'], 'none_or_default')
        signature_str_copy = signature_str_copy[vis_match.end():].strip()
    else: 
        for keyword, symbol in {"public": "+", "private": "-", "protected": "#"}.items():
            if signature_str_copy.lower().startswith(keyword + " "):
                parsed['visibility_symbol'] = symbol
                parsed['visibility_category'] = keyword
                signature_str_copy = signature_str_copy[len(keyword):].strip()
                break
    params_match = re.search(r"\((.*?)\)", signature_str_copy)
    name_and_maybe_return_str = signature_str_copy
    after_params_str = ""
    if params_match:
        params_content_str = params_match.group(1).strip()
        name_and_maybe_return_str = signature_str_copy[:params_match.start()].strip()
        after_params_str = signature_str_copy[params_match.end():].strip()
        if params_content_str:
            raw_params = re.split(r',\s*(?![^<]*>)', params_content_str)
            for p_str_full in raw_params:
                p_str = p_str_full.strip()
                if not p_str: continue
                parsed['parameters'].append(p_str) 
            parsed['parameter_count'] = len(parsed['parameters'])
    if after_params_str and after_params_str.startswith(':'):
        ret_type = after_params_str[1:].strip()
        if ret_type: parsed['return_type'] = ret_type
    name_parts = name_and_maybe_return_str.split()
    if name_parts:
        parsed['method_name_core'] = name_parts[-1]
        if len(name_parts) > 1 and parsed['return_type'] == 'void':
            potential_return_type = " ".join(name_parts[:-1]).strip()
            common_modifiers = {'static', 'final', 'abstract', 'public', 'private', 'protected', 'const', 'async', 'override', 'virtual'}
            if potential_return_type and not potential_return_type.lower() in common_modifiers and not re.search(r'[<>,]', potential_return_type):
                 parsed['return_type'] = potential_return_type
    if not parsed['method_name_core']: 
        parsed['method_name_core'] = raw_method_name if isinstance(raw_method_name, str) else "unknownMethod"
    if not parsed['return_type'] or parsed['return_type'].lower() == 'void':
        parsed['return_type'] = 'void'
    else:
        parsed['return_type'] = re.sub(r"^(final|static|const)\s+", "", parsed['return_type']).strip()
    return parsed

def is_camel_case(name: str) -> bool:
    if not name or not isinstance(name, str) or not name[0].islower(): return False
    # Allow digits after the first char, but not as the first char of a subsequent "word" part
    # This is a simplified check; a more robust one might involve splitting by uppercase letters.
    return bool(re.match(r"^[a-z]+[a-zA-Z0-9]*$", name))

def normalize_method_name_for_lexdiv(name_str: str) -> str: 
    if not isinstance(name_str, str) or pd.isna(name_str): return ""
    normalized = name_str.lower()
    normalized = re.sub(r'[^a-z0-9]', '', normalized) 
    return normalized

def calculate_normalized_levenshtein(s1: str, s2: str) -> float:
    if not isinstance(s1, str): s1 = "" 
    if not isinstance(s2, str): s2 = ""
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 1.0 
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 0.0
    return Levenshtein.distance(s1, s2) / max_len

def calculate_lexdiv_for_name_set(method_names_set: set[str], model_id_for_print: str = "") -> float:
    if not method_names_set or len(method_names_set) < 2: return 0.0 
    unique_names_list = sorted(list(method_names_set))
    LIMIT_UNIQUE_NAMES_FOR_LEXDIV = 750 
    if len(unique_names_list) > LIMIT_UNIQUE_NAMES_FOR_LEXDIV:
        print(f"    Note ({model_id_for_print}): Limiting LexDiv to {LIMIT_UNIQUE_NAMES_FOR_LEXDIV} sampled names from {len(unique_names_list)}.")
        unique_names_sample = np.random.choice(unique_names_list, size=LIMIT_UNIQUE_NAMES_FOR_LEXDIV, replace=False).tolist()
    else: unique_names_sample = unique_names_list
    distances = [calculate_normalized_levenshtein(n1, n2) for n1, n2 in itertools.combinations(unique_names_sample, 2)]
    return np.mean(distances) if distances else 0.0

def extract_sr_features_from_cache(cache: MetricCache) -> pd.DataFrame:
    print("Extracting SR features from MetricCache.global_details_df...")
    if cache is None or cache.global_details_df.empty: return pd.DataFrame()
    extracted_data = []
    for _, row in cache.global_details_df.iterrows():
        raw_name = row.get(RAW_METHOD_NAME_COL)
        signature = row.get(METHOD_SIGNATURE_COL)
        visibility_json = row.get(VISIBILITY_JSON_COL)
        parsed_details = parse_method_signature(str(signature) if pd.notna(signature) else "", raw_name)
        final_visibility_category = parsed_details['visibility_category']
        if isinstance(visibility_json, str) and visibility_json.strip():
            vis_json_lower = visibility_json.strip().lower()
            if vis_json_lower == '+': final_visibility_category = 'public'
            elif vis_json_lower == '-': final_visibility_category = 'private'
            elif vis_json_lower == '#': final_visibility_category = 'protected'
            elif vis_json_lower == '~': final_visibility_category = 'package'
            elif vis_json_lower in ['public', 'private', 'protected', 'package']: final_visibility_category = vis_json_lower
        
        core_name = parsed_details.get('method_name_core', raw_name if isinstance(raw_name, str) else "unknown")
        if not isinstance(core_name, str) or not core_name.strip(): core_name = raw_name if isinstance(raw_name, str) else "unknown"
        is_cc = is_camel_case(core_name)
        norm_name_lexdiv = normalize_method_name_for_lexdiv(core_name)
        extracted_data.append({
            MODEL_COL: row[MODEL_COL], RUN_COL: row[RUN_COL],
            'RawMethodName': raw_name, 'CoreMethodName': core_name,
            'NormalizedNameLexDiv': norm_name_lexdiv,
            'ParamCount': parsed_details['parameter_count'],
            'VisibilityCategory': final_visibility_category,
            'ReturnTypeStr': parsed_details['return_type'],
            'IsNonVoidReturn': (isinstance(parsed_details['return_type'], str) and parsed_details['return_type'].lower() != 'void' and parsed_details['return_type'] != ''),
            'IsCamelCase': is_cc
        })
    df_out = pd.DataFrame(extracted_data)
    print(f"SR Feature extraction complete. Shape: {df_out.shape}")
    return df_out

# --- Analysis Functions ---
def analyze_sr1_visibility(df_sr_features: pd.DataFrame, results_summary: list) -> pd.DataFrame | None:
    print("\n--- SR1: Visibility Markers ---"); results_summary.append("\n--- SR1: Visibility Markers ---")
    if df_sr_features.empty or 'VisibilityCategory' not in df_sr_features.columns: 
        results_summary.append("SR1 Warning: Data empty or 'VisibilityCategory' missing.")
        return None
    vis_counts = pd.crosstab(df_sr_features[MODEL_COL], df_sr_features['VisibilityCategory'])
    expected_cats = ['public', 'private', 'protected', 'package', 'none_or_default']
    for cat in expected_cats:
        if cat not in vis_counts.columns: vis_counts[cat] = 0
    totals = vis_counts.sum(axis=1); totals[totals == 0] = 1 
    vis_props = vis_counts.div(totals, axis=0) * 100
    table_data = pd.DataFrame(index=vis_props.index)
    table_data['Vis_+'] = vis_props.get('public', pd.Series(0.0, index=vis_props.index))
    table_data['Vis_--'] = vis_props.get('private', pd.Series(0.0, index=vis_props.index))
    table_data['Vis_#'] = vis_props.get('protected', pd.Series(0.0, index=vis_props.index))
    table_data['Vis_None'] = vis_props.get('package', pd.Series(0.0, index=vis_props.index)) + \
                             vis_props.get('none_or_default', pd.Series(0.0, index=vis_props.index))
    results_summary.append("Visibility Proportions (%):\n" + table_data.round(2).to_string())
    chi2_table_cols = [cat for cat in expected_cats if cat in vis_counts.columns] # Only use existing categories for chi2
    chi2_table = vis_counts[chi2_table_cols].copy()
    chi2_table_cleaned = chi2_table.loc[(chi2_table.sum(axis=1) > 0), (chi2_table.sum(axis=0) > 0)]
    if chi2_table_cleaned.shape[0] >= 2 and chi2_table_cleaned.shape[1] >= 2:
        try:
            chi2, p, dof, _ = scipy_stats.chi2_contingency(chi2_table_cleaned)
            results_summary.append(f"Chi-squared (Visibility): Chi2={chi2:.2f}, p={p:.4f}, df={dof}")
            if p < P_VALUE_THRESHOLD:
                n = chi2_table_cleaned.sum().sum(); r, k = chi2_table_cleaned.shape
                if n > 0 and min(r - 1, k - 1) > 0: 
                    v = np.sqrt((chi2/n) / min(r - 1, k - 1))
                    results_summary.append(f"  Cramér's V: {v:.3f}")
        except ValueError as e: results_summary.append(f"Chi2 Vis Error: {e}")
    else: results_summary.append("Chi2 Vis: Not enough data.")
    return table_data.reset_index()

def analyze_sr2_naming(df_sr_features: pd.DataFrame, results_summary: list) -> pd.DataFrame | None:
    print("\n--- SR2: Naming Conventions (camelCase) ---"); results_summary.append("\n--- SR2: Naming Conventions ---")
    if df_sr_features.empty or 'IsCamelCase' not in df_sr_features.columns: return None
    camel_case_adherence = df_sr_features.groupby(MODEL_COL)['IsCamelCase'].mean() * 100
    camel_case_df = camel_case_adherence.reset_index(name='camelCase (%)')
    results_summary.append("camelCase Adherence (%):\n" + camel_case_df.round(2).to_string())
    return camel_case_df

def analyze_sr3_parameters(df_sr_features: pd.DataFrame, results_summary: list) -> pd.DataFrame | None:
    print("\n--- SR3: Parameter Richness ---"); results_summary.append("\n--- SR3: Parameter Richness ---")
    if df_sr_features.empty or 'ParamCount' not in df_sr_features.columns:
        results_summary.append("SR3 Warning: Data empty or 'ParamCount' missing.")
        return None

    iqr_per_model = df_sr_features.groupby(MODEL_COL)['ParamCount'].apply(lambda x: iqr(x.dropna()) if len(x.dropna()) > 0 else np.nan)
    param_iqr_df = iqr_per_model.reset_index(name='IQR_ParamRichness').dropna(subset=['IQR_ParamRichness'])
    if not param_iqr_df.empty: results_summary.append("IQR Parameter Richness:\n" + param_iqr_df.round(2).to_string())
    else: results_summary.append("Warning: IQR DataFrame for parameters is empty.")

    if RUN_COL in df_sr_features.columns:
        mean_params_run = df_sr_features.groupby([MODEL_COL, RUN_COL])['ParamCount'].mean().reset_index()
        if not mean_params_run.empty:
            model_mean_agg = mean_params_run.groupby(MODEL_COL)['ParamCount'].agg(['mean', 'sem'])
            model_mean_agg.columns = ['Mean_Params_Per_Run_Avg', 'SEM_Params_Per_Run_Avg']
            
            models_present_in_agg = model_mean_agg.index.unique().tolist()
            plot_order_bar = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_agg]
            # If plot_order_bar is empty, use the order from model_mean_agg
            if not plot_order_bar: plot_order_bar = models_present_in_agg
            
            # Reindex to ensure correct order for plotting, then sort by mean for visual hierarchy
            model_mean_agg_ordered = model_mean_agg.reindex(plot_order_bar).dropna(subset=['Mean_Params_Per_Run_Avg'])
            model_mean_agg_ordered = model_mean_agg_ordered.sort_values(by='Mean_Params_Per_Run_Avg', ascending=False)
            
            if not model_mean_agg_ordered.empty:
                ordered_colors_bar = [LLM_COLORS.get(model, '#333333') for model in model_mean_agg_ordered.index]
                results_summary.append("Mean Parameter Richness (Avg of Per-Run Means):\n" + model_mean_agg_ordered.round(2).to_string())
                plt.figure(figsize=(10,6))
                plt.bar(model_mean_agg_ordered.index, model_mean_agg_ordered['Mean_Params_Per_Run_Avg'], 
                        yerr=model_mean_agg_ordered['SEM_Params_Per_Run_Avg'] * 1.96, capsize=4, color=ordered_colors_bar)
                plt.title("Mean Parameter Richness per LLM (95% CI via SEM of Per-Run Averages)", fontsize=13); 
                plt.xlabel("LLM Model", fontsize=11); plt.ylabel("Mean Parameters per Method", fontsize=11)
                plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)
                plt.grid(axis='y', alpha=0.7, linestyle='--'); plt.tight_layout()
                plt.savefig(ARTICLE_DIR / "SR_BarChart_MeanParamRichness.png"); print("Saved SR_BarChart_MeanParamRichness.png"); plt.close()
            else: print("Warning: model_mean_agg_ordered empty for bar chart.")
        else: results_summary.append("Warning: mean_params_per_run was empty for bar chart.")
    else: results_summary.append(f"Warning: '{RUN_COL}' column missing for bar chart.")

    if 'ParamCount' in df_sr_features.columns and df_sr_features['ParamCount'].notna().any():
        plt.figure(figsize=(12,7))
        models_present_violin = df_sr_features[MODEL_COL].unique().tolist()
        plot_order_violin = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_violin]
        if not plot_order_violin : plot_order_violin = models_present_violin # Fallback if order list doesn't match data
        
        plot_palette_violin = {model: LLM_COLORS.get(model, '#333333') for model in plot_order_violin}

        sns.violinplot(x=MODEL_COL, y='ParamCount', data=df_sr_features, 
                       order=plot_order_violin, palette=plot_palette_violin, 
                       inner="quartile", cut=0, scale="width")
        plt.title("Parameter Count Distribution per Method and Model", fontsize=14); 
        plt.xlabel("LLM Model", fontsize=12); plt.ylabel("Number of Parameters per Method", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6); plt.tight_layout()
        plt.savefig(ARTICLE_DIR / "SR_Violin_ParamCountDistribution.png"); print("Saved SR_Violin_ParamCountDistribution.png"); plt.close()

    results_summary.append("\nKruskal-Wallis (Parameter Counts):")
    df_kw_params = df_sr_features.dropna(subset=['ParamCount'])
    valid_groups_kw_params = df_kw_params.groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
    if not valid_groups_kw_params.empty and valid_groups_kw_params[MODEL_COL].nunique() >= 2:
        kw_params = pg.kruskal(data=valid_groups_kw_params, dv='ParamCount', between=MODEL_COL)
        if kw_params is not None and not kw_params.empty:
            results_summary.append(kw_params.round(4).to_string()); print(kw_params.round(4))
            if kw_params['p-unc'].iloc[0] < P_VALUE_THRESHOLD:
                if 'eps-sq' in kw_params.columns and pd.notna(kw_params['eps-sq'].iloc[0]): 
                    results_summary.append(f"  Effect Size (Epsilon-squared): {kw_params['eps-sq'].iloc[0]:.3f}"); print(f"  Effect Size (Epsilon-squared): {kw_params['eps-sq'].iloc[0]:.3f}")
                dunn_params = sp.posthoc_dunn(valid_groups_kw_params, val_col='ParamCount', group_col=MODEL_COL, p_adjust='bonferroni')
                results_summary.append("  Dunn's Posthoc (Parameter Counts):\n" + dunn_params.round(4).to_string()); print(dunn_params.round(4))
    return param_iqr_df

def analyze_sr4_return_types(df_sr_features: pd.DataFrame, results_summary: list) -> pd.DataFrame | None:
    print("\n--- SR4: Return Types ---"); results_summary.append("\n--- SR4: Return Types ---")
    if df_sr_features.empty or 'IsNonVoidReturn' not in df_sr_features.columns: return None
    ret_type_percent = df_sr_features.groupby(MODEL_COL)['IsNonVoidReturn'].mean() * 100
    ret_type_df = ret_type_percent.reset_index(name='Ret. (%)')
    results_summary.append("Non-void Return Type (%):\n" + ret_type_df.round(2).to_string())
    print("Non-void Return Type (%):\n", ret_type_df.round(2))
    
    models_present_in_data = ret_type_df[MODEL_COL].unique().tolist()
    plot_order_ret = [model for model in LLM_ORDER_FOR_PLOTS if model in models_present_in_data]
    if not plot_order_ret: plot_order_ret = models_present_in_data # Fallback
    
    plot_df_ret = ret_type_df.set_index(MODEL_COL).reindex(plot_order_ret).reset_index() # Order for plot
    plot_df_ret = plot_df_ret.sort_values(by='Ret. (%)', ascending=False) # Then sort by value for bar height
    ordered_colors_ret = [LLM_COLORS.get(model, '#333333') for model in plot_df_ret[MODEL_COL]]


    plt.figure(figsize=(10,6))
    plt.bar(plot_df_ret[MODEL_COL], plot_df_ret['Ret. (%)'], color=ordered_colors_ret)
    plt.title("Proportion of Non-Void Return Types per Model", fontsize=13); plt.xlabel("LLM Model", fontsize=11); plt.ylabel("Non-Void Return Types (%)", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(np.arange(0, 101, 10)); plt.ylim(0,105); plt.grid(axis='y', alpha=0.7, linestyle='--'); plt.tight_layout()
    plt.savefig(ARTICLE_DIR / "SR4_BarChart_ReturnTypes.png"); print("Saved SR4_BarChart_ReturnTypes.png"); plt.close()
    
    contingency_ret = pd.crosstab(df_sr_features[MODEL_COL], df_sr_features['IsNonVoidReturn'])
    if True not in contingency_ret.columns: contingency_ret[True] = 0
    if False not in contingency_ret.columns: contingency_ret[False] = 0
    contingency_ret = contingency_ret.rename(columns={True: 'NonVoid', False: 'Void'})
    contingency_ret_cleaned = contingency_ret.loc[(contingency_ret.sum(axis=1) > 0), (contingency_ret.sum(axis=0) > 0)]
    if contingency_ret_cleaned.shape[0] >= 2 and contingency_ret_cleaned.shape[1] >= 2:
        try:
            chi2,p,dof,exp = scipy_stats.chi2_contingency(contingency_ret_cleaned)
            results_summary.append(f"Chi-squared (Return Types): Chi2={chi2:.2f}, p={p:.4f}, df={dof}")
            print(f"Chi-squared (Return Types): Chi2={chi2:.2f}, p={p:.4f}, df={dof}")
            if p < P_VALUE_THRESHOLD:
                n = contingency_ret_cleaned.sum().sum(); r, k = contingency_ret_cleaned.shape
                if n > 0 and min(r-1, k-1) > 0: v = np.sqrt((chi2/n) / min(r-1, k-1)); results_summary.append(f"  Cramér's V: {v:.3f}"); print(f"  Cramér's V: {v:.3f}")
        except ValueError as e: results_summary.append(f"Chi2 ReturnType Error: {e}")
    else: results_summary.append("Chi2 ReturnType: Not enough data.")
    return ret_type_df

def analyze_sr5_lexical_coherence(df_sr_features: pd.DataFrame, results_summary: list) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    print("\n--- SR5: Lexical Coherence ---"); results_summary.append("\n--- SR5: Lexical Coherence ---")
    if df_sr_features.empty or 'NormalizedNameLexDiv' not in df_sr_features.columns or 'RawMethodName' not in df_sr_features.columns:
        results_summary.append("SR5 Warning: Missing columns for Lexical Coherence.")
        return None, None

    lexdiv_per_run_data = []
    for (model, run), group in df_sr_features.groupby([MODEL_COL, RUN_COL]):
        unique_norm_names = set(group['NormalizedNameLexDiv'].dropna())
        lexdiv_run = calculate_lexdiv_for_name_set(unique_norm_names, f"{model}-{run}")
        lexdiv_per_run_data.append({MODEL_COL: model, RUN_COL: run, 'LexDiv_Per_Run': lexdiv_run})
    lexdiv_per_run_df = pd.DataFrame(lexdiv_per_run_data)
    
    agg_lexdiv_df = lexdiv_per_run_df.groupby(MODEL_COL)['LexDiv_Per_Run'].mean().reset_index().rename(columns={'LexDiv_Per_Run':'LexDiv'})
    results_summary.append("Lexical Diversity (Agg. Mean Per-Run LexDiv):\n" + agg_lexdiv_df.round(3).to_string())
    print("Lexical Diversity (Agg. Mean Per-Run LexDiv):\n", agg_lexdiv_df.round(3))

    redundancy_data = []
    for model, group in df_sr_features.groupby(MODEL_COL):
        total_methods = len(group)
        unique_norm_names_model = group['NormalizedNameLexDiv'].nunique()
        redundancy = total_methods / unique_norm_names_model if unique_norm_names_model > 0 else np.nan
        redundancy_data.append({MODEL_COL: model, 'Redundancy_Ratio': redundancy})
    redundancy_df = pd.DataFrame(redundancy_data)
    results_summary.append("Redundancy Ratio:\n" + redundancy_df.round(2).to_string())
    print("Redundancy Ratio:\n", redundancy_df.round(2))
    
    if not agg_lexdiv_df.empty and not redundancy_df.empty:
        plot_df_lex = pd.merge(agg_lexdiv_df, redundancy_df, on=MODEL_COL).dropna()
        if not plot_df_lex.empty:
            plt.figure(figsize=(10,8)); 
            num_models_plot = plot_df_lex[MODEL_COL].nunique()
            markers_list_plot = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
            
            # Ensure plot_order_lex contains only models present in plot_df_lex
            models_present_lex = plot_df_lex[MODEL_COL].unique()
            plot_order_lex = [m for m in LLM_ORDER_FOR_PLOTS if m in models_present_lex]
            if not plot_order_lex: plot_order_lex = models_present_lex # Fallback
            
            # Create palette based on the actual models to be plotted and their order
            palette_lex = {model: LLM_COLORS.get(model, '#333333') for model in plot_order_lex}

            sns.scatterplot(x='Redundancy_Ratio', y='LexDiv', hue=MODEL_COL, style=MODEL_COL, 
                            hue_order=plot_order_lex, style_order=plot_order_lex, # Use order here
                            markers=[markers_list_plot[i % len(markers_list_plot)] for i in range(num_models_plot)],
                            data=plot_df_lex, s=120, palette=palette_lex, legend=False) # legend=False as per text
            for _, row in plot_df_lex.iterrows(): # Annotate all points
                plt.text(row['Redundancy_Ratio'] + 0.01 * plot_df_lex['Redundancy_Ratio'].max(skipna=True), 
                         row['LexDiv'], row[MODEL_COL], fontsize=9)
            plt.xlabel("Redundancy (Methods per Unique Normalized Name)", fontsize=11); 
            plt.ylabel("Lexical Diversity (Mean Norm. Levenshtein)", fontsize=11)
            plt.title("Lexical Diversity vs. Redundancy per LLM", fontsize=14); plt.grid(True, alpha=0.7)
            # Y-axis ticks as per specification: 1.0, 1.05, 1.10 for Redundancy
            # This means Redundancy is on Y-axis in the paper's figure?
            # Text: "Lexical diversity (y-axis...) against method-name redundancy (x-axis...)"
            # So X=Redundancy, Y=LexDiv. The code is correct for this.
            # The paper's y-ticks [1.0, 1.05, 1.10] seem too small for redundancy values in table (1.77-6.00)
            # Let's adjust based on data or keep automatic for now.
            # plt.yticks([0.76, 0.78, 0.80, 0.82, 0.84]); # For LexDiv
            plt.tight_layout()
            plt.savefig(ARTICLE_DIR / "SR_PointCloud_LexDiv_vs_Redundancy.png"); print("Saved SR_PointCloud_LexDiv_vs_Redundancy.png"); plt.close()

    if not lexdiv_per_run_df.empty and lexdiv_per_run_df[MODEL_COL].nunique() >= 2:
        valid_groups_kw_lexdiv = lexdiv_per_run_df.dropna(subset=['LexDiv_Per_Run']).groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
        if not valid_groups_kw_lexdiv.empty and valid_groups_kw_lexdiv[MODEL_COL].nunique() >=2:
            kw_lexdiv = pg.kruskal(data=valid_groups_kw_lexdiv, dv='LexDiv_Per_Run', between=MODEL_COL)
            if kw_lexdiv is not None and not kw_lexdiv.empty:
                results_summary.append("\nK-W (Per-Run LexDiv):\n" + kw_lexdiv.round(4).to_string()); print(kw_lexdiv.round(4))
                if kw_lexdiv['p-unc'].iloc[0] < P_VALUE_THRESHOLD:
                    if 'eps-sq' in kw_lexdiv.columns and pd.notna(kw_lexdiv['eps-sq'].iloc[0]): results_summary.append(f"  Effect Size (ε²): {kw_lexdiv['eps-sq'].iloc[0]:.3f}")
                    dunn_lexdiv = sp.posthoc_dunn(valid_groups_kw_lexdiv, val_col='LexDiv_Per_Run', group_col=MODEL_COL, p_adjust='bonferroni')
                    results_summary.append("  Dunn's (LexDiv):\n" + dunn_lexdiv.round(4).to_string()); print(dunn_lexdiv.round(4))
    
    per_run_redundancy_data = []
    for (model, run), group in df_sr_features.groupby([MODEL_COL, RUN_COL]): # Using df_sr_features to get total raw method count per run
        total_methods_run = len(group) 
        normalized_names_in_run = group['NormalizedNameLexDiv'][group['NormalizedNameLexDiv'] != ''].unique()
        unique_norm_names_run_count = len(normalized_names_in_run)
        redundancy_run = total_methods_run / unique_norm_names_run_count if unique_norm_names_run_count > 0 else np.nan
        per_run_redundancy_data.append({MODEL_COL: model, RUN_COL: run, 'Redundancy_Per_Run': redundancy_run})
    per_run_redundancy_df = pd.DataFrame(per_run_redundancy_data)
    if not per_run_redundancy_df.empty and per_run_redundancy_df[MODEL_COL].nunique() >= 2:
        valid_groups_kw_red = per_run_redundancy_df.dropna(subset=['Redundancy_Per_Run']).groupby(MODEL_COL).filter(lambda x: len(x) >= 2)
        if not valid_groups_kw_red.empty and valid_groups_kw_red[MODEL_COL].nunique() >=2:
            kw_red = pg.kruskal(data=valid_groups_kw_red, dv='Redundancy_Per_Run', between=MODEL_COL)
            if kw_red is not None and not kw_red.empty:
                results_summary.append("\nK-W (Per-Run Redundancy):\n" + kw_red.round(4).to_string()); print(kw_red.round(4))
                if kw_red['p-unc'].iloc[0] < P_VALUE_THRESHOLD:
                    if 'eps-sq' in kw_red.columns and pd.notna(kw_red['eps-sq'].iloc[0]): results_summary.append(f"  Effect Size (ε²): {kw_red['eps-sq'].iloc[0]:.3f}")
                    dunn_red = sp.posthoc_dunn(valid_groups_kw_red, val_col='Redundancy_Per_Run', group_col=MODEL_COL, p_adjust='bonferroni')
                    results_summary.append("  Dunn's (Redundancy):\n" + dunn_red.round(4).to_string()); print(dunn_red.round(4))
    return agg_lexdiv_df, redundancy_df


def main():
    print("--- Starting Signature Richness (SR) Analysis ---")
    statistical_results_all_sr = ["--- Signature Richness (SR) Statistical Analysis ---"] 

    # ... (MetricCache and df_sr_features setup as before) ...
    if not METRIC_CACHE_AVAILABLE: # ... (return if fail)
        # ...
        return
    try:
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names_list=[])
    except Exception as e: # ... (return if fail)
        # ...
        return
    if cache.global_details_df.empty: # ... (return if fail)
        # ...
        return
    df_sr_features = extract_sr_features_from_cache(cache)
    if df_sr_features.empty: # ... (return if fail)
        # ...
        return

    # Run analyses for each SR component
    visibility_table_data = analyze_sr1_visibility(df_sr_features, statistical_results_all_sr)
    camelcase_table_data = analyze_sr2_naming(df_sr_features, statistical_results_all_sr)
    param_iqr_table_data = analyze_sr3_parameters(df_sr_features, statistical_results_all_sr) 
    return_type_table_data = analyze_sr4_return_types(df_sr_features, statistical_results_all_sr)
    agg_lexdiv_df, agg_redundancy_df = analyze_sr5_lexical_coherence(df_sr_features, statistical_results_all_sr) 
    
    print("\nAssembling Table sr-metrics...")
    # Ensure MODEL_COL ('model') is used consistently for merging
    unique_models = sorted(df_sr_features[MODEL_COL].unique())
    final_sr_table = pd.DataFrame({MODEL_COL: unique_models}) # Column is named 'model'

    if param_iqr_table_data is not None and not param_iqr_table_data.empty:
        # param_iqr_table_data already has 'model' column from reset_index
        final_sr_table = pd.merge(final_sr_table, param_iqr_table_data, on=MODEL_COL, how='left')
    
    if return_type_table_data is not None and not return_type_table_data.empty:
        # return_type_table_data already has 'model' column
        final_sr_table = pd.merge(final_sr_table, return_type_table_data, on=MODEL_COL, how='left')
    
    if agg_lexdiv_df is not None and not agg_lexdiv_df.empty:
        # agg_lexdiv_df already has 'model' column
        final_sr_table = pd.merge(final_sr_table, agg_lexdiv_df, on=MODEL_COL, how='left')
    
    if visibility_table_data is not None and not visibility_table_data.empty:
        # visibility_table_data already has 'model' column
        vis_for_main_table_temp = visibility_table_data.rename(columns={'Vis_+': '+'}) # Rename before merge
        
        vis_for_main_table_temp['-- / # / None'] = vis_for_main_table_temp.apply(
            lambda r: f"{r.get('Vis_--', 0):.2f} / {r.get('Vis_#', 0):.2f} / {r.get('Vis_None', 0):.2f}", axis=1
        )
        # Select only the needed columns for this specific merge part
        columns_to_merge_vis = [MODEL_COL, '+', '-- / # / None']
        # Ensure all these columns exist in vis_for_main_table_temp
        existing_cols_vis = [col for col in columns_to_merge_vis if col in vis_for_main_table_temp.columns]

        final_sr_table = pd.merge(final_sr_table, vis_for_main_table_temp[existing_cols_vis], on=MODEL_COL, how='left')

    if camelcase_table_data is not None and not camelcase_table_data.empty:
         final_sr_table = pd.merge(final_sr_table, camelcase_table_data, on=MODEL_COL, how='left')

    # Now, after all merges are done using MODEL_COL ('model'), rename 'model' to 'Model' for display
    final_sr_table.rename(columns={MODEL_COL: 'Model'}, inplace=True)
    MODEL_COL_DISPLAY = 'Model' # Use this for final selection

    desired_cols_sr_table = [MODEL_COL_DISPLAY, 'IQR_ParamRichness', 'Ret. (%)', 'LexDiv', '+', '-- / # / None'] 
    if 'camelCase (%)' in final_sr_table.columns:
        desired_cols_sr_table.append('camelCase (%)')
        
    final_sr_table_cols_present = [col for col in desired_cols_sr_table if col in final_sr_table.columns]
    final_sr_table_for_display = final_sr_table[final_sr_table_cols_present].copy()

    print("\nFinal Assembled SR Metrics Table (for Table sr-metrics):\n", final_sr_table_for_display) # .round(2) removed for mixed types
    final_sr_table_for_display.to_csv(STATS_SR_OUTPUT_DIR / "SR_Metrics_For_Table_sr-metrics.csv", index=False, float_format='%.2f')
    statistical_results_all_sr.append("\nFinal SR Metrics Table (for Table sr-metrics):\n" + final_sr_table_for_display.to_string(index=False)) # removed .round(2)

    summary_file_path = STATS_SR_OUTPUT_DIR / "SR_Analysis_Overall_Summary.txt"
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(statistical_results_all_sr))
    print(f"\nSaved overall SR statistical summary to {summary_file_path}")
    print("\n--- Signature Richness (SR) Analysis Finished ---")

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError: print("Warning: Seaborn style not found.")
    main()