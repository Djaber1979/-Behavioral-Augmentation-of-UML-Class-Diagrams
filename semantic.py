#!/usr/bin/env python3
"""
semantic.py – Computes Behavioral Correctness (BC) sensitivity at thresholds 0.5,0.6,0.7
Analyzes ALL methods, omitting class name based filtering for method selection.
Counts unique gold actions covered per model for coverage percentage.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
# from sentence_transformers import SentenceTransformer # Not directly used if SENTENCE_MODEL is imported
from sentence_transformers.util import cos_sim

# --- Import shared pipeline components ---
from main3 import (
    MetricCache,
    JSON_INPUT_DIR,
    BASELINE_JSON_FNAME,
    NLP_MODEL_NAME,
    DEVICE,
    SENTENCE_MODEL,
    generate_method_annotation_report, # Assumes this function in main3.py is updated
                                       # to handle empty class_names by returning all methods.
    split_method_name
)

# --- Configuration & constants ---
SEMANTIC_SIMILARITY_THRESHOLDS = [0.5, 0.6, 0.7]

# --- Text‐preparation helper (semantic.py specific: no class name in embedding input) ---
def prepare_method_embedding_text_semantic(signature: str, uc_action: str) -> str:
    """
    Given a signature and an //action: text, strip visibility and types,
    split camelCase/underscores, and append the action description words.
    This version is specific to semantic.py. Class name is NOT used.
    """
    sig = re.sub(r'^(public|private|protected)\s+', '', signature or "", flags=re.IGNORECASE)
    
    # Attempt to remove return type: split into first word and the rest.
    # If first word doesn't look like part of a method call (e.g., no '('), assume it's a type.
    sig_parts = sig.split(None, 1)
    processed_sig = sig # Default to original if split fails or not applicable
    if len(sig_parts) > 1: # If there's more than one "word"
        first_word, rest_of_sig = sig_parts
        if '(' not in first_word: # Heuristic: if first word is not method name itself
            processed_sig = rest_of_sig
        else: # First word is likely the method name
            processed_sig = sig 
    elif len(sig_parts) == 1: # Only one word
        processed_sig = sig_parts[0]
    else: # Empty signature
        processed_sig = ""
        
    processed_sig = re.sub(r'\(.*\)', '', processed_sig) # Drop parameter list
    
    parts = split_method_name(processed_sig).split()
    action_tokens = split_method_name(uc_action or "").split()
    return " ".join(parts + action_tokens).strip()

# --- Core mapping function (semantic.py specific version, optimized) ---
def map_methods_to_actions_semantic(annotation_df: pd.DataFrame, gold_data: dict) -> pd.DataFrame:
    """
    Embed each generated method (signature + action annotation) and match to
    the gold‐standard actions (embedding of ideal_method + parameters + return + action).
    Uses semantic.py's prepare_method_embedding_text_semantic.
    """
    if annotation_df.empty:
        out_df_empty = annotation_df.copy()
        # Ensure expected columns are present even if input is empty
        for col in ['Best_Match_Action', 'SimilarityScore']:
            if col not in out_df_empty:
                 out_df_empty[col] = pd.Series(dtype='object' if col == 'Best_Match_Action' else 'float')
        return out_df_empty

    gen_texts = [
        prepare_method_embedding_text_semantic(row.get('Signature', ""), row.get('UC_Action', ""))
        for _, row in annotation_df.iterrows()
    ]
    if not gen_texts: # Can happen if all rows resulted in empty text
        out_df_empty = annotation_df.copy()
        out_df_empty['Best_Match_Action'] = pd.Series(dtype='object', index=annotation_df.index)
        out_df_empty['SimilarityScore'] = pd.Series(dtype='float', index=annotation_df.index)
        return out_df_empty

    emb_gen = SENTENCE_MODEL.encode(gen_texts, convert_to_tensor=True, device=DEVICE)

    gold_actions, gold_texts = [], []
    if not gold_data or 'action_to_details' not in gold_data:
        print("Warning: gold_data is missing or doesn't have 'action_to_details'. Mapping will be incomplete.")
        out_df_empty = annotation_df.copy()
        out_df_empty['Best_Match_Action'] = pd.Series(dtype='object', index=annotation_df.index)
        out_df_empty['SimilarityScore'] = np.nan
        return out_df_empty
            
    for act, details in gold_data['action_to_details'].items():
        ideal  = details.get('ideal_method', '')
        params = details.get('expected_parameter_concepts', [])
        ret    = details.get('expected_return_concept', '')
        
        txt_parts = []
        txt_parts.extend(split_method_name(ideal).split())
        for p in params:
            txt_parts.extend(split_method_name(p).split())
        txt_parts.extend(split_method_name(ret).split())
        txt_parts.extend(split_method_name(act).split())
        
        gold_actions.append(act)
        gold_texts.append(" ".join(filter(None,txt_parts)))

    if not gold_texts:
        print("Warning: No gold standard texts generated for comparison.")
        out_df_empty = annotation_df.copy()
        out_df_empty['Best_Match_Action'] = pd.Series(dtype='object', index=annotation_df.index)
        out_df_empty['SimilarityScore'] = np.nan
        return out_df_empty

    emb_gold = SENTENCE_MODEL.encode(gold_texts, convert_to_tensor=True, device=DEVICE)
    sims = cos_sim(emb_gen, emb_gold).clamp(-1.0, 1.0)

    if sims.numel() == 0: # Should only happen if emb_gen or emb_gold is empty
        print("Warning: Cosine similarity matrix is empty.")
        out_df_empty = annotation_df.copy()
        out_df_empty['Best_Match_Action'] = pd.Series(dtype='object', index=annotation_df.index)
        out_df_empty['SimilarityScore'] = np.nan
        return out_df_empty

    best_match_indices_tensor = torch.argmax(sims, dim=1)
    best_match_indices_list = best_match_indices_tensor.cpu().tolist()
    
    if sims.ndim == 1:
        # If sims is 1D, best_match_indices_tensor will be a 0D tensor (scalar)
        # or 1D if sims came from comparing a vector to a scalar (not typical here)
        if best_match_indices_tensor.ndim == 0:
            scores_tensor = sims[best_match_indices_tensor]
        else: # Should not be hit if sims is 1D from cos_sim(vector, matrix)
            scores_tensor = sims.gather(dim=0, index=best_match_indices_tensor)

    elif sims.ndim == 2:
        scores_tensor = sims[torch.arange(sims.shape[0], device=sims.device), best_match_indices_tensor]
    else:
        print(f"Warning: Unexpected sims dimensions: {sims.ndim}")
        scores_tensor = torch.tensor([], device=sims.device, dtype=torch.float)


    best = [gold_actions[idx] for idx in best_match_indices_list]
    scores = scores_tensor.cpu().tolist()

    out = annotation_df.copy()
    out['Best_Match_Action'] = pd.Series(best if best else [], index=out.index, dtype='object')
    out['SimilarityScore']   = pd.Series(scores if scores else [], index=out.index, dtype='float')
    return out

# --- Main: compute coverage at multiple thresholds ---
def main():
    print(f"semantic.py: Using NLP model {NLP_MODEL_NAME} on {DEVICE}")
    cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, class_names=[])
    
    print("semantic.py: Generating annotation report for ALL methods...")
    ann_df = generate_method_annotation_report(cache) 
    
    if ann_df.empty:
        print("Annotation DataFrame is empty (no methods found or processed). No methods to map. Exiting.")
        pd.DataFrame(columns=['Model','Threshold','Sem_Covered_Action_Count','Sem_Action_Coverage_Percent']).to_csv("bc_sensitivity_summary.csv", index=False)
        print("Wrote empty bc_sensitivity_summary.csv")
        return

    print(f"semantic.py: Mapping {len(ann_df)} methods to actions...")
    map_df = map_methods_to_actions_semantic(ann_df, cache.gold)

    if map_df.empty or 'SimilarityScore' not in map_df.columns:
        print("Mapping DataFrame is empty or missing 'SimilarityScore'. Cannot compute sensitivity. Exiting.")
        pd.DataFrame(columns=['Model','Threshold','Sem_Covered_Action_Count','Sem_Action_Coverage_Percent']).to_csv("bc_sensitivity_summary.csv", index=False)
        print("Wrote empty bc_sensitivity_summary.csv")
        return

    results = []
    total_gold_actions_count = cache.gold.get('total_actions', 0)
    if total_gold_actions_count == 0: # Fallback if 'total_actions' key was missing or zero
        total_gold_actions_count = len(cache.gold.get('all_actions', []))
        if total_gold_actions_count == 0:
             print("Critical Warning: Total gold actions count is 0 even after fallback. Coverage percentages will be 0 or NaN.")


    print(f"semantic.py: Calculating sensitivity across thresholds. Total unique gold actions: {total_gold_actions_count}")
    for thr in SEMANTIC_SIMILARITY_THRESHOLDS:
        current_map_df = map_df.copy()
        current_map_df['Match'] = current_map_df['SimilarityScore'] >= thr
        
        if 'Model' not in current_map_df.columns:
            print(f"Warning: 'Model' column not found in mapping dataframe for threshold {thr}. Skipping.")
            continue
        if 'Best_Match_Action' not in current_map_df.columns:
            print(f"Warning: 'Best_Match_Action' column not found in mapping dataframe for threshold {thr}. Skipping.")
            continue

        # For each model, get the set of unique 'Best_Match_Action's where 'Match' is True
        def count_unique_matched_gold_actions(group_df):
            return group_df[group_df['Match']]['Best_Match_Action'].nunique()

        agg_df = (
            current_map_df
            .groupby('Model', observed=True)
            .apply(count_unique_matched_gold_actions)
            .reset_index(name='Sem_Covered_Action_Count') 
        )
        
        agg_df['Total_Gold_Actions'] = total_gold_actions_count # Add this for reference, though not strictly needed for percentage

        if total_gold_actions_count > 0:
            agg_df['Sem_Action_Coverage_Percent'] = (agg_df['Sem_Covered_Action_Count'] / total_gold_actions_count) * 100.0
        else:
            agg_df['Sem_Action_Coverage_Percent'] = 0.0 if not pd.isna(total_gold_actions_count) else np.nan


        agg_df['Threshold'] = thr
        results.append(agg_df[['Model','Threshold','Sem_Covered_Action_Count','Sem_Action_Coverage_Percent']])

    if not results:
        print("No results generated for BC sensitivity. Writing empty CSV.")
        out_df = pd.DataFrame(columns=['Model','Threshold','Sem_Covered_Action_Count','Sem_Action_Coverage_Percent'])
    else:
        out_df = pd.concat(results, ignore_index=True)

    out_df.to_csv("bc_sensitivity_summary.csv", index=False)
    print(f"Wrote bc_sensitivity_summary.csv with {len(out_df)} rows.")

if __name__ == "__main__":
    main()