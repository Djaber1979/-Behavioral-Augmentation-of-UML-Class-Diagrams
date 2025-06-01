#!/usr/bin/env python3
"""
main3.py - Consolidated LLM‐enhanced class diagram metrics pipeline (optimized)
Produces "master" CSVs including BC Sensitivity (Per-Run Averaged, Gold Action Coverage).
All calculations are performed on ALL methods if CLASS_NAMES is empty.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# --- Configuration ---
JSON_INPUT_DIR = Path("JSON")
BASELINE_JSON_FNAME = "methodless.json"
GOLD_STANDARD_MAP_FNAME = "uc_action_method_map.json"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
TOP_N_CORE = 38
CLASS_NAMES = [] # IMPORTANT: Empty list means ALL classes will be processed by default for most metrics.

SEMANTIC_SIMILARITY_THRESHOLD = 0.5 # General threshold for some main coverage metrics
NLP_MODEL_NAME = 'all-mpnet-base-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STANDARD_TYPES = {"float", "boolean", "integer", "int", "string", "double", "long", "char", "byte", "short"}

SENTENCE_MODEL = SentenceTransformer(NLP_MODEL_NAME).to(DEVICE)

for seed in (17, 42, 123):
    torch.manual_seed(seed)
    np.random.seed(seed)

# --- Utilities ---
def split_method_name(name: str) -> str:
    if not name: return ""
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    s2 = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', s1)
    return ' '.join(s2.replace('_', ' ').lower().split())

def safe_divide(n, d, default=np.nan):
    if isinstance(d, pd.Series):
        mask_invalid_d = (d == 0) | d.isnull()
        if isinstance(n, pd.Series):
            result = pd.Series(default, index=n.index.union(d.index), dtype=float)
            n_aligned, d_aligned = n.align(d, join='left', copy=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                division_result = n_aligned / d_aligned
            result[~mask_invalid_d] = division_result[~mask_invalid_d]
        else: # n is a scalar
            result = pd.Series(default, index=d.index, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                division_result = n / d
            result[~mask_invalid_d] = division_result[~mask_invalid_d]
        return result
    else: # Scalar case
        if d == 0 or d is None or (isinstance(d, float) and math.isnan(d)): return default
        try: r = n / d
        except ZeroDivisionError: return default
        except TypeError: return default
        return default if isinstance(r, float) and math.isnan(r) else r


def parse_classname_with_stereotype(class_name_full: str) -> tuple[str, str | None]:
    if not isinstance(class_name_full, str): class_name_full = str(class_name_full)
    match = re.fullmatch(r'(.*?)\s*<<\s*(.*?)\s*>>\s*', class_name_full)
    if match:
        main_name = match.group(1).strip()
        stereotype_name = match.group(2).strip()
        return main_name if main_name else "", stereotype_name if stereotype_name else None
    return class_name_full.strip(), None

def sanitize_classname_for_filename(class_name_full: str, default_filename_part: str = "class_data") -> str:
    main_name, stereotype_name = parse_classname_with_stereotype(class_name_full)
    parts_for_filename = []
    if stereotype_name: parts_for_filename.append(stereotype_name)
    if main_name: parts_for_filename.append(main_name)
    base_name_str = " ".join(filter(None, parts_for_filename)) if parts_for_filename else default_filename_part
    sane_name = base_name_str.lower()
    invalid_os_chars_pattern = r'[<>:"/\\|?*\x00-\x1F]'
    sane_name = re.sub(invalid_os_chars_pattern, '_', sane_name)
    sane_name = re.sub(r'\s+', '_', sane_name)
    sane_name = re.sub(r'[^\w_]+', '_', sane_name)
    sane_name = re.sub(r'_+', '_', sane_name)
    sane_name = sane_name.strip('_')
    if not sane_name:
        if stereotype_name:
            temp_stereotype_sane = stereotype_name.lower()
            temp_stereotype_sane = re.sub(invalid_os_chars_pattern, '_', temp_stereotype_sane)
            temp_stereotype_sane = re.sub(r'\s+', '_', temp_stereotype_sane)
            temp_stereotype_sane = re.sub(r'[^\w_]+', '_', temp_stereotype_sane)
            temp_stereotype_sane = re.sub(r'_+', '_', temp_stereotype_sane).strip('_')
            if temp_stereotype_sane: return temp_stereotype_sane
        return default_filename_part
    return sane_name

# --- JSON Structure Extraction & Comparison ---
def extract_structure_from_json(json_data: dict) -> dict[str, set]:
    elems = {k: set() for k in ['packages', 'enums', 'enum_values', 'classes', 'attributes', 'relationships']}
    if not isinstance(json_data, dict): return elems
    for pkg in json_data.get('packages', []):
        if isinstance(pkg, str): elems['packages'].add(pkg)
    for enum in json_data.get('enums', []):
        name = enum.get('name')
        if name:
            elems['enums'].add(name)
            for v in enum.get('values', []):
                if isinstance(v, str): elems['enum_values'].add(f"{name}::{v}")
    for cls in json_data.get('classes', []):
        cname = cls.get('name')
        if cname:
            elems['classes'].add(cname)
            for attr in cls.get('attributes', []):
                an, at = attr.get('name'), attr.get('type')
                if an and at is not None: elems['attributes'].add(f"{cname}::{an}: {at}")
    for rel in json_data.get('relationships', []):
        s, t = rel.get('source'), rel.get('target')
        if s and t:
            sym, lbl = rel.get('type_symbol', '--'), rel.get('label', '')
            sc, tc = rel.get('source_cardinality', ''), rel.get('target_cardinality', '')
            elems['relationships'].add(f"{s}{' '+sc if sc else ''} {sym}{' '+tc if tc else ''} {t}{' : '+lbl if lbl else ''}")
    return elems

def compare_structures(base: dict[str, set], enriched: dict[str, set]) -> dict:
    report, total_base, preserved = {}, 0, 0
    keys = sorted(set(base) | set(enriched))
    for k in keys:
        b, e = base.get(k, set()), enriched.get(k, set())
        p, m, a = len(b & e), len(b - e), len(e - b)
        report[f"{k}_preserved"], report[f"{k}_missing"], report[f"{k}_added"], report[f"{k}_total_baseline"] = p, m, a, len(b)
        total_base += len(b); preserved += p
    report['Overall_Preservation_%'] = round(safe_divide(preserved * 100.0, total_base, 0.0), 2) if total_base else 100.0
    report['Total_Baseline_Elements'], report['Total_Preserved_Elements'] = total_base, preserved
    report['Total_Added_Elements'] = sum(len(enriched.get(k, set()) - base.get(k, set())) for k in enriched)
    return report

# --- Gold Map Preprocessing ---
def _preprocess_gold_map(gold_map_data: dict) -> dict:
    d = {'all_uc_ids': set(), 'all_actions': set(), 'uc_to_actions': defaultdict(set), 'action_to_details': {},
         'class_to_uc_ids': defaultdict(set), 'uc_action_counts': Counter()}
    for uc_id, entries in gold_map_data.items():
        uc, cnt = str(uc_id), 0
        d['all_uc_ids'].add(uc)
        if isinstance(entries, list):
            for e in entries:
                act, cls, ideal = e.get('action'), e.get('assigned_class'), e.get('ideal_method', '')
                params, ret = e.get('expected_parameter_concepts', []), e.get('expected_return_concept', '')
                if act and cls:
                    d['all_actions'].add(act); d['uc_to_actions'][uc].add(act)
                    d['action_to_details'][act] = {'uc_id': uc, 'assigned_class': cls, 'ideal_method': ideal,
                                                  'expected_parameter_concepts': params, 'expected_return_concept': ret}
                    d['class_to_uc_ids'][cls].add(uc); cnt += 1
        d['uc_action_counts'][uc] = cnt
    d['total_actions'], d['total_ucs'] = len(d['all_actions']), len(d['all_uc_ids'])
    return d

# --- Flexible Text Preparation ---
def prepare_method_embedding_text(signature: str, class_name: str, action_annotation: str = "",
                                  include_class_name: bool = True, include_param_names: bool = True,
                                  include_return_type: bool = True) -> str:
    sig_clean = re.sub(r'^(public|private|protected)\s+', '', signature or '', flags=re.IGNORECASE)
    return_type_str, method_name_core_str, params_str_content = "", sig_clean.strip(), ""
    params_match = re.search(r'\((.*?)\)', sig_clean)
    if params_match:
        params_str_content = params_match.group(1)
        before_params = sig_clean[:params_match.start()].strip()
        name_and_type_parts = before_params.split()
        if name_and_type_parts:
            method_name_core_str = name_and_type_parts[-1]
            if include_return_type and len(name_and_type_parts) > 1:
                return_type_str = " ".join(name_and_type_parts[:-1])
        else: method_name_core_str = ""
    else:
        parts_no_params = sig_clean.strip().split()
        if parts_no_params:
            method_name_core_str = parts_no_params[-1]
            if include_return_type and len(parts_no_params) > 1:
                return_type_str = " ".join(parts_no_params[:-1])
        else: method_name_core_str = ""
    tokens = []
    if method_name_core_str: tokens.extend(split_method_name(method_name_core_str).split())
    if include_param_names and params_str_content:
        for p_text in params_str_content.split(','):
            p_text_stripped = p_text.strip()
            if p_text_stripped:
                param_name_parts = p_text_stripped.split()
                if param_name_parts:
                    param_name, param_name_cleaned = param_name_parts[-1], re.sub(r'[^a-zA-Z0-9_].*$', '', param_name_parts[-1])
                    if param_name_cleaned: tokens.extend(split_method_name(param_name_cleaned).split())
    if include_return_type and return_type_str: tokens.extend(split_method_name(return_type_str).split())
    if include_class_name and class_name: tokens.extend(split_method_name(class_name).split())
    if action_annotation: tokens.extend(split_method_name(action_annotation).split())
    return " ".join(filter(None, tokens))

# --- MetricCache ---
class MetricCache:
    def __init__(self, json_dir: Path, baseline_fname: str, class_names_list: list[str]):
        self.json_dir, self.class_names = json_dir, class_names_list
        base_file = json_dir / baseline_fname
        self.baseline_structure = extract_structure_from_json(json.load(open(base_file)) if base_file.is_file() else {})
        files_to_exclude = {baseline_fname, GOLD_STANDARD_MAP_FNAME}
        self.files = [
            p.name for p in json_dir.glob("*.json") 
            if p.name not in files_to_exclude # Check against the set of files to exclude
        ]
        if not self.files:
            print(f"Warning: No JSON files found in {json_dir} (excluding {files_to_exclude}). Analysis might be empty.")
            # Initialize empty structures to prevent downstream errors
            self.json_data = {}
            self.file_info = {}
            self.global_details = []
            self.global_details_df = pd.DataFrame()
            self.global_method_counter = Counter()
            self.gold = _preprocess_gold_map({}) # Empty gold map
            self.model_methods_target_classes = defaultdict(set)
            return
        self.json_data = {f_name: json.load(open(json_dir / f_name)) for f_name in self.files}
        self.file_info = {}
        for f_name in self.files:
            stem = Path(f_name).stem
            model_name = re.sub(r"_run\d+$", "", stem)
            run_match = re.search(r'_run(\d+)$', stem)
            run_id = run_match.group(1) if run_match else '1'
            self.file_info[f_name] = {'model': model_name, 'run': run_id}
        self.global_details = []
        for f_name, data in self.json_data.items():
            if not isinstance(data, dict):
                print(f"Warning: Skipping file {f_name} as its content is not a JSON object (dictionary).")
            for cls_item in data.get('classes', []):
                cname = cls_item.get('name', '')
                for m_item in cls_item.get('methods', []):
                    method_name = m_item.get('name')
                    if method_name:
                        annotation, uc_refs_raw = m_item.get('annotation', {}), m_item.get('annotation', {}).get('uc_references', m_item.get('annotation', {}).get('uc_reference'))
                        ucs_list = []
                        if isinstance(uc_refs_raw, str):
                            ucs_list = [u.strip().upper() for u in uc_refs_raw.split(',') if u.strip()]
                        elif isinstance(uc_refs_raw, list):
                            ucs_list = [str(u).strip().upper() for u in uc_refs_raw if str(u).strip()]
                        action_text = (annotation.get('uc_action', annotation.get('action', '')) or '').strip()
                        self.global_details.append({
            'name': method_name,
            'class': cname,
            'file': f_name,
            'model': self.file_info[f_name]['model'],
            'run': self.file_info[f_name]['run'],
            'signature': m_item.get('signature', ''),
            'ucs': ucs_list,
            'action': action_text,
            'visibility_json': m_item.get('visibility', None) # <<< ADD THIS LINE
        })
        self.global_details_df = pd.DataFrame(self.global_details) if self.global_details else pd.DataFrame()
        self.global_method_counter = Counter(d['name'] for d in self.global_details)
        gold_map_path = Path(GOLD_STANDARD_MAP_FNAME)
        if gold_map_path.is_file():
            self.gold = _preprocess_gold_map(json.load(open(gold_map_path)))
        else:
            print(f"Warning: Gold standard map file '{GOLD_STANDARD_MAP_FNAME}' not found for MetricCache.gold.")
            self.gold = _preprocess_gold_map({})
        if not self.gold.get('all_actions') and gold_map_path.is_file():
            print(f"Warning: Gold standard map '{GOLD_STANDARD_MAP_FNAME}' loaded but seems empty or invalid.")
            
        self.model_methods_target_classes = self._extract_model_methods_for_analysis()

    def _extract_model_methods_for_analysis(self) -> defaultdict[str, set]:
        mm = defaultdict(set)
        for f_name, data_json_content in self.json_data.items():
            mdl = self.file_info[f_name]['model']
            for c_data in data_json_content.get('classes', []):
                if not self.class_names or c_data.get('name') in self.class_names:
                    for m_data in c_data.get('methods', []):
                        if m_data.get('name'): mm[mdl].add(m_data['name'])
        return mm

    def get_structure_reports_df(self) -> pd.DataFrame:
        rows = []
        for f_name, data in self.json_data.items():
            enr = extract_structure_from_json(data)
            rows.append({**compare_structures(self.baseline_structure, enr), 'File': f_name})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['File'])

    def get_per_file_method_metrics(self) -> pd.DataFrame:
        rows = []
        if self.global_details_df.empty: return pd.DataFrame(columns=['Model', 'Run', 'File', 'Target_Class_Methods', 'Extra_Methods_Count', 'Unique_Method_Names', 'Redundancy', 'ParamRichness', 'ReturnTypeCompleteness', 'Percentage_Methods_With_UC', 'Percentage_Methods_With_Action', 'Percentage_Methods_With_Both', 'Percentage_Methods_Without'])
        for file_name, file_data_json in self.json_data.items():
            model_name, run_id = self.file_info[file_name]['model'], self.file_info[file_name]['run']
            methods_in_scope_details, extra_methods_count = [], 0
            for cls_item_json in file_data_json.get('classes', []):
                cls_name_json, methods_in_cls_json = cls_item_json.get('name'), cls_item_json.get('methods', [])
                if not self.class_names or cls_name_json in self.class_names: methods_in_scope_details.extend(m for m in methods_in_cls_json if m.get('name'))
                else: extra_methods_count += len(methods_in_cls_json)
            total_target_methods = len(methods_in_scope_details)
            if total_target_methods == 0:
                rows.append({'Model': model_name, 'Run': run_id, 'File': file_name, 'Target_Class_Methods': 0, 'Extra_Methods_Count': extra_methods_count, 'Unique_Method_Names': 0, 'Redundancy': np.nan, 'ParamRichness': 0, 'ReturnTypeCompleteness': np.nan, 'Percentage_Methods_With_UC': np.nan, 'Percentage_Methods_With_Action': np.nan, 'Percentage_Methods_With_Both': np.nan, 'Percentage_Methods_Without': 100.0 if extra_methods_count == 0 and total_target_methods == 0 else np.nan })
                continue
            unique_method_names_in_scope = len(set(m['name'] for m in methods_in_scope_details))
            param_lengths = [len(m.get('parameters', [])) for m in methods_in_scope_details]
            avg_param_richness = np.mean(param_lengths) if param_lengths else 0
            return_types_present_count = sum(1 for m in methods_in_scope_details if m.get('return_type') and m['return_type'].lower() != 'void')
            return_type_completeness = safe_divide(return_types_present_count * 100.0, total_target_methods, 0.0)
            scoped_file_annotations = self.global_details_df[(self.global_details_df['file'] == file_name) & (self.global_details_df.apply(lambda r: not self.class_names or r['class'] in self.class_names, axis=1))]
            num_methods_for_ann_pct = len(scoped_file_annotations) if not scoped_file_annotations.empty else total_target_methods
            if num_methods_for_ann_pct == 0 and total_target_methods > 0: num_methods_for_ann_pct = total_target_methods
            uc_present_count = scoped_file_annotations['ucs'].apply(bool).sum() if not scoped_file_annotations.empty else 0
            action_present_count = scoped_file_annotations['action'].apply(bool).sum() if not scoped_file_annotations.empty else 0
            both_present_count = scoped_file_annotations.apply(lambda r: bool(r['ucs']) and bool(r['action']), axis=1).sum() if not scoped_file_annotations.empty else 0
            neither_present_count = num_methods_for_ann_pct - uc_present_count - action_present_count + both_present_count
            rows.append({'Model': model_name, 'Run': run_id, 'File': file_name, 'Target_Class_Methods': total_target_methods, 'Extra_Methods_Count': extra_methods_count, 'Unique_Method_Names': unique_method_names_in_scope, 'Redundancy': safe_divide(total_target_methods, unique_method_names_in_scope, 1.0 if total_target_methods > 0 else np.nan), 'ParamRichness': avg_param_richness, 'ReturnTypeCompleteness': return_type_completeness, 'Percentage_Methods_With_UC': safe_divide(uc_present_count*100.0,num_methods_for_ann_pct,0.0), 'Percentage_Methods_With_Action': safe_divide(action_present_count*100.0,num_methods_for_ann_pct,0.0), 'Percentage_Methods_With_Both': safe_divide(both_present_count*100.0,num_methods_for_ann_pct,0.0), 'Percentage_Methods_Without': safe_divide(neither_present_count*100.0,num_methods_for_ann_pct,100.0 if num_methods_for_ann_pct==0 and total_target_methods == 0 else 0.0)})
        return pd.DataFrame(rows)

# --- Annotation & Semantic Mapping ---
def generate_method_annotation_report(cache: MetricCache, details_df_override: pd.DataFrame | None = None) -> pd.DataFrame:
    source_df = details_df_override if details_df_override is not None else cache.global_details_df
    if source_df.empty: return pd.DataFrame(columns=['Model', 'Run', 'File', 'Class', 'MethodName', 'Signature', 'Has_UC_Annotation', 'UC_References', 'UC_Action'])
    methods_to_process_df = pd.DataFrame()
    if details_df_override is not None: methods_to_process_df = source_df.copy()
    elif cache.class_names: methods_to_process_df = source_df[source_df['class'].isin(cache.class_names)].copy()
    else: methods_to_process_df = source_df.copy()
    if methods_to_process_df.empty: return pd.DataFrame(columns=['Model', 'Run', 'File', 'Class', 'MethodName', 'Signature', 'Has_UC_Annotation', 'UC_References', 'UC_Action'])
    methods_to_process_df['Has_UC_Annotation'] = methods_to_process_df['ucs'].apply(lambda x_list: bool(x_list))
    methods_to_process_df['UC_References'] = methods_to_process_df['ucs'].apply(lambda x_list: ','.join(x_list) if isinstance(x_list, list) else "")
    return methods_to_process_df.rename(columns={'model': 'Model', 'run': 'Run', 'file': 'File', 'class': 'Class', 'name': 'MethodName', 'signature': 'Signature', 'action': 'UC_Action'})[['Model', 'Run', 'File', 'Class', 'MethodName', 'Signature', 'Has_UC_Annotation', 'UC_References', 'UC_Action']]

def map_methods_to_actions(annotation_df: pd.DataFrame, gold_data: dict, include_class_name_in_embedding: bool = True, include_param_names_in_embedding: bool = True, include_return_type_in_embedding: bool = True) -> pd.DataFrame:
    if annotation_df.empty:
        out_empty = annotation_df.copy()
        for col in ['EmbedText','Best_Match_Action','SimilarityScore']:
            if col not in out_empty: out_empty[col]=pd.Series(dtype='object' if col!='SimilarityScore' else 'float')
        return out_empty
    if not gold_data or not gold_data.get('action_to_details'):
        print("Warning: Gold data missing/incomplete in map_methods_to_actions.")
        out_partial=annotation_df.copy()
        out_partial['EmbedText']=[prepare_method_embedding_text(r.get('Signature',''),r.get('Class',''),r.get('UC_Action',''),include_class_name_in_embedding,include_param_names_in_embedding,include_return_type_in_embedding) for _,r in annotation_df.iterrows()]
        out_partial['Best_Match_Action']=""; out_partial['SimilarityScore']=np.nan
        return out_partial
    texts_to_embed=[prepare_method_embedding_text(row.get('Signature',''),row.get('Class',''),row.get('UC_Action',''),include_class_name=include_class_name_in_embedding,include_param_names=include_param_names_in_embedding,include_return_type=include_return_type_in_embedding) for _,row in annotation_df.iterrows()]
    if not texts_to_embed:
        out_empty=annotation_df.copy(); out_empty['EmbedText']=""; out_empty['Best_Match_Action']=""; out_empty['SimilarityScore']=np.nan
        return out_empty
    emb_methods=SENTENCE_MODEL.encode(texts_to_embed,convert_to_tensor=True,device=DEVICE)
    gold_actions_list,gold_texts_to_embed=[],[]
    for act,details in gold_data['action_to_details'].items():
        ideal,params,ret=details.get('ideal_method',''),details.get('expected_parameter_concepts',[]),details.get('expected_return_concept','')
        gold_actions_list.append(act)
        current_gold_text_parts=[]
        current_gold_text_parts.extend(split_method_name(ideal).split())
        for p_concept in params: current_gold_text_parts.extend(split_method_name(p_concept).split())
        current_gold_text_parts.extend(split_method_name(ret).split()); current_gold_text_parts.extend(split_method_name(act).split())
        gold_texts_to_embed.append(" ".join(filter(None,current_gold_text_parts)))
    if not gold_texts_to_embed:
        out_partial=annotation_df.copy(); out_partial['EmbedText']=texts_to_embed; out_partial['Best_Match_Action']=""; out_partial['SimilarityScore']=np.nan
        return out_partial
    emb_gold=SENTENCE_MODEL.encode(gold_texts_to_embed,convert_to_tensor=True,device=DEVICE)
    sims=cos_sim(emb_methods,emb_gold).clamp(-1.0,1.0)
    if sims.numel()==0:
        out_partial=annotation_df.copy(); out_partial['EmbedText']=texts_to_embed; out_partial['Best_Match_Action']=""; out_partial['SimilarityScore']=np.nan
        return out_partial
    best_match_indices=torch.argmax(sims,dim=1)
    if sims.ndim==1: scores_tensor=sims.gather(dim=0,index=best_match_indices) if best_match_indices.ndim>0 else sims[best_match_indices]
    elif sims.ndim==2: scores_tensor=sims[torch.arange(sims.shape[0],device=sims.device),best_match_indices]
    else: scores_tensor=torch.tensor([],device=sims.device,dtype=torch.float)
    best_matched_actions_list=[gold_actions_list[idx_val] for idx_val in best_match_indices.cpu().tolist()]
    similarity_scores_list=scores_tensor.cpu().tolist()
    df_out=annotation_df.copy()
    df_out['EmbedText']=pd.Series(texts_to_embed,index=df_out.index)
    df_out['Best_Match_Action']=pd.Series(best_matched_actions_list if best_matched_actions_list else [],index=df_out.index,dtype='object')
    df_out['SimilarityScore']=pd.Series(similarity_scores_list if similarity_scores_list else [],index=df_out.index,dtype='float')
    return df_out

# --- START OF FUNCTION DEFINITIONS ---

def calculate_counts(cache: MetricCache) -> pd.DataFrame:
    data_rows = []
    classes_to_iterate = cache.class_names
    if not classes_to_iterate:
        if cache.global_details_df.empty: return pd.DataFrame(columns=['Class', 'Total'])
        classes_to_iterate = sorted(list(cache.global_details_df['class'].unique()))
        if not classes_to_iterate: return pd.DataFrame(columns=['Class', 'Total'])
    for f_name, data_json in cache.json_data.items():
        row = {'File': f_name}
        for cls_name_target in classes_to_iterate:
            cnt = 0
            for c_item_json in data_json.get('classes', []):
                if c_item_json.get('name') == cls_name_target:
                    cnt = len(c_item_json.get('methods', []))
                    break
            row[cls_name_target] = cnt
        data_rows.append(row)
    if not data_rows: return pd.DataFrame(columns=['Class', 'Total'])
    df = pd.DataFrame(data_rows)
    if 'File' not in df.columns: return pd.DataFrame(columns=['Class', 'Total'])
    df = df.set_index('File')
    df_transposed = df.T 
    df_transposed['Total'] = df_transposed.sum(axis=1)
    return df_transposed.reset_index().rename(columns={'index': 'Class'})

def generate_core_methods_report(cache: MetricCache, top_n: int, main_map_df: pd.DataFrame) -> pd.DataFrame:
    if not cache.global_method_counter: return pd.DataFrame(columns=['MethodName', 'Signature', 'GlobalFrequency', 'MaxSimilarity', 'TopClass', 'TopLLM'])
    common = cache.global_method_counter.most_common(top_n)
    df = pd.DataFrame(common, columns=['MethodName', 'GlobalFrequency'])
    if df.empty: return df
    mapping = main_map_df 
    if mapping.empty or not all(col in mapping.columns for col in ['MethodName', 'Signature', 'SimilarityScore', 'Class', 'Model']):
        df['Signature'], df['MaxSimilarity'], df['TopClass'], df['TopLLM'] = '', 0.0, '', ''
        return df[['MethodName', 'Signature', 'GlobalFrequency', 'MaxSimilarity', 'TopClass', 'TopLLM']]
    sig_counts = mapping.groupby(['MethodName', 'Signature'], observed=True).size()
    top_sig = pd.Series(dtype='object')
    if not sig_counts.empty: top_sig = sig_counts.groupby(level=0, observed=True).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) and len(x)>1 else pd.NA)
    max_sim = mapping.groupby('MethodName', observed=True)['SimilarityScore'].max()
    cls_counts = mapping.groupby(['MethodName', 'Class'], observed=True).size()
    top_class = pd.Series(dtype='object')
    if not cls_counts.empty: top_class = cls_counts.groupby(level=0, observed=True).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) and len(x)>1 else pd.NA)
    llm_counts = mapping.groupby(['MethodName', 'Model'], observed=True).size()
    top_llm = pd.Series(dtype='object')
    if not llm_counts.empty: top_llm = llm_counts.groupby(level=0, observed=True).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) and len(x)>1 else pd.NA)
    df['Signature'] = df['MethodName'].map(top_sig).fillna('')
    df['MaxSimilarity'] = df['MethodName'].map(max_sim).fillna(0.0)
    df['TopClass'] = df['MethodName'].map(top_class).fillna('')
    df['TopLLM'] = df['MethodName'].map(top_llm).fillna('')
    return df[['MethodName', 'Signature', 'GlobalFrequency', 'MaxSimilarity', 'TopClass', 'TopLLM']]

def calculate_action_annotation_coverage(annotation_df: pd.DataFrame, gold_data: dict) -> pd.DataFrame:
    rows = []
    if annotation_df.empty or not gold_data.get('all_actions'): return pd.DataFrame(columns=['Model', 'Annot_Covered_Action_Count', 'Annot_Action_Coverage_%', 'Avg_Per_UC_Annot_Action_Coverage'])
    total_actions = gold_data.get('total_actions', len(gold_data.get('all_actions', [])))
    for mdl, grp in annotation_df.groupby('Model'):
        acts = set(grp['UC_Action'][grp['UC_Action'].isin(gold_data.get('all_actions', set()))])
        count, pct = len(acts), safe_divide(len(acts) * 100.0, total_actions, 0.0)
        per_uc_cov = [safe_divide(len(acts & actions_in_uc), len(actions_in_uc), 0.0) for uc, actions_in_uc in gold_data.get('uc_to_actions', {}).items() if actions_in_uc]
        avg_cov = float(np.mean(per_uc_cov)) if per_uc_cov else 0.0
        rows.append({'Model': mdl, 'Annot_Covered_Action_Count': count, 'Annot_Action_Coverage_%': pct, 'Avg_Per_UC_Annot_Action_Coverage': avg_cov})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model', 'Annot_Covered_Action_Count', 'Annot_Action_Coverage_%', 'Avg_Per_UC_Annot_Action_Coverage'])

def calculate_uc_annotation_coverage(annotation_df: pd.DataFrame, gold_data: dict) -> pd.DataFrame:
    rows = []
    if annotation_df.empty or not gold_data.get('all_uc_ids'): return pd.DataFrame(columns=['Model', 'Annot_Covered_UC_Count', 'Annot_UC_Hit_Coverage_Percent', 'Avg_Per_UC_Annot_Action_Coverage'])
    total_ucs = gold_data.get('total_ucs', len(gold_data.get('all_uc_ids',[])))
    for mdl, grp in annotation_df.groupby('Model'):
        ucs_referenced = set()
        if 'UC_References' in grp.columns:
            for refs_str in grp['UC_References'].dropna():
                if refs_str: ucs_referenced.update(uc_val.strip() for uc_val in refs_str.split(',') if uc_val.strip())
        valid_model_ucs = ucs_referenced & gold_data.get('all_uc_ids', set())
        annotated_actions_in_model = set(grp['UC_Action'][grp['UC_Action'].isin(gold_data.get('all_actions', set()))])
        per_uc_action_cov = [safe_divide(len(annotated_actions_in_model & gold_data.get('uc_to_actions',{}).get(uc_id,set())),len(gold_data.get('uc_to_actions',{}).get(uc_id,set())),0.0) for uc_id in valid_model_ucs if gold_data.get('uc_to_actions',{}).get(uc_id,set())]
        avg_action_cov_in_hit_ucs = float(np.mean(per_uc_action_cov)) if per_uc_action_cov else 0.0
        rows.append({'Model': mdl, 'Annot_Covered_UC_Count': len(valid_model_ucs), 'Annot_UC_Hit_Coverage_Percent': safe_divide(len(valid_model_ucs)*100.0,total_ucs,0.0), 'Avg_Per_UC_Annot_Action_Coverage': avg_action_cov_in_hit_ucs})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model', 'Annot_Covered_UC_Count', 'Annot_UC_Hit_Coverage_Percent', 'Avg_Per_UC_Annot_Action_Coverage'])

def calculate_action_semantic_coverage(map_df: pd.DataFrame, gold_data: dict, threshold: float) -> pd.DataFrame:
    rows = []
    if map_df.empty or 'SimilarityScore' not in map_df.columns or not gold_data.get('all_actions'): return pd.DataFrame(columns=['Model', 'Sem_Covered_Action_Count', 'Sem_Action_Coverage_Percent', 'Avg_Per_UC_Sem_Action_Coverage'])
    total_actions = gold_data.get('total_actions', len(gold_data.get('all_actions',[])))
    for mdl, grp in map_df.groupby('Model'):
        valid_matches = grp[grp['SimilarityScore'] >= threshold]
        distinct_matched_actions, set_distinct_matched_actions = valid_matches['Best_Match_Action'].nunique(), set(valid_matches['Best_Match_Action'].unique()) & gold_data.get('all_actions',set())
        pct = safe_divide(distinct_matched_actions*100.0,total_actions,0.0)
        per_uc_cov = [safe_divide(len(set_distinct_matched_actions & actions_in_uc_gold),len(actions_in_uc_gold),0.0) for uc,actions_in_uc_gold in gold_data.get('uc_to_actions',{}).items() if actions_in_uc_gold]
        avg_per_uc = float(np.mean(per_uc_cov)) if per_uc_cov else 0.0
        rows.append({'Model':mdl,'Sem_Covered_Action_Count':distinct_matched_actions,'Sem_Action_Coverage_Percent':pct,'Avg_Per_UC_Sem_Action_Coverage':avg_per_uc})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model', 'Sem_Covered_Action_Count', 'Sem_Action_Coverage_Percent', 'Avg_Per_UC_Sem_Action_Coverage'])

def calculate_uc_semantic_coverage(map_df: pd.DataFrame, gold_data: dict, threshold: float) -> pd.DataFrame:
    rows = []
    if map_df.empty or 'SimilarityScore' not in map_df.columns or not gold_data.get('all_uc_ids'): return pd.DataFrame(columns=['Model', 'Sem_Covered_UC_Count', 'Sem_UC_Hit_Coverage_Percent', 'Avg_Per_UC_Sem_Action_Coverage'])
    total_ucs = gold_data.get('total_ucs', len(gold_data.get('all_uc_ids',[])))
    if not all(col in map_df.columns for col in ['Model','Best_Match_Action','SimilarityScore']): return pd.DataFrame(columns=['Model','Sem_Covered_UC_Count','Sem_UC_Hit_Coverage_Percent','Avg_Per_UC_Sem_Action_Coverage'])
    df_copy = map_df.copy(); df_copy['Match'] = df_copy['SimilarityScore'] >= threshold
    for mdl, grp in df_copy.groupby('Model'):
        semantically_matched_actions = set(grp.loc[grp['Match'],'Best_Match_Action'])
        semantically_covered_ucs = {gold_data['action_to_details'][act]['uc_id'] for act in semantically_matched_actions if act in gold_data.get('action_to_details',{})} & gold_data.get('all_uc_ids',set())
        per_uc_action_cov = [safe_divide(len(semantically_matched_actions & gold_data.get('uc_to_actions',{}).get(uc_id,set())),len(gold_data.get('uc_to_actions',{}).get(uc_id,set())),0.0) for uc_id in semantically_covered_ucs if gold_data.get('uc_to_actions',{}).get(uc_id,set())]
        avg_action_cov_in_hit_ucs = float(np.mean(per_uc_action_cov)) if per_uc_action_cov else 0.0
        rows.append({'Model':mdl,'Sem_Covered_UC_Count':len(semantically_covered_ucs),'Sem_UC_Hit_Coverage_Percent':safe_divide(len(semantically_covered_ucs)*100.0,total_ucs,0.0),'Avg_Per_UC_Sem_Action_Coverage':avg_action_cov_in_hit_ucs})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model','Sem_Covered_UC_Count','Sem_UC_Hit_Coverage_Percent','Avg_Per_UC_Sem_Action_Coverage'])

def calculate_action_combined_coverage(annotation_df: pd.DataFrame, map_df: pd.DataFrame, gold_data: dict, threshold: float) -> pd.DataFrame:
    rows = []
    if(annotation_df.empty and map_df.empty) or not gold_data.get('all_actions'): return pd.DataFrame(columns=['Model','Comb_Covered_Action_Count','Comb_Action_Coverage_%','Avg_Per_UC_Comb_Action_Coverage'])
    all_models_set=set();
    if not annotation_df.empty and 'Model' in annotation_df.columns: all_models_set.update(annotation_df['Model'].unique())
    if not map_df.empty and 'Model' in map_df.columns: all_models_set.update(map_df['Model'].unique())
    if not all_models_set: return pd.DataFrame(columns=['Model','Comb_Covered_Action_Count','Comb_Action_Coverage_%','Avg_Per_UC_Comb_Action_Coverage'])
    total_actions = gold_data.get('total_actions', len(gold_data.get('all_actions', [])))
    for mdl in sorted(list(all_models_set)):
        ann_acts_model,sem_acts_model=set(),set()
        if not annotation_df.empty and 'Model' in annotation_df.columns and 'UC_Action' in annotation_df.columns:
            model_ann_df=annotation_df[annotation_df['Model']==mdl]
            if not model_ann_df.empty: ann_acts_model=set(model_ann_df['UC_Action'])&gold_data.get('all_actions',set())
        if not map_df.empty and 'Model' in map_df.columns and 'SimilarityScore' in map_df.columns and 'Best_Match_Action' in map_df.columns:
            model_map_df=map_df[map_df['Model']==mdl]
            if not model_map_df.empty: sem_acts_model=set(model_map_df[model_map_df['SimilarityScore']>=threshold]['Best_Match_Action'].unique())&gold_data.get('all_actions',set())
        union_actions,count=ann_acts_model|sem_acts_model,len(ann_acts_model|sem_acts_model)
        pct=safe_divide(count*100.0,total_actions,0.0)
        uc_covs=[safe_divide(len(union_actions&actions_in_uc_gold),len(actions_in_uc_gold),0.0) for uc,actions_in_uc_gold in gold_data.get('uc_to_actions',{}).items() if actions_in_uc_gold]
        avg_uc_cov=np.mean(uc_covs) if uc_covs else 0.0
        rows.append({'Model':mdl,'Comb_Covered_Action_Count':count,'Comb_Action_Coverage_%':pct,'Avg_Per_UC_Comb_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model','Comb_Covered_Action_Count','Comb_Action_Coverage_%','Avg_Per_UC_Comb_Action_Coverage'])

def calculate_uc_combined_coverage(annotation_df: pd.DataFrame, map_df: pd.DataFrame, gold_data: dict, threshold: float) -> pd.DataFrame:
    rows=[]
    if(annotation_df.empty and map_df.empty) or not gold_data.get('all_uc_ids'): return pd.DataFrame(columns=['Model','Comb_Covered_UC_Count','Comb_UC_Coverage_%'])
    all_models_set=set();
    if not annotation_df.empty and 'Model' in annotation_df.columns: all_models_set.update(annotation_df['Model'].unique())
    if not map_df.empty and 'Model' in map_df.columns: all_models_set.update(map_df['Model'].unique())
    if not all_models_set: return pd.DataFrame(columns=['Model','Comb_Covered_UC_Count','Comb_UC_Coverage_%'])
    total_ucs=gold_data.get('total_ucs',len(gold_data.get('all_uc_ids',[])))
    for mdl in sorted(list(all_models_set)):
        u_anns,u_sems=set(),set()
        if not annotation_df.empty and 'Model' in annotation_df.columns and 'UC_References' in annotation_df.columns:
            model_ann_df=annotation_df[annotation_df['Model']==mdl]
            if not model_ann_df.empty:
                for refs_str in model_ann_df['UC_References'].dropna():
                    if refs_str: u_anns.update(uc_val.strip() for uc_val in refs_str.split(',') if uc_val.strip())
        u_anns&=gold_data.get('all_uc_ids',set())
        if not map_df.empty and 'Model' in map_df.columns and 'SimilarityScore' in map_df.columns and 'Best_Match_Action' in map_df.columns and gold_data.get('action_to_details'):
            model_map_df=map_df[map_df['Model']==mdl]
            if not model_map_df.empty:
                matched_actions_above_thresh=set(model_map_df[model_map_df['SimilarityScore']>=threshold]['Best_Match_Action'].unique())
                u_sems={gold_data['action_to_details'][act]['uc_id'] for act in matched_actions_above_thresh if act in gold_data['action_to_details']}
        u_sems&=gold_data.get('all_uc_ids',set())
        union_ucs,count=u_anns|u_sems,len(u_anns|u_sems)
        pct=safe_divide(count*100.0,total_ucs,0.0)
        rows.append({'Model':mdl,'Comb_Covered_UC_Count':count,'Comb_UC_Coverage_%':pct})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model','Comb_Covered_UC_Count','Comb_UC_Coverage_%'])

def calculate_bootstrap_overlap(perfile_df:pd.DataFrame,n_iter:int=1000)->pd.DataFrame:
    if perfile_df.empty or 'Model' not in perfile_df.columns or 'Target_Class_Methods' not in perfile_df.columns: return pd.DataFrame()
    vals={mdl:grp['Target_Class_Methods'].values for mdl,grp in perfile_df.groupby('Model')}
    models,M=list(vals.keys()),len(list(vals.keys()))
    if M==0: return pd.DataFrame()
    mat=pd.DataFrame(np.nan,index=models,columns=models)
    for i,a in enumerate(models):
        for j,b in enumerate(models):
            if a==b: continue 
            samples_a,samples_b=vals[a],vals[b]
            if len(samples_a)==0 or len(samples_b)==0: mat.loc[a,b]=np.nan; continue
            greater=sum(1 for _ in range(n_iter) if np.mean(np.random.choice(samples_a,len(samples_a),True))>np.mean(np.random.choice(samples_b,len(samples_b),True)))
            mat.loc[a,b]=greater/n_iter
    return mat

def calculate_jaccard_global(cache:MetricCache)->pd.DataFrame:
    mm=cache.model_methods_target_classes 
    models,M=list(mm.keys()),len(list(mm.keys()))
    if M==0: return pd.DataFrame()
    mat=pd.DataFrame(np.zeros((M,M)),index=models,columns=models)
    for i,a in enumerate(models):
        for j,b in enumerate(models):
            if i==j : mat.iloc[i,j]=1.0
            else: ia,ib=mm[a],mm[b]; mat.iloc[i,j]=safe_divide(len(ia&ib),len(ia|ib),0.0)
    return mat

def calculate_per_class_jaccard(cache:MetricCache)->dict[str,pd.DataFrame]:
    out={}
    class_list_for_jaccard=cache.class_names
    if not class_list_for_jaccard:
        if cache.global_details_df.empty: return {}
        class_list_for_jaccard=sorted(list(cache.global_details_df['class'].unique()))
    for cls_name in class_list_for_jaccard:
        mm_class=defaultdict(set)
        for f_name,data_json in cache.json_data.items():
            mdl=cache.file_info[f_name]['model']
            for c_item_json in data_json.get('classes',[]):
                if c_item_json.get('name')==cls_name:
                    for m_item_json in c_item_json.get('methods',[]): 
                        if m_item_json.get('name'): mm_class[mdl].add(m_item_json['name'])
                    break
        models_in_class,M_class=list(mm_class.keys()),len(list(mm_class.keys()))
        if M_class==0: out[cls_name]=pd.DataFrame(); continue
        mat_class=pd.DataFrame(np.zeros((M_class,M_class)),index=models_in_class,columns=models_in_class)
        for i,a_mdl in enumerate(models_in_class):
            for j,b_mdl in enumerate(models_in_class):
                if i==j: mat_class.iloc[i,j]=1.0
                else: set_a,set_b=mm_class[a_mdl],mm_class[b_mdl]; mat_class.iloc[i,j]=safe_divide(len(set_a&set_b),len(set_a|set_b),0.0)
        out[cls_name]=mat_class
    return out

def calculate_method_metrics_summary(perfile_df: pd.DataFrame) -> pd.DataFrame:
    if perfile_df.empty or 'Model' not in perfile_df.columns: return pd.DataFrame(columns=['Model','Avg_Target_Class_Methods','Std_Target_Class_Methods','Avg_Redundancy','Avg_ParamRichness','Avg_ReturnTypeCompleteness','Total_Extra_Methods','Total_Unique_Method_Names_Across_Runs'])
    required_cols=['Target_Class_Methods','Redundancy','ParamRichness','ReturnTypeCompleteness','Extra_Methods_Count','Unique_Method_Names']
    df_to_agg=perfile_df.copy()
    for col in required_cols:
        if col not in df_to_agg.columns: df_to_agg[col]=np.nan if 'Std' in col or 'Avg' in col or 'Redundancy' in col else 0
    grp=df_to_agg.groupby('Model')
    summary=pd.DataFrame({'Avg_Target_Class_Methods':grp['Target_Class_Methods'].mean(),'Std_Target_Class_Methods':grp['Target_Class_Methods'].std(ddof=1),'Avg_Redundancy':grp['Redundancy'].mean(),'Avg_ParamRichness':grp['ParamRichness'].mean(),'Avg_ReturnTypeCompleteness':grp['ReturnTypeCompleteness'].mean(),'Total_Extra_Methods':grp['Extra_Methods_Count'].sum(),'Total_Unique_Method_Names_Across_Runs':grp['Unique_Method_Names'].sum()}).reset_index()
    return summary

def calculate_variability(perfile_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if perfile_df.empty or 'Model' not in perfile_df.columns or 'Target_Class_Methods' not in perfile_df.columns: return pd.DataFrame(columns=['Model','Mean','CV','CI_low','CI_high','ConvergenceSlope','NumRuns'])
    for mdl,grp in perfile_df.groupby('Model'):
        vals,n=grp['Target_Class_Methods'].dropna().values,len(grp['Target_Class_Methods'].dropna().values)
        if n==0: m,s,cv,ci_low,ci_high,slope=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        else:
            m,s=vals.mean(),(vals.std(ddof=1) if n>1 else 0.0)
            cv=safe_divide(s,m,0.0 if m==0 and s==0 else np.nan)
            ci_low,ci_high=m,m
            if n>1 and s>0:
                try: ci_low,ci_high=t.interval(0.95,n-1,loc=m,scale=max(s/math.sqrt(n),1e-9))
                except Exception: pass
            slope=np.polyfit(np.arange(1,n+1),vals,1)[0] if n>1 else 0.0
        rows.append({'Model':mdl,'Mean':m,'CV':cv,'CI_low':ci_low,'CI_high':ci_high,'ConvergenceSlope':slope,'NumRuns':n})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model','Mean','CV','CI_low','CI_high','ConvergenceSlope','NumRuns'])

def calculate_diversity(cache: MetricCache) -> pd.DataFrame:
    rows=[]
    mcc_model_class_counts,model_method_sets=defaultdict(lambda:Counter()),cache.model_methods_target_classes
    for f_name,data_json_content in cache.json_data.items():
        mdl=cache.file_info[f_name]['model']
        for c_data in data_json_content.get('classes',[]):
            class_name=c_data.get('name')
            if not cache.class_names or class_name in cache.class_names: mcc_model_class_counts[mdl][class_name]+=len(c_data.get('methods',[]))
    if not mcc_model_class_counts and not model_method_sets: return pd.DataFrame(columns=['Model','Gini','Entropy','NormalizedEntropy','ExclusiveMethodsCount'])
    all_models_with_data=set(mcc_model_class_counts.keys())|set(model_method_sets.keys())
    num_classes_for_norm=len(cache.class_names) if cache.class_names else (len(set(cls for counts in mcc_model_class_counts.values() for cls in counts.keys())) if any(mcc_model_class_counts.values()) else 1)
    for mdl in all_models_with_data:
        counts_per_class_for_model,total_methods_in_model_scope=mcc_model_class_counts.get(mdl,Counter()),sum(mcc_model_class_counts.get(mdl,Counter()).values())
        gini,ent,norm_ent=0.0,0.0,0.0
        if total_methods_in_model_scope>0:
            props=[v/total_methods_in_model_scope for v in counts_per_class_for_model.values() if v>0]
            if props:
                ent= -sum(p*math.log2(p) for p in props if p>0)
                norm_ent=safe_divide(ent,math.log2(num_classes_for_norm),0.0) if num_classes_for_norm>1 else(1.0 if ent>0 else 0.0)
                gini=1-sum(p*p for p in props)
        current_model_methods,other_models_methods_union=model_method_sets.get(mdl,set()),set().union(*(model_method_sets.get(m,set()) for m in model_method_sets if m!=mdl))
        excl_count=len(current_model_methods-other_models_methods_union)
        rows.append({'Model':mdl,'Gini':gini,'Entropy':ent,'NormalizedEntropy':norm_ent,'ExclusiveMethodsCount':excl_count})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Model','Gini','Entropy','NormalizedEntropy','ExclusiveMethodsCount'])

def calculate_core_method_coverage(cache: MetricCache, core_methods_list: list[str]) -> pd.DataFrame:
    results=[]
    if not core_methods_list or cache.global_details_df.empty: return pd.DataFrame(columns=['Model','Unique_Core_Methods','Total_Core_Occurrences'])
    core_set=set(core_methods_list)
    scoped_methods_df=cache.global_details_df[cache.global_details_df['class'].isin(cache.class_names)].copy() if cache.class_names else cache.global_details_df.copy()
    if scoped_methods_df.empty: return pd.DataFrame(columns=['Model','Unique_Core_Methods','Total_Core_Occurrences'])
    for model_name,group in scoped_methods_df.groupby('model'):
        method_names_in_model_scope=group['name'].tolist()
        unique_core_found,total_core_occurrences=len(set(method_names_in_model_scope)&core_set),sum(1 for name_item in method_names_in_model_scope if name_item in core_set)
        results.append({'Model':model_name,'Unique_Core_Methods':unique_core_found,'Total_Core_Occurrences':total_core_occurrences})
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['Model','Unique_Core_Methods','Total_Core_Occurrences'])

def calculate_global_ranking(cache:MetricCache, perfile_df:pd.DataFrame, annotation_df:pd.DataFrame, mapping_df:pd.DataFrame)->pd.DataFrame:
    def get_metric_series(df_metric,index_col_metric,value_col_metric):
        if df_metric.empty or index_col_metric not in df_metric.columns or value_col_metric not in df_metric.columns: return pd.Series(dtype=float,name=value_col_metric)
        df_metric_no_dupes=df_metric.drop_duplicates(subset=[index_col_metric])
        return df_metric_no_dupes.set_index(index_col_metric)[value_col_metric]
    cov_act=get_metric_series(calculate_action_combined_coverage(annotation_df,mapping_df,cache.gold,SEMANTIC_SIMILARITY_THRESHOLD),'Model','Comb_Action_Coverage_%')
    cov_uc=get_metric_series(calculate_uc_combined_coverage(annotation_df,mapping_df,cache.gold,SEMANTIC_SIMILARITY_THRESHOLD),'Model','Comb_UC_Coverage_%')
    sem_scores_grouped=pd.Series(dtype=float)
    if not mapping_df.empty and 'Model' in mapping_df and 'SimilarityScore' in mapping_df: sem_scores_grouped=mapping_df.groupby('Model')['SimilarityScore'].mean()
    sem=get_metric_series(sem_scores_grouped.reset_index(),'Model','SimilarityScore')
    summary_metrics=calculate_method_metrics_summary(perfile_df) 
    red=get_metric_series(summary_metrics,'Model','Avg_Redundancy')
    pr=get_metric_series(summary_metrics,'Model','Avg_ParamRichness')
    rtc=get_metric_series(summary_metrics,'Model','Avg_ReturnTypeCompleteness')
    var_metrics=calculate_variability(perfile_df) 
    cv=get_metric_series(var_metrics,'Model','CV')
    slope=get_metric_series(var_metrics,'Model','ConvergenceSlope')
    div_metrics=calculate_diversity(cache) 
    ent=get_metric_series(div_metrics,'Model','NormalizedEntropy')
    excl=get_metric_series(div_metrics,'Model','ExclusiveMethodsCount')
    jac_mat,jaccard=calculate_jaccard_global(cache),pd.Series(dtype=float)
    if not jac_mat.empty: jaccard=(jac_mat.sum(axis=1)-1)/(jac_mat.shape[1]-1) if jac_mat.shape[1]>1 else pd.Series(1.0,index=jac_mat.index)
    all_models_ranking=set().union(*(s.index for s in[cov_act,cov_uc,sem,red,pr,rtc,cv,slope,ent,excl,jaccard] if not s.empty))
    df_metrics_ranking=pd.DataFrame(index=sorted(list(all_models_ranking)))
    for name,s_metric in zip(['cov_act','cov_uc','sem','red','pr','rtc','cv','slope','ent','excl','jaccard'],[cov_act,cov_uc,sem,red,pr,rtc,cv,slope,ent,excl,jaccard]): df_metrics_ranking[name]=s_metric.reindex(df_metrics_ranking.index)
    def norm(s_in,inv=False):
        s_clean=s_in.dropna()
        if s_clean.empty: return pd.Series(0.5,index=s_in.index)
        mn,mx=s_clean.min(),s_clean.max()
        if mx==mn: return pd.Series(0.0 if inv else 1.0,index=s_in.index).fillna(0.5)
        sc=(s_clean-mn)/(mx-mn); norm_s_clean=1.0-sc if inv else sc
        return norm_s_clean.reindex(s_in.index).fillna(0.5)
    normed_df_ranking=pd.DataFrame(index=df_metrics_ranking.index)
    norm_cols,norm_inv,normed_col_names=['cov_act','cov_uc','sem','red','pr','rtc','cv','slope','ent','excl','jaccard'],[False,False,False,True,False,False,True,False,False,False,False],['coverage_act','coverage_uc','semantic','redundancy','paramRich','returnComp','vol_cv','vol_slope','breadth_ent','breadth_excl','agreement']
    for i,col_orig in enumerate(norm_cols): normed_df_ranking[normed_col_names[i]]=norm(df_metrics_ranking[col_orig],inv=norm_inv[i])
    normed_df_ranking.dropna(how='all',inplace=True)
    if normed_df_ranking.empty: return pd.DataFrame(columns=['FinalScore'])
    weights={'coverage_act':0.15,'coverage_uc':0.10,'semantic':0.20,'redundancy':0.05,'paramRich':0.05,'returnComp':0.05,'vol_cv':0.05,'vol_slope':0.05,'breadth_ent':0.10,'breadth_excl':0.10,'agreement':0.10}
    weights_s=pd.Series(weights).reindex(normed_df_ranking.columns).fillna(0)
    scores=(normed_df_ranking*weights_s).sum(axis=1)
    return scores.sort_values(ascending=False).to_frame('FinalScore')

# --- END OF FUNCTION DEFINITIONS ---

# --- Main ---
def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading SentenceTransformer model: {NLP_MODEL_NAME}...")
    print(f"Initializing MetricCache. Target classes for most metrics: {CLASS_NAMES if CLASS_NAMES else 'ALL'}")
    cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, CLASS_NAMES)
    print("MetricCache initialized.")

    print("Generating structure reports...")
    struct_df = cache.get_structure_reports_df()
    temp_counts_list = []
    classes_in_scope_for_counts = cache.class_names
    if not classes_in_scope_for_counts:
        if not cache.global_details_df.empty: classes_in_scope_for_counts = sorted(list(cache.global_details_df['class'].unique()))
        else: classes_in_scope_for_counts = []
    for f_name_json, data_json_content in cache.json_data.items():
        row_cts, file_total_methods_in_scope = {'File': f_name_json}, 0
        for cls_n in classes_in_scope_for_counts:
            cls_methods = next((len(c.get('methods',[])) for c in data_json_content.get('classes',[]) if c.get('name')==cls_n), 0)
            row_cts[cls_n], file_total_methods_in_scope = cls_methods, file_total_methods_in_scope + cls_methods
        row_cts['TotalMethodsInScope'] = file_total_methods_in_scope
        temp_counts_list.append(row_cts)
    counts_wide_df = pd.DataFrame(temp_counts_list) if temp_counts_list else pd.DataFrame(columns=['File'])
    
    print("Calculating per-file method metrics...")
    perfile_df = cache.get_per_file_method_metrics() 
    dynamic_top_n = TOP_N_CORE
    if not perfile_df.empty and 'Target_Class_Methods' in perfile_df.columns:
        mean_methods = perfile_df['Target_Class_Methods'].mean()
        if not pd.isna(mean_methods) and mean_methods > 0 : dynamic_top_n = int(round(mean_methods))
    print(f"Dynamic TOP_N_CORE for core methods set to: {dynamic_top_n}")

    print("Combining structural, counts, and per-file metrics...")
    combined_struct_metrics_df = struct_df
    if not counts_wide_df.empty: combined_struct_metrics_df = combined_struct_metrics_df.merge(counts_wide_df, on='File', how='left')
    if not perfile_df.empty: combined_struct_metrics_df = combined_struct_metrics_df.merge(perfile_df, on='File', how='left')
    combined_struct_metrics_df.to_csv(REPORT_DIR / "Combined_Struct_Counts_Metrics.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'Combined_Struct_Counts_Metrics.csv'}")

    print("Generating method annotation report (main)...")
    ann_df = generate_method_annotation_report(cache)
    print("Mapping methods to actions (main version, with class/param/return context)...")
    map_df = map_methods_to_actions(ann_df, cache.gold, include_class_name_in_embedding=False, include_param_names_in_embedding=False, include_return_type_in_embedding=False)
    ann_map_combined_df = pd.DataFrame()
    if not ann_df.empty:
        ann_map_combined_df = ann_df.copy()
        if not map_df.empty and all(c in map_df.columns for c in ['File','Class','MethodName','Best_Match_Action','SimilarityScore']):
            merge_keys = list(set(['Model','Run','File','Class','MethodName']) & set(ann_df.columns) & set(map_df.columns))
            cols_from_map = list(set(merge_keys + ['Best_Match_Action','SimilarityScore']) & set(map_df.columns))
            map_df_to_merge = map_df[cols_from_map].drop_duplicates(subset=merge_keys)
            ann_map_combined_df = ann_map_combined_df.merge(map_df_to_merge, on=merge_keys, how='left')
        else: ann_map_combined_df['Best_Match_Action'], ann_map_combined_df['SimilarityScore'] = "", np.nan
    expected_ann_map_cols = ['Model','Run','File','Class','MethodName','Signature','Has_UC_Annotation','UC_References','UC_Action','Best_Match_Action','SimilarityScore']
    for col in expected_ann_map_cols:
        if col not in ann_map_combined_df.columns: ann_map_combined_df[col] = "" if col != 'SimilarityScore' else np.nan
    ann_map_combined_df[expected_ann_map_cols].to_csv(REPORT_DIR / "Annotation_and_Mapping_Combined.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'Annotation_and_Mapping_Combined.csv'}")

    print("Calculating model-level summary metrics...")
    summary_df, variability_df, diversity_df = calculate_method_metrics_summary(perfile_df), calculate_variability(perfile_df), calculate_diversity(cache)
    act_ann_cov_df = calculate_action_annotation_coverage(ann_df,cache.gold).rename(columns=lambda c:c.replace('Annot_','CovActAnn_') if 'Count' in c or '%' in c else c)
    uc_ann_cov_df = calculate_uc_annotation_coverage(ann_df,cache.gold).rename(columns=lambda c:c.replace('Annot_','CovUCAnn_') if 'Count' in c or '%' in c else c)
    act_sem_cov_df = calculate_action_semantic_coverage(map_df,cache.gold,SEMANTIC_SIMILARITY_THRESHOLD).rename(columns=lambda c:c.replace('Sem_','CovActSem_') if 'Count' in c or '%' in c else c)
    uc_sem_cov_df = calculate_uc_semantic_coverage(map_df,cache.gold,SEMANTIC_SIMILARITY_THRESHOLD).rename(columns=lambda c:c.replace('Sem_','CovUCSem_') if 'Count' in c or '%' in c else c)
    model_summary_list, model_summary_combined_df = [summary_df,variability_df,diversity_df,act_ann_cov_df,uc_ann_cov_df,act_sem_cov_df,uc_sem_cov_df], pd.DataFrame(columns=['Model'])
    for df_item in model_summary_list:
        if not df_item.empty and 'Model' in df_item.columns:
            df_item['Model'] = df_item['Model'].astype(str)
            cols_to_drop = [col for col in df_item.columns if col in model_summary_combined_df.columns and col != 'Model']
            model_summary_combined_df = model_summary_combined_df.merge(df_item.drop(columns=cols_to_drop), on='Model', how='outer')
    model_summary_combined_df.to_csv(REPORT_DIR / "Model_Summary_CombinedMetrics.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'Model_Summary_CombinedMetrics.csv'}")

    print(f"Generating core methods report (Top {dynamic_top_n})...")
    core_methods_df = generate_core_methods_report(cache, dynamic_top_n, map_df)
    core_methods_df.to_csv(REPORT_DIR / "CoreMethods_TopN.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'CoreMethods_TopN.csv'}")
    if not core_methods_df.empty and 'MethodName' in core_methods_df.columns:
        core_cov_df = calculate_core_method_coverage(cache,core_methods_df['MethodName'].tolist()).rename(columns={'Unique_Core_Methods':'UniqueCore','Total_Core_Occurrences':'TotalCore'})
        core_cov_df.to_csv(REPORT_DIR / "CoreCoverageMetrics.csv", index=False)
        print(f"Saved: {REPORT_DIR / 'CoreCoverageMetrics.csv'}")

    print("Calculating bootstrap overlap...")
    calculate_bootstrap_overlap(perfile_df).to_csv(REPORT_DIR / "BootstrapOverlap.csv")
    print(f"Saved: {REPORT_DIR / 'BootstrapOverlap.csv'}")
    print("Calculating Jaccard matrices (global and per-class)...")
    calculate_jaccard_global(cache).to_csv(REPORT_DIR / "JaccardMatrix_Global.csv")
    print(f"Saved: {REPORT_DIR / 'JaccardMatrix_Global.csv'}")
    jaccard_per_class_dict = calculate_per_class_jaccard(cache)
    if jaccard_per_class_dict:
        print(f"Saving per-class Jaccard matrices ({len(jaccard_per_class_dict)} classes found)...")
        saved_count = 0
        for cls_name_jac_original, jac_df_cls in jaccard_per_class_dict.items():
            if not jac_df_cls.empty:
                sanitized_cls_name_for_file = sanitize_classname_for_filename(cls_name_jac_original)
                output_filename = f"JaccardMatrix_{sanitized_cls_name_for_file}.csv"
                try: jac_df_cls.to_csv(REPORT_DIR/output_filename); saved_count += 1
                except OSError as e: print(f"Error saving Jaccard for '{cls_name_jac_original}' (file: '{output_filename}'): {e}")
        print(f"Saved {saved_count} per-class Jaccard matrices to: {REPORT_DIR}")
    else: print("No per-class Jaccard matrices generated.")

    print("Calculating final LLM ranking...")
    final_ranking_df = calculate_global_ranking(cache, perfile_df, ann_df, map_df)
    final_ranking_df.to_csv(REPORT_DIR / "LLM_Final_Ranking_Weighted.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'LLM_Final_Ranking_Weighted.csv'}")
    
    # 7) Behavioral Correctness (BC) Sensitivity Summary (Per-Run Metrics, then Averaged/Max)
    print("Calculating BC sensitivity summary (Per-Run Metrics, then Averaged/Max)...")
    SEMANTIC_SIMILARITY_THRESHOLDS_BC = [0.5, 0.6, 0.7]
    print("Generating annotation report for ALL methods specifically for BC sensitivity...")
    ann_df_for_bc = generate_method_annotation_report(cache, details_df_override=cache.global_details_df)
    print(f"Total method instances considered for BC sensitivity (from global_details_df, across all runs): {len(ann_df_for_bc)}")

    if not ann_df_for_bc.empty:
        print(f"Mapping {len(ann_df_for_bc)} method instances for BC sensitivity (embedding excludes class/param/return context)...")
        map_df_for_bc = map_methods_to_actions(
            ann_df_for_bc, cache.gold,
            include_class_name_in_embedding=False,
            include_param_names_in_embedding=False,
            include_return_type_in_embedding=False 
        )

        if not map_df_for_bc.empty and 'SimilarityScore' in map_df_for_bc.columns and 'Run' in map_df_for_bc.columns \
           and 'Best_Match_Action' in map_df_for_bc.columns:
            
            bc_model_summary_results = []
            total_gold_actions_bc = cache.gold.get('total_actions', 0)
            if total_gold_actions_bc == 0: 
                total_gold_actions_bc = len(cache.gold.get('all_actions',[]))
            if total_gold_actions_bc == 0: 
                print("Critical Warning (BC): Total gold actions is 0. Coverage percentages will be 0 or NaN.")

            for bc_thr in SEMANTIC_SIMILARITY_THRESHOLDS_BC:
                current_bc_map_df = map_df_for_bc.copy()
                current_bc_map_df['SuccessfulMatch'] = current_bc_map_df['SimilarityScore'] >= bc_thr
                
                per_run_metrics = []
                for (model_name, run_id), run_group_df in current_bc_map_df.groupby(['Model', 'Run'], observed=True):
                    total_methods_in_this_run = len(run_group_df)
                    successfully_matched_gold_actions_in_run = run_group_df[run_group_df['SuccessfulMatch']]['Best_Match_Action'].nunique()
                    per_run_metrics.append({'Model': model_name, 'Run': run_id, 'Threshold': bc_thr, 
                                            'TotalMethodsInRun': total_methods_in_this_run, 
                                            'UniqueGoldActionsCoveredInRun': successfully_matched_gold_actions_in_run})
                
                if not per_run_metrics: 
                    print(f"BC (thr={bc_thr}): No per-run metrics generated. Skipping model aggregation.")
                    continue
                    
                per_run_stats_df = pd.DataFrame(per_run_metrics)
                
                if not per_run_stats_df.empty:
                    model_agg_stats = per_run_stats_df.groupby('Model', observed=True).agg(
                        Avg_UniqueGoldActionsCovered_PerRun=('UniqueGoldActionsCoveredInRun', 'mean'),
                        Max_UniqueGoldActionsCovered_InASingleRun=('UniqueGoldActionsCoveredInRun', 'max'),
                        Avg_TotalMethodsGenerated_PerRun=('TotalMethodsInRun', 'mean') 
                    ).reset_index()
                    
                    model_agg_stats['BC_AvgGoldActionCoverage_PerRun_Percent'] = safe_divide(
                        model_agg_stats['Avg_UniqueGoldActionsCovered_PerRun'] * 100.0, 
                        total_gold_actions_bc, 0.0
                    )
                    model_agg_stats['Threshold'] = bc_thr
                    
                    final_threshold_df = model_agg_stats.rename(columns={
                        'Avg_UniqueGoldActionsCovered_PerRun': 'Sem_Covered_Action_Count_AvgPerRun',
                        'BC_AvgGoldActionCoverage_PerRun_Percent': 'Sem_Action_Coverage_Percent'
                    })
                    bc_model_summary_results.append(final_threshold_df[['Model', 'Threshold', 
                                                                         'Sem_Covered_Action_Count_AvgPerRun', 
                                                                         'Max_UniqueGoldActionsCovered_InASingleRun',
                                                                         'Sem_Action_Coverage_Percent']])
                else:
                    print(f"BC (thr={bc_thr}): per_run_stats_df is empty after processing runs.")

            if bc_model_summary_results:
                bc_final_df = pd.concat(bc_model_summary_results, ignore_index=True)
                bc_final_df.to_csv(REPORT_DIR / "BC_Sensitivity_GoldActionCoverage.csv", index=False)
                print(f"Saved: {REPORT_DIR / 'BC_Sensitivity_GoldActionCoverage.csv'}")
            else:
                print("No BC sensitivity (Gold Action Coverage) results generated.")
        else:
            print("BC Sensitivity (Gold Action Coverage): map_df_for_bc empty or missing required columns. Skipping.")
    else:
        print("BC Sensitivity (Gold Action Coverage): ann_df_for_bc (all methods) is empty. Skipping.")

    print("Pipeline finished successfully.")

if __name__ == '__main__':
    main()