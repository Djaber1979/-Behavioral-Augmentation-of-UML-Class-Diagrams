# fmt: off
from __future__ import annotations # Must be the first non-comment line
# fmt: on

import json
import math
import os
import re # For name splitting
import sys
import traceback
import inspect
import itertools as it

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import torch # For tensor operations

# --- NLP Imports ---
try:
    # Ensure both SentenceTransformer and cos_sim are imported
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print("Error: sentence-transformers library not found.")
    print("Please install it: pip install sentence-transformers torch")
    sys.exit(1)

# --- Configuration ---
BASELINE_JSON_FNAME = "methodless.json"
GOLD_STANDARD_MAP_FNAME = "uc_action_method_map.json" # Expected in root dir
TOP_N_CORE = 35
CLASS_NAMES = [ # Ensure this list includes all classes you want metrics for AND all classes mentioned in gold_map['assigned_class']
    "ValidationResult", "Coordinates", "Address", "TimeRange", "OpeningHours",
    "UserAccount", "UserProfile", "UserCredentials", "RoleManager",
    "ServiceRequest", "CollectionRequest", "TransportRequest", "RoutePlan", "WasteJourney", "Product",
    "SearchFilter", "CollectionPoint", "NotificationTemplate",
    "NotificationService", "ServiceProvider", "PlatformService", # Added based on gold map examples
]

# --- Directory Paths ---
JSON_INPUT_DIR = Path("JSON")
REPORTS_OUTPUT_DIR = Path("reports")
REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- NLP Model ---
NLP_MODEL_NAME = 'all-mpnet-base-v2' # Or 'all-MiniLM-L6-v2'

# --- Semantic Mapping & Coverage Config ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.5 # Threshold for semantic matches in coverage calculations

# --- Constants for Column Names ---
COL_FILE = "file"; COL_MODEL = "Model"; COL_RUN = "Run"; COL_CV = "CV"
COL_TOTAL_REDUNDANCY = "Total_Redundancy"; COL_NORM_ENTROPY = "NormalizedEntropy"
COL_UNIQUE_CORE = "Unique_Core_Methods"; COL_FINAL_SCORE = "Final_Score"
COL_TOTAL_METHODS = "Target_Class_Methods"; COL_REDUNDANCY = "Redundancy"
COL_PARAM_RICHNESS = "ParamRichness"; COL_RETURN_COMPLETENESS = "ReturnTypeCompleteness"
COL_PARAM_TYPE_COMPLETENESS = "ParamTypeCompleteness"; COL_PERCENT_UC = "Percentage_Methods_With_UC"
COL_METHODS_WITH_RETURN = "Methods_With_ReturnType_Count"; COL_PARAMS_WITH_TYPE = "Params_With_Type_Count"
COL_METHOD_NAME = "MethodName"; COL_GLOBAL_FREQ = "GlobalFrequency"
COL_METHODS_WITH_UC = "Methods_With_Any_UC"; COL_COUNT_UC_ACTION = "Count_UC_Action"
COL_COUNT_UC_ONLY = "Count_UC_Only"; COL_COUNT_ACTION_ONLY = "Count_Action_Only"
COL_COUNT_NONE = "Count_None"; COL_EXTRA_METHODS = "Extra_Methods_Count"
COL_AVG_SIM_GOLD = "Avg_Similarity_To_Gold" # Overall average semantic score per model

# --- Random Seed ---
np.random.seed(42)


# --- Helper Functions ---

def split_method_name(name):
    """Splits camelCase or snake_case names into space-separated words."""
    if not name or not isinstance(name, str): return ""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1)
    s3 = s2.replace('_', ' ')
    return ' '.join(s3.lower().split())

def model_key(filename_or_stem):
    """Extracts the model identifier by removing the '_run<N>' suffix."""
    if isinstance(filename_or_stem, Path): stem = filename_or_stem.stem
    elif isinstance(filename_or_stem, str): stem = Path(filename_or_stem).stem
    else: print(f"[model_key WARNING] Unexpected type: {type(filename_or_stem)}"); return str(filename_or_stem)
    model_name = re.sub(r"_run\d+$", "", stem); return model_name

def compare_structures(baseline_elements, enriched_elements):
    """Compares two structure element dictionaries (sets)."""
    report = {}; total_baseline=0; preserved=0; all_keys=set(baseline_elements.keys())|set(enriched_elements.keys())
    for key in sorted(list(all_keys)):
        base_set=baseline_elements.get(key,set()); enrich_set=enriched_elements.get(key,set())
        pres_count=len(base_set & enrich_set); miss_count=len(base_set - enrich_set); add_count=len(enrich_set - base_set)
        total_cat=len(base_set)
        if key in baseline_elements: total_baseline+=total_cat; preserved+=pres_count
        report[f"{key}_preserved"]=pres_count; report[f"{key}_missing"]=miss_count; report[f"{key}_added"]=add_count; report[f"{key}_total_baseline"]=total_cat
    pres_pct=round((preserved / total_baseline) * 100, 2) if total_baseline > 0 else 100.0
    report['Overall_Preservation_%']=pres_pct; report['Total_Baseline_Elements']=total_baseline; report['Total_Preserved_Elements']=preserved
    report['Total_Added_Elements']=sum(len(enriched_elements.get(k,set())-baseline_elements.get(k,set())) for k in enriched_elements)
    return report

def extract_structure_from_json(json_data):
    """Extracts structural elements into sets from the parsed JSON data."""
    elements={'packages':set(), 'enums':set(), 'enum_values':set(), 'classes':set(), 'attributes':set(), 'relationships':set()}
    if not json_data or not isinstance(json_data, dict): return elements
    elements['packages'] = set(json_data.get('packages', []))
    for enum in json_data.get('enums', []):
        name=enum.get('name');
        if name: elements['enums'].add(name); [elements['enum_values'].add(f"{name}::{v}") for v in enum.get('values',[])]
    for cls in json_data.get('classes', []):
        cls_name=cls.get('name')
        if cls_name:
            elements['classes'].add(cls_name)
            for attr in cls.get('attributes', []):
                a_name=attr.get('name'); a_type=attr.get('type')
                if a_name and a_type is not None: norm_type=' '.join(a_type.split()) if isinstance(a_type, str) else a_type; elements['attributes'].add(f"{cls_name}::{a_name}: {norm_type}")
    for rel in json_data.get('relationships', []):
        src=rel.get('source'); tgt=rel.get('target')
        if src and tgt:
            sym=rel.get('type_symbol','--'); lbl=rel.get('label'); s_card=rel.get('source_cardinality'); t_card=rel.get('target_cardinality')
            rel_str=f"{src}" + (f" {s_card}" if s_card else "") + f" {sym}" + (f" {t_card}" if t_card else "") + f" {tgt}" + (f" : {lbl}" if lbl else "")
            elements['relationships'].add(rel_str)
    return elements

def safe_divide(numerator, denominator, default=np.nan):
    """
    Safely divides two numbers or pandas Series, returning default if the
    denominator is zero or NaN, or if the numerator is entirely invalid.
    """
    # Convert numerator to numeric, coercing errors
    num_val = pd.to_numeric(numerator, errors='coerce')

    # --- CORRECTED NUMERATOR CHECK ---
    # Check if the numerator is a Series and if ALL its values are NaN after coercion
    if isinstance(num_val, pd.Series) and num_val.isnull().all():
        # If the entire numerator column is invalid, return a Series of the default value
        print(f"  [WARN] safe_divide: Numerator column resulted in all NaNs.")
        return pd.Series(default, index=num_val.index)
    # Check if the numerator is a single scalar NaN
    elif not isinstance(num_val, pd.Series) and pd.isna(num_val):
         return default
    # --- END CORRECTION ---

    # Denominator handling remains the same
    if isinstance(denominator, (pd.Series, np.ndarray)):
        den_num = pd.to_numeric(denominator, errors='coerce')
        # Use np.where for element-wise conditional division
        return np.where(den_num.notna() & (den_num != 0), num_val / den_num, default)
    else: # Assume single numbers for denominator
        den_num = pd.to_numeric(denominator, errors='coerce')
        # Check if denominator is valid and non-zero
        if pd.notna(den_num) and den_num != 0:
            # Numerator could still be a Series here if denominator was scalar
            # Division of Series by scalar works element-wise
            return num_val / den_num
        else:
            # If denominator is invalid/zero, return default
            # If numerator was a Series, return a Series of defaults
            if isinstance(num_val, pd.Series):
                 return pd.Series(default, index=num_val.index)
            else:
                 return default
            
def _preprocess_gold_map(gold_map):
    """Helper to extract structured info from the gold map, including action counts per UC."""
    if not gold_map or not isinstance(gold_map, dict): print("  [ERROR] _preprocess_gold_map: Invalid gold_map input."); return None
    preprocessed = {'all_uc_ids': set(), 'all_actions': set(), 'uc_to_actions': defaultdict(set), 'action_to_details': {}, 'class_to_uc_ids': defaultdict(set), 'uc_action_counts': Counter()}
    invalid_entries = 0
    for uc_id, entries in gold_map.items():
        uc_id_str = str(uc_id); preprocessed['all_uc_ids'].add(uc_id_str); uc_action_count = 0
        if isinstance(entries, list):
            for entry in entries:
                action = entry.get("action"); assigned_class = entry.get("assigned_class")
                if action and isinstance(action, str) and assigned_class and isinstance(assigned_class, str):
                    preprocessed['all_actions'].add(action); preprocessed['uc_to_actions'][uc_id_str].add(action);
                    preprocessed['action_to_details'][action] = {'uc_id': uc_id_str, 'assigned_class': assigned_class}
                    preprocessed['class_to_uc_ids'][assigned_class].add(uc_id_str); uc_action_count += 1
                else: invalid_entries += 1
        else: invalid_entries += 1
        preprocessed['uc_action_counts'][uc_id_str] = uc_action_count # Store count
    if invalid_entries > 0: print(f"  [WARN] _preprocess_gold_map: Found {invalid_entries} invalid entries.")
    if not preprocessed['all_actions'] or not preprocessed['all_uc_ids']: print("  [ERROR] _preprocess_gold_map: No valid actions or UCs found."); return None
    preprocessed['total_actions'] = len(preprocessed['all_actions']); preprocessed['total_ucs'] = len(preprocessed['all_uc_ids'])
    print(f"  ✔ Preprocessed gold map: Found {preprocessed['total_actions']} actions across {preprocessed['total_ucs']} UCs."); return preprocessed

# --- Metric Cache Class ---
class MetricCache:
    """Caches data parsed from JSON files for metric calculation."""
    def __init__(self, json_files_dir: Path, baseline_json_fname: str, class_names: list[str]):
        self.json_files_dir = json_files_dir; self.baseline_fname = baseline_json_fname
        self.class_names = class_names; self.target_class_set = set(self.class_names)
        self.baseline_json_data = None; baseline_path = self.json_files_dir/self.baseline_fname
        if baseline_path.is_file():
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f: self.baseline_json_data=json.load(f)
            except Exception as e: print(f"❌ Error loading baseline JSON '{baseline_path}': {e}.")
        else: print(f"Warning: Baseline JSON not found: {baseline_path}")

        self.files_paths = sorted([p for p in self.json_files_dir.glob("*.json") if p.name!=self.baseline_fname], key=lambda p:p.name)
        self.files = [p.name for p in self.files_paths]
        if not self.files: print(f"⚠️⚠️⚠️ WARNING: No generated JSON files found in '{self.json_files_dir}'. Most calculations will be skipped."); self.json_data = {}; self.files_paths = []
        else:
            self.json_data = {}; loaded = []
            print(f"Attempting to load {len(self.files_paths)} generated JSON files...")
            for p in self.files_paths:
                try:
                    with open(p,'r',encoding='utf-8') as f: self.json_data[p.stem]=json.load(f); loaded.append(p.name)
                except Exception as e: print(f"❌ Error loading generated JSON {p.name}: {e}. Skipping.")
            self.files = sorted(loaded); self.files_paths=[self.json_files_dir/f for f in self.files]
            if not self.files: print("⚠️⚠️⚠️ WARNING: No generated JSON files were successfully loaded. Most calculations will be skipped.")

        self.file_info={}
        for f in self.files:
            stem=Path(f).stem;
            if stem in self.json_data: m=model_key(stem); r_m=re.search(r'_run(\d+)', stem); run=r_m.group(1) if r_m else '1'; self.file_info[f]={'stem':stem, 'model':m, 'run':run}

        self.structure_reports = {}; self.baseline_structure = {}; self.detailed_methods = {}
        self.class_detailed_methods = {}; self.extra_method_counts = {}
        self._global_method_details_list = []; self._global_method_counter = Counter()
        self._global_uc_counter = Counter(); self._global_action_counter = Counter(); self._global_unique_method_names = set()
        self.method_annot = pd.DataFrame(); self.mapping_df = pd.DataFrame(); self.annot_action_matches = {} # Init extra attrs

        if self.files:
            try: self._parse(); self._post_parse_aggregations()
            except Exception as e: print(f"❌❌❌ CRITICAL ERROR during cache parsing/aggregation: {e}"); traceback.print_exc()

    def _parse(self):
        print("Parsing baseline structure...")
        if self.baseline_json_data: self.baseline_structure = extract_structure_from_json(self.baseline_json_data)
        else: self.baseline_structure = extract_structure_from_json(None); print("Warning: Baseline structure is empty.")
        print(f"Parsing {len(self.files)} successfully loaded generated JSON files...")
        if not self.json_data: print("Warning: No generated JSON data available to parse."); return
        for json_filename in self.files:
            stem = self.file_info.get(json_filename, {}).get('stem')
            if not stem or stem not in self.json_data: print(f"Warning: Skipping file '{json_filename}' - stem/data mismatch."); continue
            file_json_data = self.json_data[stem]
            try: generated_structure = extract_structure_from_json(file_json_data)
            except Exception as e: print(f"  [ERROR] Structure extraction failed for {json_filename}: {e}"); generated_structure = {}
            if self.baseline_structure:
                try: self.structure_reports[json_filename] = compare_structures(self.baseline_structure, generated_structure)
                except Exception as e: print(f"  [ERROR] Structure comparison failed for {json_filename}: {e}"); self.structure_reports[json_filename] = {}
            else: self.structure_reports[json_filename] = {}
            file_all_methods = []; file_target_class_methods = defaultdict(list); extra_methods_in_file = 0
            try:
                for class_info in file_json_data.get("classes", []):
                    class_name = class_info.get("name");
                    if not class_name: continue
                    is_target = class_name in self.target_class_set
                    for m_info in class_info.get("methods", []):
                        m_details = m_info.copy(); m_details.setdefault("name", None); m_details.setdefault("parameters", []); m_details.setdefault("return_type", "void"); m_details.setdefault("annotation", {}); m_details.setdefault("signature", ""); m_details.setdefault("visibility", "+")
                        if not m_details.get("name"): print(f"  [WARN] Skipping method without name in {json_filename}, class {class_name}"); continue
                        m_details[COL_FILE] = json_filename; m_details["class"] = class_name
                        params = m_details.get("parameters", []); m_details["param_count"] = len(params)
                        m_details["params_str"] = ", ".join([f"{p.get('name','?')}:{p.get('type','?')}" for p in params])
                        ret_type = m_details.get("return_type"); m_details["has_return_type"] = bool(ret_type and ret_type.lower() != "void")
                        annot = m_details.get("annotation"); m_details["has_uc_annotation"] = bool(annot and annot.get("uc_references"))
                        raw_ucs = annot.get("uc_references", []); m_details["ucs"] = [str(uc) for uc in raw_ucs if uc] if isinstance(raw_ucs, list) else ([str(raw_ucs)] if raw_ucs else [])
                        m_details["action"] = annot.get("uc_action", "")
                        m_details["full_sig"] = m_details["signature"]
                        file_all_methods.append(m_details)
                        if is_target: file_target_class_methods[class_name].append(m_details)
                        else: extra_methods_in_file += 1
            except Exception as e: print(f"  [ERROR] Method extraction failed for {json_filename}: {e}"); traceback.print_exc(); file_all_methods = []; file_target_class_methods = defaultdict(list); extra_methods_in_file = 0
            self.detailed_methods[json_filename] = file_all_methods; self.class_detailed_methods[json_filename] = dict(file_target_class_methods); self.extra_method_counts[json_filename] = extra_methods_in_file

# Inside the MetricCache class definition

    def _post_parse_aggregations(self):
        """ Aggregate global counts after parsing all files. """
        print("Aggregating global metrics...")
        temp_list = []
        processed_files_count = 0
        total_methods_in_detailed = 0

        # Check detailed_methods exists and is a dict before iterating
        if hasattr(self, 'detailed_methods') and isinstance(self.detailed_methods, dict):
            print(f"DEBUG: Processing self.detailed_methods with {len(self.detailed_methods)} file entries.") # Check how many files are keys
            for file_key, m_list in self.detailed_methods.items():
                processed_files_count += 1
                if isinstance(m_list, list):
                    total_methods_in_detailed += len(m_list)
                    for m_detail in m_list:
                        # Add stricter checks for expected keys before appending
                        # Ensure 'name' exists and is not None/empty for the global list intended for counting names
                        if isinstance(m_detail, dict) and all(k in m_detail for k in [COL_FILE, 'name', 'class']) and m_detail.get('name'):
                            temp_list.append(m_detail)
                        # else: # Optional: Log skipped details
                        #    print(f"DEBUG: Skipping detail in aggregation: {m_detail}")
                # else: # Optional: Log invalid file entries
                #    print(f"DEBUG: Invalid entry type for file '{file_key}' in detailed_methods: {type(m_list)}")

        self._global_method_details_list = temp_list
        print(f"DEBUG: Processed {processed_files_count} files from detailed_methods.")
        print(f"DEBUG: Found {total_methods_in_detailed} total methods in detailed_methods values.")
        print(f"DEBUG: Populated _global_method_details_list with {len(self._global_method_details_list)} valid method details.") # Check this count

        # --- MODIFIED SECTION FOR COUNTER CREATION & DEBUGGING ---
        if self._global_method_details_list:
            # Explicitly extract valid names first
            valid_names = [m.get('name') for m in self._global_method_details_list if isinstance(m.get('name'), str) and m.get('name')] # Get only non-empty string names
            print(f"DEBUG: Number of valid, non-empty string names found: {len(valid_names)}")

            if not valid_names:
                print("DEBUG: ALL method names in global list are missing, None, or empty strings!")
                self._global_method_counter = Counter() # Explicitly set empty counter
            else:
                self._global_method_counter = Counter(valid_names) # Count only valid names
                print(f"DEBUG: Global Method Counter size: {len(self._global_method_counter)}") # Check size HERE
                if self._global_method_counter:
                    print(f"DEBUG: Most common methods in counter: {self._global_method_counter.most_common(5)}")
                else:
                    # This case should theoretically not be reached if valid_names is not empty
                    print("DEBUG: Global Method Counter IS EMPTY after creation despite valid names list.")

            # Other counters (should be less problematic but use .get() for safety)
            self._global_uc_counter = Counter(uc for m in self._global_method_details_list for uc in m.get('ucs', []) if uc) # Ensure UC is not empty/None
            self._global_action_counter = Counter(m['action'] for m in self._global_method_details_list if m.get('action')) # Already checks if action exists/is truthy
            self._global_unique_method_names = set(self._global_method_counter.keys())

            # This message confirms the size of the list used *before* counting
            print(f"Aggregated {len(self._global_method_details_list)} methods globally.")
        else:
            # This block executes if _global_method_details_list was empty
            print("Warning: No global method details found for aggregation.")
            self._global_method_counter = Counter()
            self._global_uc_counter = Counter()
            self._global_action_counter = Counter()
            self._global_unique_method_names = set()
        # --- END MODIFIED SECTION ---

    # --- Public Methods --- (Robust versions)
    def get_structure_reports_df(self):
        if not hasattr(self, 'structure_reports') or not self.structure_reports: print("Warning: No structure reports available."); return pd.DataFrame()
        reports = [self.structure_reports.get(f, {}) for f in self.files];
        if not any(reports): return pd.DataFrame()
        df = pd.DataFrame(reports).fillna(0); df[COL_FILE] = self.files
        df[COL_MODEL] = df[COL_FILE].map(lambda f: self.file_info.get(f, {}).get('model')); df[COL_RUN] = df[COL_FILE].map(lambda f: self.file_info.get(f, {}).get('run'))
        if COL_MODEL not in df.columns: df[COL_MODEL] = None;
        if COL_RUN not in df.columns: df[COL_RUN] = None;
        for c in df.columns:
            if any(k in c for k in ['preserved','missing','added','_total_baseline','Total_Baseline','Total_Preserved','Total_Added']):
                 try: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                 except: pass
        cols = [COL_FILE, COL_MODEL, COL_RUN] + sorted([c for c in df.columns if c not in [COL_FILE, COL_MODEL, COL_RUN]]); cols = [c for c in cols if c in df.columns]
        return df[cols] if cols else pd.DataFrame()

    def get_per_file_method_metrics(self):
        if not hasattr(self, 'class_detailed_methods') or not self.class_detailed_methods: print("Warning: Target class methods not parsed, cannot calculate per-file metrics."); return pd.DataFrame()
        if not hasattr(self, 'extra_method_counts'): self.extra_method_counts = {}
        rows = []
        for fname in self.files:
            methods = [m for m_list in self.class_detailed_methods.get(fname, {}).values() for m in m_list]; total_target = len(methods); extra = self.extra_method_counts.get(fname, 0)
            row = {COL_FILE: fname, COL_TOTAL_METHODS: total_target, COL_EXTRA_METHODS: extra}
            if total_target == 0:
                 row.update({"Unique_Method_Names": 0, COL_REDUNDANCY: np.nan, "DuplicatedOccurrences_Name": 0, "DuplicatedOccurrences_NameParam": 0, "DuplicatedOccurrences_FullSig": 0, COL_PARAM_RICHNESS: 0.0, COL_RETURN_COMPLETENESS: 0.0, COL_METHODS_WITH_RETURN: 0, "Visibility_Public": 0, "Visibility_Private": 0, "Visibility_Protected": 0, "Visibility_Package": 0, COL_METHODS_WITH_UC: 0, COL_PERCENT_UC: 0.0, "Total_UC_References_File": 0, "Unique_UCs_File": 0, "Unique_Actions_File": 0, COL_COUNT_UC_ACTION: 0, COL_COUNT_UC_ONLY: 0, COL_COUNT_ACTION_ONLY: 0, COL_COUNT_NONE: 0, COL_PARAMS_WITH_TYPE: 0, "Total_Params_Count": 0, COL_PARAM_TYPE_COMPLETENESS: np.nan})
            else:
                names=[m.get('name','') for m in methods]; name_params=[f"{m.get('name','')}({m.get('params_str','')})" for m in methods]; sigs=[m.get('full_sig','') for m in methods]; name_ctr=Counter(names); param_ctr=Counter(name_params); sig_ctr=Counter(sigs); unique_names=set(names)
                row["DuplicatedOccurrences_Name"]=sum(c-1 for c in name_ctr.values() if c>1); row["DuplicatedOccurrences_NameParam"]=sum(c-1 for c in param_ctr.values() if c>1); row["DuplicatedOccurrences_FullSig"]=sum(c-1 for c in sig_ctr.values() if c>1)
                row["Unique_Method_Names"]=len(unique_names); row[COL_REDUNDANCY]=safe_divide(total_target,len(unique_names)); p_counts=[m.get('param_count',0) for m in methods]; row[COL_PARAM_RICHNESS]=np.mean(p_counts) if p_counts else 0.0
                m_ret=sum(m.get('has_return_type',False) for m in methods); row[COL_METHODS_WITH_RETURN]=m_ret; row[COL_RETURN_COMPLETENESS]=safe_divide(m_ret, total_target, default=0.0)
                vis=Counter(m.get('visibility','+') for m in methods); row.update({"Visibility_Public":vis.get('+',0),"Visibility_Private":vis.get('-',0),"Visibility_Protected":vis.get('#',0),"Visibility_Package":vis.get('~',0)})
                c_uc_act,c_uc_only,c_act_only,c_none,t_uc_refs,m_any_uc=0,0,0,0,0,0; u_ucs,u_acts=set(),set()
                for m in methods:
                    has_ucs=bool(m.get('ucs')); has_action=bool(m.get('action'))
                    if has_ucs and has_action: c_uc_act+=1
                    elif has_ucs: c_uc_only+=1
                    elif has_action: c_act_only+=1
                    else: c_none+=1
                    if has_ucs: m_any_uc+=1; ucs=m.get('ucs',[]); t_uc_refs+=len(ucs); u_ucs.update(ucs)
                    if has_action: u_acts.add(m['action'])
                row.update({COL_METHODS_WITH_UC:m_any_uc, COL_PERCENT_UC:safe_divide(m_any_uc, total_target, default=0.0), COL_COUNT_UC_ACTION:c_uc_act, COL_COUNT_UC_ONLY:c_uc_only, COL_COUNT_ACTION_ONLY:c_act_only, COL_COUNT_NONE:c_none, "Total_UC_References_File":t_uc_refs, "Unique_UCs_File":len(u_ucs), "Unique_Actions_File":len(u_acts)})
                p_w_type=sum(len([p for p in m.get('parameters',[]) if p.get('type')]) for m in methods); t_params=sum(m.get('param_count',0) for m in methods)
                row[COL_PARAMS_WITH_TYPE]=p_w_type; row["Total_Params_Count"]=t_params; row[COL_PARAM_TYPE_COMPLETENESS]=safe_divide(p_w_type, t_params)
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
             df[COL_MODEL]=df[COL_FILE].map(lambda f:self.file_info.get(f,{}).get('model')); df[COL_RUN]=df[COL_FILE].map(lambda f:self.file_info.get(f,{}).get('run'))
             if COL_MODEL not in df.columns: df[COL_MODEL] = None;
             if COL_RUN not in df.columns: df[COL_RUN] = None;
             base_cols=[COL_FILE,COL_MODEL,COL_RUN]; metric_cols=sorted([c for c in df.columns if c not in base_cols]); cols=base_cols+metric_cols; cols = [c for c in cols if c in df.columns]
             df=df[cols]
        return df

    def get_global_method_counter(self): return getattr(self, '_global_method_counter', Counter())
    def get_global_uc_counter(self): return getattr(self, '_global_uc_counter', Counter())
    def get_global_action_counter(self): return getattr(self, '_global_action_counter', Counter())
    def get_core_methods_list(self, top_n): counter = self.get_global_method_counter(); return [m for m, _ in counter.most_common(top_n)] if counter else []
    def get_exclusive_methods_list(self):
        excl = defaultdict(list)
        # Use getters which handle checks internally
        ctr = self.get_global_method_counter()
        details = self.get_method_details_list()
        info = getattr(self, 'file_info', {}) # Use getattr for safety

        # Check prerequisites
        if not ctr or not details or not info:
             print("Warning: Cannot generate exclusive methods list, prerequisite data missing.")
             return {}

        for m in details:
             m_name = m.get('name')
             m_file = m.get(COL_FILE)

             # Check 1: Is this method exclusive (appears only once globally)?
             if m_name and m_file and ctr.get(m_name, 0) == 1:
                 # Check 2: Can we find the model associated with this exclusive method's file?
                 model = info.get(m_file, {}).get('model')
                 # Check 3: If we found a valid model name...
                 if model:
                     # ...then add this method name to that model's list of exclusives.
                     excl[model].append(m_name)
                 # else: # Optional: Warn if exclusive method's file has no model info
                 #    print(f"  [WARN] Exclusive method '{m_name}' found in file '{m_file}' but could not determine model.")

        return dict(excl)
    def get_method_details_list(self): return getattr(self, '_global_method_details_list', [])
    def get_class_detailed_methods(self): return getattr(self, 'class_detailed_methods', {})
    def get_all_files(self): return getattr(self, 'files', [])

# --- End of MetricCache Class ---


# --- Metric Calculation Functions ---

# --- Standard Metrics ---
# Paste the FULL definitions for:
# calculate_counts, calculate_variability, calculate_method_metrics_summary,
# calculate_diversity, calculate_coverage, calculate_overlap,
# calculate_per_class_overlap, calculate_annotation_metrics,
# calculate_added_class_llm_counts
# ... (Definitions omitted for brevity, assume they are correctly pasted from previous responses) ...
# Example: Paste calculate_counts here...
def calculate_counts(cache: MetricCache):
    if not hasattr(cache, 'class_detailed_methods'): print("[calculate_counts] Error: Cache missing 'class_detailed_methods'."); return pd.DataFrame()
    sorted_files=cache.get_all_files(); class_names=cache.class_names
    if not sorted_files: print("[calculate_counts] Warning: No files loaded in cache."); return pd.DataFrame(index=class_names)
    df = pd.DataFrame(0, index=class_names, columns=sorted_files, dtype=int); df.index.name = "Classes"; file_totals = {f: 0 for f in sorted_files}; found_any = False
    class_methods_data = cache.get_class_detailed_methods()
    for fname in sorted_files:
        cls_data = class_methods_data.get(fname, {}); f_total = 0
        for cls_name, m_list in cls_data.items():
            if cls_name in df.index: count = len(m_list);
            # Minor correction: Check count > 0 before assignment AND setting found_any
            if count > 0:
                df.loc[cls_name, fname] = count;
                found_any = True;
                f_total += count # Only add to total if methods found for this class
            # Note: Original code added f_total regardless of count > 0, might inflate if empty lists exist?
            # Let's keep original logic for now: f_total += count # Add count regardless (includes 0)
        file_totals[fname] = f_total # Original logic assigns f_total here, seems correct.

    if not found_any:
        print("[calculate_counts] Warning: No target class methods found.") # Message seems correct

    if not df.columns.empty:
        totals = pd.Series(file_totals, index=df.columns, name=COL_TOTAL_METHODS).fillna(0).astype(int); df = pd.concat([df, totals.to_frame().T])
    else:
        print("[calculate_counts] Warning: DataFrame has no columns.");

    # --- CORRECTED TRY-EXCEPT BLOCK ---
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        # The 'if' condition needs to be INSIDE the try block
        if not num_cols.empty:
             # This line can also raise errors (e.g., memory, invalid types after fillna)
             df[num_cols] = df[num_cols].fillna(0).astype(int)
    except Exception as e:
        # This will now catch errors from select_dtypes OR the fillna/astype operations
        print(f"[calculate_counts] Warning: Numeric conversion/casting failed: {e}")
    # --- END CORRECTION ---

    return df

# ... PASTE ALL OTHER STANDARD CALCULATION FUNCTIONS HERE ...

# --- Semantic Mapping Function ---
def map_methods_to_actions(cache: MetricCache, gold_map_details: dict, nlp_model_name: str):
    """Maps generated methods to best-matching action using split names."""
    # (Definition from Response #11 - Returns mapping_df, embedding_map)
    print(f"🚀 Starting method-to-action mapping using model '{nlp_model_name}' (with name splitting)...")
    if not gold_map_details: print("  [ERROR] Preprocessed gold map details missing."); return pd.DataFrame(), {}
    try: model = SentenceTransformer(nlp_model_name); print(f"  ✔ Model '{nlp_model_name}' loaded. Device: {model.device}")
    except Exception as e: print(f"  [ERROR] Failed load model '{nlp_model_name}': {e}"); return pd.DataFrame(), {}
    generated_methods_details = []; unique_split_names = set()
    print("  Extracting generated method details and splitting names...")
    if not hasattr(cache, 'file_info') or not hasattr(cache, 'class_detailed_methods'): print("  [ERROR] Cache missing attributes."); return pd.DataFrame(), {}
    for fname, info in cache.file_info.items():
        model_name = info.get('model'); run = info.get('run');
        if not model_name or not run: continue
        class_methods = cache.class_detailed_methods.get(fname, {})
        for class_name, method_list in class_methods.items():
            for method in method_list:
                gen_name = method.get("name");
                if gen_name: split_name = split_method_name(gen_name)
                else: continue
                details = {COL_MODEL: model_name, COL_RUN: run, COL_FILE: fname, 'Class': class_name, COL_METHOD_NAME: gen_name, 'SplitMethodName': split_name, 'UCs': method.get('ucs', [])}
                generated_methods_details.append(details)
                if split_name: unique_split_names.add(split_name)
    if not generated_methods_details: print("  [WARN] No generated methods in target classes."); return pd.DataFrame(), {}
    print(f"  Found {len(generated_methods_details)} methods ({len(unique_split_names)} unique split names).")
    list_all_gold_actions = sorted(list(gold_map_details['all_actions']))
    strings_to_embed = list(unique_split_names | set(list_all_gold_actions))
    if not strings_to_embed: print("  [ERROR] No strings to embed."); return pd.DataFrame(), {}
    print(f"  Embedding {len(strings_to_embed)} unique strings..."); embedding_map = {}
    try: embeddings = model.encode(strings_to_embed, convert_to_tensor=True, show_progress_bar=True); embedding_map = {text: emb for text, emb in zip(strings_to_embed, embeddings)}; print("  ✔ Embeddings created.")
    except Exception as e: print(f"  [ERROR] Embedding failed: {e}"); traceback.print_exc(); return pd.DataFrame(), {}
    gold_action_embeddings = []; map_idx_to_action = {}; missing_embed_actions = []
    for idx, action in enumerate(list_all_gold_actions):
        emb = embedding_map.get(action)
        if emb is not None: gold_action_embeddings.append(emb); map_idx_to_action[len(gold_action_embeddings)-1] = action
        else: missing_embed_actions.append(action)
    if missing_embed_actions: print(f"  [WARN] Embeddings missing for {len(missing_embed_actions)} gold actions.")
    if not gold_action_embeddings: print("  [ERROR] No embeddings for gold actions."); return pd.DataFrame(), embedding_map
    try: gold_embeddings_tensor = torch.stack(gold_action_embeddings).to(model.device); print(f"  Prepared {gold_embeddings_tensor.shape[0]} gold action embeddings tensor.")
    except Exception as e: print(f"  [ERROR] Failed stacking gold embeddings: {e}"); return pd.DataFrame(), embedding_map
    results = []; print("  Mapping methods to best matching actions..."); calculation_errors = 0; methods_processed = 0
    for method_detail in generated_methods_details:
        methods_processed += 1; split_name = method_detail['SplitMethodName']; gen_emb = embedding_map.get(split_name)
        best_match_action_text = None; max_similarity_score = -1.0;
        if gen_emb is None: results.append({**method_detail, 'Best_Match_Action': None, 'Max_Similarity_Score': np.nan}); continue
        gen_emb = gen_emb.to(gold_embeddings_tensor.device)
        try:
            all_sims = cos_sim(gen_emb, gold_embeddings_tensor).squeeze()
            if all_sims.numel() > 0:
                if all_sims.dim() == 0: score, idx = all_sims.item(), 0
                else: score_tensor, idx_tensor = torch.max(all_sims, dim=0); score, idx = score_tensor.item(), idx_tensor.item()
                max_similarity_score = score; best_match_action_text = map_idx_to_action.get(idx)
                if best_match_action_text is None: max_similarity_score = -1.0
        except Exception as e: print(f"  [ERROR] Cosine sim failed for '{method_detail[COL_METHOD_NAME]}' (split: '{split_name}'): {e}"); calculation_errors += 1; max_similarity_score = np.nan
        results.append({**method_detail, 'Best_Match_Action': best_match_action_text, 'Max_Similarity_Score': max_similarity_score if max_similarity_score > -1.0 else np.nan})
    print(f"  ✔ Mapping finished. Processed {methods_processed} methods, {calculation_errors} errors.")
    if not results: print("  [WARN] No mapping results generated."); return pd.DataFrame(), embedding_map
    output_df = pd.DataFrame(results)
    cols_order = [COL_MODEL, COL_RUN, COL_FILE, 'Class', COL_METHOD_NAME, 'SplitMethodName', 'UCs', 'Best_Match_Action', 'Max_Similarity_Score']
    output_df = output_df[[col for col in cols_order if col in output_df.columns]]
    return output_df, embedding_map # Return embeddings map

# --- Coverage Functions ---

def precompute_annotated_action_matches(method_annot_df, gold_map_details, embedding_map, nlp_model_name, similarity_threshold):
    """ Precomputes semantic matches for unique annotated action strings against gold actions."""
    # (Definition from Response #11)
    print("  Pre-calculating semantic matches for annotated action strings...")
    if method_annot_df.empty or not gold_map_details or embedding_map is None: print("  [WARN] Skipping precomputation for annotated actions: Missing inputs."); return {}
    try: model = SentenceTransformer(nlp_model_name)
    except Exception as e: print(f"  [ERROR] Failed load model '{nlp_model_name}': {e}"); return {}
    required_cols = [COL_MODEL, 'UC_Action'];
    if not all(col in method_annot_df.columns for col in required_cols): print(f"  [ERROR] Annot report missing: {required_cols}"); return {}
    unique_annot_actions = {a for a in method_annot_df['UC_Action'].dropna().unique() if isinstance(a, str) and a}
    if not unique_annot_actions: print("  No non-empty annotated actions found."); return {}
    list_all_gold_actions = sorted(list(gold_map_details['all_actions']))
    gold_action_embeddings = []; map_idx_to_action = {};
    for idx, action in enumerate(list_all_gold_actions):
        emb = embedding_map.get(action)
        if emb is not None: gold_action_embeddings.append(emb); map_idx_to_action[len(gold_action_embeddings)-1] = action
    if not gold_action_embeddings: print("  [ERROR] No embeddings for gold actions."); return {}
    gold_tensor = torch.stack(gold_action_embeddings).to(model.device)
    annot_action_matches = defaultdict(list)
    for annot_action in unique_annot_actions:
        annot_emb = embedding_map.get(annot_action);
        if annot_emb is None: continue
        annot_emb = annot_emb.to(model.device)
        try:
            sims = cos_sim(annot_emb, gold_tensor).squeeze()
            if sims.numel() > 0:
                indices = torch.where(sims >= similarity_threshold)[0]
                for idx in indices.tolist(): matched_gold = map_idx_to_action.get(idx);
                if matched_gold: annot_action_matches[annot_action].append(matched_gold)
        except Exception as e: print(f"  [WARN] Sim calc failed for annot action '{annot_action}': {e}")
    print(f"  ✔ Pre-calculated matches for {len(annot_action_matches)} unique annotated actions.")
    return annot_action_matches


def calculate_action_annotation_coverage(method_annot_df, gold_map_details, annot_action_matches):
    """Calculates Action coverage via semantic match on ANNOTATED action text, respecting class."""
    # (Definition from Response #11 - Modified to return detailed sets)
    print(f"📊 Calculating Annotation-Based Action Coverage (Class-Aware)...")
    results = []; covered_actions_by_model = defaultdict(set)
    if method_annot_df.empty or not gold_map_details or annot_action_matches is None: print("  ℹ Skipping action annot coverage: Input missing."); return pd.DataFrame(), covered_actions_by_model
    total_gold_actions = gold_map_details['total_actions']; action_to_details = gold_map_details['action_to_details']
    if total_gold_actions == 0: print("  [ERROR] No gold actions found."); return pd.DataFrame(), covered_actions_by_model
    required_cols = [COL_MODEL, 'Class', 'UC_Action'];
    if not all(col in method_annot_df.columns for col in required_cols): print(f"  [ERROR] Annot report missing: {required_cols}"); return pd.DataFrame(), covered_actions_by_model

    all_models = method_annot_df[COL_MODEL].unique()
    uc_action_coverage = defaultdict(lambda: defaultdict(set))
    for model_name, group in method_annot_df.groupby(COL_MODEL):
        model_set = set()
        for _, row in group.iterrows():
            annot_action = row['UC_Action']; gen_class = row['Class']
            if annot_action in annot_action_matches:
                for matched_gold in annot_action_matches[annot_action]:
                    gold_details = action_to_details.get(matched_gold)
                    if gold_details and gold_details['assigned_class'] == gen_class:
                        model_set.add(matched_gold); uc_action_coverage[model_name][gold_details['uc_id']].add(matched_gold)
        covered_actions_by_model[model_name] = model_set # Store the set

    for model_name in all_models:
        model_set = covered_actions_by_model.get(model_name, set())
        action_cov_pct = safe_divide(len(model_set), total_gold_actions, default=0.0) * 100
        avg_per_uc_coverage = 0.0; uc_coverages = []
        uc_to_actions_map = gold_map_details['uc_to_actions']; uc_action_counts = gold_map_details['uc_action_counts']
        for uc_id in gold_map_details['all_uc_ids']:
            total_actions_in_uc = uc_action_counts.get(uc_id, 0) # Get count for this UC
            if total_actions_in_uc == 0: continue # Skip UCs with no actions
            matched_in_this_uc = uc_action_coverage[model_name].get(uc_id, set())
            # Ensure we only count actions relevant to this specific UC for the denominator
            actions_in_this_uc = uc_to_actions_map.get(uc_id, set())
            relevant_matched_count = len(matched_in_this_uc & actions_in_this_uc)
            uc_coverage = safe_divide(relevant_matched_count, total_actions_in_uc, default=0.0)
            uc_coverages.append(uc_coverage)
        if uc_coverages: avg_per_uc_coverage = np.mean(uc_coverages) * 100
        results.append({COL_MODEL: model_name, 'Annot_Covered_Action_Count': len(model_set), 'Annot_Action_Coverage_Percent': action_cov_pct, 'Avg_Per_UC_Annot_Action_Coverage': avg_per_uc_coverage})

    print("  ✔ Action Annotation Coverage calculated.")
    # Return summary df AND detailed sets per model
    return pd.DataFrame(results), covered_actions_by_model


def calculate_uc_annotation_coverage(method_annot_df, gold_map_details):
    """Calculates Avg % Action Coverage per UC based SOLELY on explicit UC_References annotations matching actions within those UCs."""
    print("📊 Calculating Annotation-Based UC Coverage (Avg Action % per UC)...")
    results = []; covered_ucs_sets_by_model = defaultdict(set) # Store sets of UCs HIT AT ALL
    covered_actions_per_uc_by_model = defaultdict(lambda: defaultdict(set)) # Store actions covered within each UC per model

    if method_annot_df.empty or not gold_map_details: print("  ℹ Skipping UC annotation coverage: Input missing."); return pd.DataFrame(), covered_ucs_sets_by_model
    all_gold_ucs = gold_map_details['all_uc_ids']; total_gold_ucs = gold_map_details['total_ucs']
    if total_gold_ucs == 0: print("  [ERROR] No UCs in gold map."); return pd.DataFrame(), covered_ucs_sets_by_model
    required_cols = [COL_MODEL, 'UC_References', 'UC_Action', 'Class'] # Need Action and Class to check action validity
    if not all(col in method_annot_df.columns for col in required_cols): print(f"  [ERROR] Annot report missing: {required_cols}"); return pd.DataFrame(), covered_ucs_sets_by_model

    action_to_details = gold_map_details['action_to_details']

    all_models = method_annot_df[COL_MODEL].unique()
    for model_name, group in method_annot_df.groupby(COL_MODEL):
        model_uc_set = set() # UCs hit by this model
        model_action_uc_set = defaultdict(set) # Actions hit per UC by this model
        for _, row in group.iterrows():
            annot_ucs = row['UC_References']; annot_action = row['UC_Action']; gen_class = row['Class']
            if isinstance(annot_ucs, list):
                 for uc_ref in annot_ucs:
                     if uc_ref in all_gold_ucs:
                         model_uc_set.add(uc_ref) # Mark UC as hit by annotation
                         # Check if the annotated action is valid for THIS annotated UC and class
                         if annot_action:
                             gold_details = action_to_details.get(annot_action)
                             if gold_details and gold_details['uc_id'] == uc_ref and gold_details['assigned_class'] == gen_class:
                                 model_action_uc_set[uc_ref].add(annot_action)

        covered_ucs_sets_by_model[model_name] = model_uc_set # Store UCs hit

        # Calculate Avg Per UC Action Coverage
        uc_coverage_scores = []
        uc_action_counts = gold_map_details['uc_action_counts']
        for uc_id in all_gold_ucs:
            total_actions_in_uc = uc_action_counts.get(uc_id, 0)
            if total_actions_in_uc == 0: continue
            matched_actions_in_uc = model_action_uc_set.get(uc_id, set())
            coverage_percent_for_this_uc = safe_divide(len(matched_actions_in_uc), total_actions_in_uc, default=0.0) * 100
            uc_coverage_scores.append(coverage_percent_for_this_uc)

        avg_per_uc_coverage = np.mean(uc_coverage_scores) if uc_coverage_scores else 0.0
        # Simple % of UCs hit at least once
        uc_hit_coverage_percent = safe_divide(len(model_uc_set), total_gold_ucs, default=0.0) * 100

        results.append({
            COL_MODEL: model_name,
            'Annot_Covered_UC_Count': len(model_uc_set), # How many UCs were mentioned
            'Annot_UC_Hit_Coverage_Percent': uc_hit_coverage_percent, # % of UCs mentioned
            'Avg_Per_UC_Annot_Action_Coverage': avg_per_uc_coverage # Avg action coverage within UCs
        })
        

    print("  ✔ UC Annotation Coverage (Avg Action % per UC) calculated.")
    # Return summary df AND detailed sets of UCs HIT by model
    return pd.DataFrame(results), covered_ucs_sets_by_model


def calculate_action_semantic_coverage(mapping_df, gold_map_details, similarity_threshold):
    """Calculates Action coverage based on semantic match score, respecting class."""
    # (Definition from Response #11 - Modified to return detailed sets and per-UC avg)
    print(f"📊 Calculating Semantic-Based Action Coverage (Threshold: {similarity_threshold}, Class-Aware)...")
    results = []; covered_actions_by_model = defaultdict(set)
    if mapping_df.empty or not gold_map_details: print("  ℹ Skipping semantic action coverage: Input missing."); return pd.DataFrame(), covered_actions_by_model
    total_gold_actions = gold_map_details['total_actions']; action_to_details = gold_map_details['action_to_details']
    if total_gold_actions == 0: print("  [ERROR] No actions in gold map."); return pd.DataFrame(), covered_actions_by_model
    required_cols = [COL_MODEL, 'Class', 'Best_Match_Action', 'Max_Similarity_Score']
    if not all(col in mapping_df.columns for col in required_cols): print(f"  [ERROR] Mapping report missing: {required_cols}"); return pd.DataFrame(), covered_actions_by_model
    mapping_df['Score_Numeric'] = pd.to_numeric(mapping_df['Max_Similarity_Score'], errors='coerce')

    all_models = mapping_df[COL_MODEL].unique()
    uc_action_coverage = defaultdict(lambda: defaultdict(set))
    for model_name, group in mapping_df.groupby(COL_MODEL):
        model_set = set()
        good_matches = group[group['Score_Numeric'] >= similarity_threshold]
        for _, row in good_matches.iterrows():
            best_action = row['Best_Match_Action']; gen_class = row['Class']; gold_details = action_to_details.get(best_action)
            if gold_details and gold_details['assigned_class'] == gen_class:
                model_set.add(best_action); uc_action_coverage[model_name][gold_details['uc_id']].add(best_action)
        covered_actions_by_model[model_name] = model_set

    for model_name in all_models:
        model_set = covered_actions_by_model.get(model_name, set())
        action_cov_pct = safe_divide(len(model_set), total_gold_actions, default=0.0) * 100
        avg_per_uc_coverage = 0.0; uc_coverages = []
        uc_to_actions_map = gold_map_details['uc_to_actions']; uc_action_counts = gold_map_details['uc_action_counts']
        for uc_id in gold_map_details['all_uc_ids']:
            total_actions_in_uc = uc_action_counts.get(uc_id, 0)
            if total_actions_in_uc == 0 : continue
            actions_in_this_uc = uc_to_actions_map.get(uc_id, set())
            matched_in_this_uc = uc_action_coverage[model_name].get(uc_id, set()) & actions_in_this_uc
            uc_coverage = safe_divide(len(matched_in_this_uc), total_actions_in_uc, default=0.0)
            uc_coverages.append(uc_coverage)
        if uc_coverages: avg_per_uc_coverage = np.mean(uc_coverages) * 100
        results.append({COL_MODEL: model_name, 'Sem_Covered_Action_Count': len(model_set), 'Sem_Action_Coverage_Percent': action_cov_pct, 'Avg_Per_UC_Sem_Action_Coverage': avg_per_uc_coverage})

    print("  ✔ Action Semantic Coverage calculated.")
    return pd.DataFrame(results), covered_actions_by_model # Return sets


def calculate_uc_semantic_coverage(mapping_df, gold_map_details, similarity_threshold):
    """Calculates Avg % Action Coverage per UC based on semantic matches."""
    # This function now calculates the same per-UC average as calculate_action_semantic_coverage
    # It can be simplified or merged if only the average is needed.
    # Let's keep it distinct for now, calculating the simple % of UCs hit AND the avg action % per UC.
    print(f"📊 Calculating Semantic-Based UC Coverage (Avg Action % per UC, Threshold: {similarity_threshold})...")
    results = []; covered_ucs_sets_by_model = defaultdict(set) # Store UCs hit
    covered_actions_per_uc_by_model = defaultdict(lambda: defaultdict(set)) # Store actions covered per UC

    if mapping_df.empty or not gold_map_details: print("  ℹ Skipping semantic UC coverage: Input missing."); return pd.DataFrame(), covered_ucs_sets_by_model
    all_gold_ucs = gold_map_details['all_uc_ids']; total_gold_ucs = gold_map_details['total_ucs']
    action_to_details = gold_map_details['action_to_details']
    if total_gold_ucs == 0: print("  [ERROR] No UCs in gold map."); return pd.DataFrame(), covered_ucs_sets_by_model
    required_cols = [COL_MODEL, 'Class', 'Best_Match_Action', 'Max_Similarity_Score'] # Need Class for potential future refinement
    if not all(col in mapping_df.columns for col in required_cols): print(f"  [ERROR] Mapping report missing: {required_cols}"); return pd.DataFrame(), covered_ucs_sets_by_model
    mapping_df['Score_Numeric'] = pd.to_numeric(mapping_df['Max_Similarity_Score'], errors='coerce')

    all_models = mapping_df[COL_MODEL].unique()
    for model_name, group in mapping_df.groupby(COL_MODEL):
        model_uc_set = set()
        model_action_uc_set = defaultdict(set)
        good_matches = group[group['Score_Numeric'] >= similarity_threshold]
        for _, row in good_matches.iterrows():
            best_action = row['Best_Match_Action']; gen_class = row['Class'] # Keep gen_class if needed later
            gold_details = action_to_details.get(best_action)
            if gold_details:
                # Class check is usually done in Action coverage, but keep for consistency
                # if gold_details['assigned_class'] == gen_class: # Optional class check here too?
                uc_id = gold_details['uc_id']
                model_uc_set.add(uc_id)
                model_action_uc_set[uc_id].add(best_action)

        covered_ucs_sets_by_model[model_name] = model_uc_set # Store UCs hit

        # Calculate Avg Per UC Action Coverage
        uc_coverage_scores = []
        uc_action_counts = gold_map_details['uc_action_counts']
        for uc_id in all_gold_ucs:
            total_actions_in_uc = uc_action_counts.get(uc_id, 0)
            if total_actions_in_uc == 0: continue
            matched_actions_in_uc = model_action_uc_set.get(uc_id, set())
            coverage_percent_for_this_uc = safe_divide(len(matched_actions_in_uc), total_actions_in_uc, default=0.0) * 100
            uc_coverage_scores.append(coverage_percent_for_this_uc)

        avg_per_uc_coverage = np.mean(uc_coverage_scores) if uc_coverage_scores else 0.0
        uc_hit_coverage_percent = safe_divide(len(model_uc_set), total_gold_ucs, default=0.0) * 100

        results.append({
            COL_MODEL: model_name,
            'Sem_Covered_UC_Count': len(model_uc_set), # Count of UCs hit
            'Sem_UC_Hit_Coverage_Percent': uc_hit_coverage_percent, # % of UCs hit
            'Avg_Per_UC_Sem_Action_Coverage': avg_per_uc_coverage # Avg action coverage within UCs
        })

    print("  ✔ UC Semantic Coverage (Avg Action % per UC) calculated.")
    return pd.DataFrame(results), covered_ucs_sets_by_model


def calculate_action_combined_coverage(annot_action_sets, sem_action_sets, gold_map_details):
    """Combines Action coverage results from annotation and semantic sets."""
    # (Definition from Response #11 - Modified to use detailed sets)
    print("📊 Calculating Combined Action Coverage...")
    results = []; combined_actions_by_model = defaultdict(set)
    if annot_action_sets is None or sem_action_sets is None or not gold_map_details: print("  ℹ Skipping combined action coverage: Input missing."); return pd.DataFrame(), combined_actions_by_model
    total_gold_actions = gold_map_details['total_actions']; action_to_details = gold_map_details['action_to_details']
    if total_gold_actions == 0: print("  [ERROR] No actions in gold map."); return pd.DataFrame(), combined_actions_by_model

    all_models = set(annot_action_sets.keys()) | set(sem_action_sets.keys())
    uc_action_coverage = defaultdict(lambda: defaultdict(set))

    for model_name in sorted(list(all_models)):
        annot_set = annot_action_sets.get(model_name, set())
        sem_set = sem_action_sets.get(model_name, set())
        combined_set = annot_set | sem_set
        combined_actions_by_model[model_name] = combined_set # Store combined set

        for action in combined_set:
             gold_details = action_to_details.get(action)
             if gold_details: uc_action_coverage[model_name][gold_details['uc_id']].add(action)

        action_cov_pct = safe_divide(len(combined_set), total_gold_actions, default=0.0) * 100
        avg_per_uc_coverage = 0.0; uc_coverages = []
        uc_to_actions_map = gold_map_details['uc_to_actions']; uc_action_counts = gold_map_details['uc_action_counts']
        for uc_id in gold_map_details['all_uc_ids']:
            total_actions_in_uc = uc_action_counts.get(uc_id, 0)
            if total_actions_in_uc == 0: continue
            actions_in_this_uc = uc_to_actions_map.get(uc_id, set())
            matched_in_this_uc = uc_action_coverage[model_name].get(uc_id, set()) & actions_in_this_uc
            uc_coverage = safe_divide(len(matched_in_this_uc), total_actions_in_uc, default=0.0)
            uc_coverages.append(uc_coverage)
        if uc_coverages: avg_per_uc_coverage = np.mean(uc_coverages) * 100

        results.append({COL_MODEL: model_name, 'Comb_Covered_Action_Count': len(combined_set), 'Comb_Action_Coverage_Percent': action_cov_pct, 'Avg_Per_UC_Comb_Action_Coverage': avg_per_uc_coverage})

    print("  ✔ Combined Action Coverage calculated.")
    return pd.DataFrame(results), combined_actions_by_model


def calculate_uc_combined_coverage(annot_uc_sets, sem_uc_sets, gold_map_details):
    """Combines UC coverage results from annotation and semantic sets (simple hit coverage)."""
    # (Definition from Response #11 - Modified to use detailed sets)
    print("📊 Calculating Combined UC Coverage (Hit Rate)...")
    results = []; combined_ucs_by_model = defaultdict(set)
    if annot_uc_sets is None or sem_uc_sets is None or not gold_map_details: print("  ℹ Skipping combined UC coverage: Input missing."); return pd.DataFrame(), combined_ucs_by_model
    total_gold_ucs = gold_map_details['total_ucs']
    if total_gold_ucs == 0: print("  [ERROR] No UCs in gold map."); return pd.DataFrame(), combined_ucs_by_model

    all_models = set(annot_uc_sets.keys()) | set(sem_uc_sets.keys())

    for model_name in sorted(list(all_models)):
        annot_set = annot_uc_sets.get(model_name, set())
        sem_set = sem_uc_sets.get(model_name, set())
        combined_set = annot_set | sem_set
        combined_ucs_by_model[model_name] = combined_set # Store combined set

        uc_cov_pct = safe_divide(len(combined_set), total_gold_ucs, default=0.0) * 100
        results.append({COL_MODEL: model_name, 'Comb_Covered_UC_Count': len(combined_set), 'Comb_UC_Hit_Coverage_Percent': uc_cov_pct}) # Renamed metric

    print("  ✔ Combined UC Coverage (Hit Rate) calculated.")
    return pd.DataFrame(results), combined_ucs_by_model


# --- Reporting Functions ---

# --- Reporting Functions ---
# (Place this with other generate_* functions)

def generate_method_annotation_report(cache: MetricCache, target_classes_only=True):
    """
    Generates a detailed report listing methods and their associated UC annotation details.
    Can be filtered to include only methods from target classes.

    Args:
        cache: The MetricCache object.
        target_classes_only (bool): If True, only report methods in target classes.
                                     If False, report all methods found.

    Returns:
        A pandas DataFrame with detailed method annotation information, sorted.
    """
    print(f"📊 Generating Method Annotation Details Report ({'Target Classes Only' if target_classes_only else 'All Classes'})...")
    rows=[]
    methods_to_process=[]
    info=getattr(cache, 'file_info', {}) # Use getattr for safety

    # Select the source of methods based on the flag
    if target_classes_only:
        class_methods_map = getattr(cache, 'class_detailed_methods', {})
        if isinstance(class_methods_map, dict):
            print(f"  Processing methods from {len(class_methods_map)} files (target classes)...")
            # Flatten the dictionary values (which are dicts of class->list) into a single list
            for file_data in class_methods_map.values():
                 if isinstance(file_data, dict):
                      for method_list in file_data.values():
                           if isinstance(method_list, list):
                                methods_to_process.extend(method_list)
        else:
             print("  [WARN] generate_method_annotation_report: class_detailed_methods not found or not a dict.")
    else:
        all_methods_map = getattr(cache, 'detailed_methods', {})
        if isinstance(all_methods_map, dict):
            print(f"  Processing methods from {len(all_methods_map)} files (all classes)...")
            # Flatten the dictionary values (which are lists of methods)
            for method_list in all_methods_map.values():
                 if isinstance(method_list, list):
                      methods_to_process.extend(method_list)
        else:
            print("  [WARN] generate_method_annotation_report: detailed_methods not found or not a dict.")


    if not methods_to_process:
         print("  [WARN] generate_method_annotation_report: No methods found to process.")
         return pd.DataFrame()

    processed=0; skipped=0
    for m in methods_to_process:
        # Basic validation of the method dictionary structure
        if not isinstance(m, dict): skipped+=1; continue
        f=m.get(COL_FILE); c=m.get('class'); n=m.get('name'); s=m.get('full_sig')
        # Ensure essential keys have values
        if not all([f, c, n]): skipped+=1; continue

        # Get model/run info safely
        i=info.get(f,{}); mdl=i.get('model'); run=i.get('run')

        # Extract annotation details safely
        has_a=m.get('has_uc_annotation', False) # Default to False if key missing
        ucs=m.get('ucs',[]) # Default to empty list (already list from cache parse)
        action=m.get('action','') # Default to empty string

        # Append row data
        rows.append({
            COL_FILE:f, COL_MODEL:mdl, COL_RUN:run,
            'Class':c, COL_METHOD_NAME:n, 'Signature':s,
            'Has_UC_Annotation':has_a,
            'UC_References':", ".join(sorted(ucs)) if ucs else None, # Format list
            'UC_Action':action if action else None # Use None if action is empty string
        })
        processed+=1

    if skipped > 0: print(f"  [WARN] generate_method_annotation_report: Skipped {skipped} invalid/incomplete method entries.")

    if not rows:
        print("  [WARN] No valid rows generated for the method annotation report.")
        return pd.DataFrame()

    # Create DataFrame and sort
    df=pd.DataFrame(rows)
    if not df.empty:
        # Define sort columns and ensure they exist
        s_cols=[COL_MODEL, COL_RUN, COL_FILE, 'Class', COL_METHOD_NAME]
        ex_cols=[c for c in s_cols if c in df.columns]
        if ex_cols:
             try:
                 # Attempt conversion for sorting if Run is numeric-like
                 if COL_RUN in ex_cols:
                      df[COL_RUN] = pd.to_numeric(df[COL_RUN], errors='ignore')
                 df.sort_values(by=ex_cols, inplace=True, na_position='last', kind='stable') # Use stable sort
             except Exception as e:
                 print(f"  [WARN] Sort failed in method annotation report: {e}. Report may be unsorted.")
        else:
             print("  [WARN] Could not sort method annotation report - key columns missing.")

    print(f"  ✔ Generated method annotation report with {len(df)} rows.")
    return df

def calculate_coverage(cache: MetricCache, core_methods: list[str]):
    """
    Calculates coverage metrics: how many unique core methods and total core method
    occurrences are found per model within the target classes.
    Also calculates a basic consensus score based on global method occurrences.

    Args:
        cache: The MetricCache object.
        core_methods: A list of core method names.

    Returns:
        A tuple containing:
            - final_coverage_df (pd.DataFrame): DataFrame with coverage per model.
            - consensus_strength (float): Basic consensus score (0-1) or NaN.
    """
    print(f"📊 Calculating Core Method Coverage (Top {len(core_methods)})...")
    core_set = set(core_methods)
    coverage_results = [] # Store results per model found
    model_methods_target = defaultdict(list) # Stores list of TARGET method names per model

    # --- Input and Cache Checks ---
    if not core_methods:
        print("  [WARN] Core methods list is empty. Cannot calculate coverage.")
        return pd.DataFrame(), np.nan
    if not hasattr(cache, 'file_info') or not hasattr(cache, 'class_detailed_methods'):
         print("  [ERROR] Cache missing required attributes (file_info or class_detailed_methods). Returning empty.")
         return pd.DataFrame(), np.nan

    # --- Collect method names per model from target classes ---
    print("  Collecting target class method names per model...")
    num_target_methods_found = 0
    for fname in cache.get_all_files(): # Iterate through successfully loaded files
        mdl = cache.file_info.get(fname, {}).get('model')
        if not mdl: continue # Skip if model name not found

        cls_methods = cache.class_detailed_methods.get(fname, {})
        for class_name, method_list in cls_methods.items():
            # Filter only methods from target classes (redundant check, but safe)
            if class_name in cache.target_class_set:
                for method in method_list:
                    m_name = method.get('name')
                    if m_name:
                        model_methods_target[mdl].append(m_name)
                        num_target_methods_found += 1

    print(f"  Found {num_target_methods_found} total method instances in target classes across all models/runs.")

    # --- Get all unique models known to the cache ---
    all_models_in_cache = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
    if not all_models_in_cache:
        print("  [ERROR] No models found in cache info.")
        return pd.DataFrame(), np.nan

    models_with_target_methods = sorted(list(model_methods_target.keys()))
    if not models_with_target_methods:
         print("  [WARN] No target methods found for any model. Coverage will be zero.")
         # Return structure with 0 coverage for all known models
         empty_coverage_df = pd.DataFrame([
             {COL_MODEL: m, COL_UNIQUE_CORE: 0, "Total_Core_Occurrences": 0}
             for m in all_models_in_cache
         ])
         return empty_coverage_df, np.nan # Consensus likely NaN or 0

    # --- Calculate coverage per model THAT HAS target methods ---
    print("  Calculating coverage statistics per model...")
    for mdl in models_with_target_methods:
        m_list = model_methods_target.get(mdl, []) # Get collected target method names
        if not m_list: continue # Should not happen if looping models_with_target_methods

        # Count unique core methods found in this model's target class methods
        unique_core_count = len(set(m_list) & core_set)
        # Count total occurrences of core methods in this model's target class methods
        total_core_occurrences = sum(m in core_set for m in m_list)

        coverage_results.append(
            {
                COL_MODEL: mdl,
                COL_UNIQUE_CORE: unique_core_count,
                "Total_Core_Occurrences": total_core_occurrences,
            }
        )

    coverage_df = pd.DataFrame(coverage_results)

    # --- Basic Consensus Strength Calculation (Global Scope) ---
    # Counts how many core methods appear in at least half the models (based on any occurrence)
    print("  Calculating global consensus strength...")
    consensus_strength = np.nan # Default
    global_method_counter = cache.get_global_method_counter()
    num_models_total = len(all_models_in_cache) # Use count of all known models

    # Check if prerequisites for consensus calculation are met
    if global_method_counter and core_set and num_models_total > 0:
        methods_in_models = defaultdict(set)
        all_details = cache.get_method_details_list() # Uses ALL parsed methods
        if all_details: # Check if details list is populated
            for detail in all_details:
                m_name = detail.get('name')
                m_file = detail.get(COL_FILE)
                if m_name and m_file:
                    model = cache.file_info.get(m_file, {}).get('model')
                    if model:
                        methods_in_models[m_name].add(model)

            # Count core methods appearing in >= half the models
            consensus_threshold = math.ceil(num_models_total / 2.0)
            consensus_method_count = 0
            for core_method in core_set:
                # Check how many distinct models generated this core method name *anywhere*
                if len(methods_in_models.get(core_method, set())) >= consensus_threshold:
                    consensus_method_count += 1

            # Normalize by the total number of core methods considered
            consensus_strength = safe_divide(consensus_method_count, len(core_set), default=0.0)
            print(f"  Consensus: {consensus_method_count}/{len(core_set)} core methods found in >= {consensus_threshold}/{num_models_total} models. Score: {consensus_strength:.3f}")
        else:
            print("  [WARN] Skipping consensus strength calculation: Global method details list is empty.")
    else:
        print("  [WARN] Skipping consensus strength calculation: Prerequisites missing (counter, core_set, or models).")

    # --- Ensure final DataFrame includes all models ---
    # Create a base DataFrame with all models found in the cache
    final_coverage_df = pd.DataFrame({COL_MODEL: all_models_in_cache})

    if not coverage_df.empty:
        # Merge the calculated coverage results; use left merge to keep all models
        final_coverage_df = pd.merge(final_coverage_df, coverage_df, on=COL_MODEL, how='left')
        # Fill NaNs introduced by merge for models with 0 coverage (or no target methods)
        final_coverage_df[COL_UNIQUE_CORE] = final_coverage_df[COL_UNIQUE_CORE].fillna(0).astype(int)
        final_coverage_df['Total_Core_Occurrences'] = final_coverage_df['Total_Core_Occurrences'].fillna(0).astype(int)
    else: # If coverage_df was empty (e.g., no target methods found for any model)
        final_coverage_df[COL_UNIQUE_CORE] = 0
        final_coverage_df['Total_Core_Occurrences'] = 0

    # Ensure desired column order
    cols_order = [COL_MODEL, COL_UNIQUE_CORE, "Total_Core_Occurrences"]
    # Check if columns exist (they should due to initialization or merge)
    final_coverage_df = final_coverage_df[[col for col in cols_order if col in final_coverage_df.columns]]

    return final_coverage_df, consensus_strength

def calculate_method_metrics_summary(
    cache: MetricCache, per_file_method_metrics_df: pd.DataFrame
):
    """
    Aggregates per-file method metrics (calculated on target classes) by model.

    Args:
        cache: The MetricCache object (currently unused in this specific func but good practice).
        per_file_method_metrics_df: DataFrame generated by get_per_file_method_metrics.

    Returns:
        A pandas DataFrame with aggregated metrics per model.
    """
    print("📊 Aggregating method metrics per model...")

    # --- Input Validation ---
    if per_file_method_metrics_df.empty:
        print("  [WARN] Input per_file_method_metrics_df is empty for summary calculation.")
        return pd.DataFrame()
    if COL_MODEL not in per_file_method_metrics_df.columns:
         print(f"  [ERROR] Missing '{COL_MODEL}' column in per_file_method_metrics_df. Cannot calculate summary.")
         return pd.DataFrame()

    # --- Debugging Input ---
    print(f"  DEBUG_SUMMARY: Input per_file_method_metrics_df shape: {per_file_method_metrics_df.shape}")
    # print(f"  DEBUG_SUMMARY: Input dtypes:\n{per_file_method_metrics_df.dtypes}") # Can be verbose
    print(f"  DEBUG_SUMMARY: Models in input: {per_file_method_metrics_df[COL_MODEL].unique()}")


    # --- Define Columns for Aggregation ---
    mean_cols = [
        COL_REDUNDANCY, COL_PARAM_RICHNESS, COL_RETURN_COMPLETENESS,
        COL_PARAM_TYPE_COMPLETENESS, COL_PERCENT_UC,
    ]
    sum_cols = [
        COL_TOTAL_METHODS, "Unique_Method_Names",
        "DuplicatedOccurrences_Name", "DuplicatedOccurrences_NameParam", "DuplicatedOccurrences_FullSig",
        "Visibility_Public", "Visibility_Private", "Visibility_Protected", "Visibility_Package",
        COL_METHODS_WITH_UC, "Total_UC_References_File", "Unique_UCs_File", "Unique_Actions_File",
        COL_PARAMS_WITH_TYPE, "Total_Params_Count", COL_METHODS_WITH_RETURN, COL_EXTRA_METHODS,
        COL_COUNT_UC_ACTION, COL_COUNT_UC_ONLY, COL_COUNT_ACTION_ONLY, COL_COUNT_NONE,
    ]

    # Filter to columns that actually exist in the input DataFrame
    mean_present = [c for c in mean_cols if c in per_file_method_metrics_df.columns]
    sum_present = [c for c in sum_cols if c in per_file_method_metrics_df.columns]

    print(f"  DEBUG_SUMMARY: Columns intended for mean: {mean_present}")
    print(f"  DEBUG_SUMMARY: Columns intended for sum: {sum_present}")

    # --- START NEW DEBUG SECTION ---
    if mean_present:
        print(f"  DEBUG_SUMMARY: Dtypes of mean_present columns:\n{per_file_method_metrics_df[mean_present].dtypes}")
        # print(f"  DEBUG_SUMMARY: Describe mean_present columns:\n{per_file_method_metrics_df[mean_present].describe(include='all')}") # Can be very verbose
        print(f"  DEBUG_SUMMARY: NaN counts in mean_present columns:\n{per_file_method_metrics_df[mean_present].isnull().sum()}")
    else:
        print("  DEBUG_SUMMARY: No columns identified for mean aggregation.")
    if sum_present:
        print(f"  DEBUG_SUMMARY: Dtypes of sum_present columns:\n{per_file_method_metrics_df[sum_present].dtypes}")
        # print(f"  DEBUG_SUMMARY: Describe sum_present columns:\n{per_file_method_metrics_df[sum_present].describe(include='all')}") # Can be very verbose
        print(f"  DEBUG_SUMMARY: NaN counts in sum_present columns:\n{per_file_method_metrics_df[sum_present].isnull().sum()}")
    else:
        print("  DEBUG_SUMMARY: No columns identified for sum aggregation.")
    # --- END NEW DEBUG SECTION ---

    if not mean_present and not sum_present:
        print("  [WARN] No metric columns found in per-file data for aggregation.")
        models = per_file_method_metrics_df[COL_MODEL].unique()
        return pd.DataFrame({COL_MODEL: models}) if len(models) > 0 else pd.DataFrame()

    # --- Perform Aggregation ---
    summary_mean = pd.DataFrame()
    summary_sum = pd.DataFrame()
    # Ensure grouping column is not all NaN
    if per_file_method_metrics_df[COL_MODEL].isnull().all():
        print(f"  [ERROR] '{COL_MODEL}' column is all NaN. Cannot group for summary.")
        return pd.DataFrame()

    grouped = per_file_method_metrics_df.groupby(COL_MODEL)

    if mean_present:
        try:
            summary_mean = grouped[mean_present].mean(numeric_only=True).reset_index()
            print(f"  DEBUG_SUMMARY: summary_mean shape: {summary_mean.shape}")
            # if not summary_mean.empty: print(f"  DEBUG_SUMMARY: summary_mean head:\n{summary_mean.head()}")
        except Exception as e:
            print(f"  [ERROR] Aggregating means failed: {e}")
            traceback.print_exc() # Print full traceback for detailed error
    if sum_present:
        try:
            summary_sum = grouped[sum_present].sum(numeric_only=True).reset_index()
            print(f"  DEBUG_SUMMARY: summary_sum shape: {summary_sum.shape}")
            # if not summary_sum.empty: print(f"  DEBUG_SUMMARY: summary_sum head:\n{summary_sum.head()}")
        except Exception as e:
            print(f"  [ERROR] Aggregating sums failed: {e}")
            traceback.print_exc() # Print full traceback

    # --- Combine Aggregated Results ---
    method_summary = pd.DataFrame() # Initialize
    if summary_mean.empty and summary_sum.empty:
        print("  [WARN] Method summary aggregation yielded empty results (check input data validity or aggregation errors).")
        models = per_file_method_metrics_df[COL_MODEL].unique()
        return pd.DataFrame({COL_MODEL: models}) if len(models) > 0 else pd.DataFrame()
    elif summary_mean.empty:
        method_summary = summary_sum.copy() # Use copy to avoid modifying original
        for c in reversed(mean_present):
             # Insert only if column doesn't already exist from a failed sum_present
             if f"Avg_{c}_per_file" not in method_summary.columns:
                 method_summary.insert(1, f"Avg_{c}_per_file", np.nan)
    elif summary_sum.empty:
        method_summary = summary_mean.copy()
        for c in sum_present:
             if f"Total_{c}" not in method_summary.columns:
                 method_summary[f"Total_{c}"] = 0 if "Count" in c or "Occurrences" in c or "Visibility" in c else np.nan
    else:
        try:
            method_summary = pd.merge(summary_mean, summary_sum, on=COL_MODEL, how="outer")
        except Exception as e:
             print(f"  [ERROR] Merging mean and sum summaries failed: {e}")
             traceback.print_exc()
             return pd.DataFrame() # Return empty if merge fails

    if not method_summary.empty:
        print(f"  DEBUG_SUMMARY: method_summary after merge/construction shape: {method_summary.shape}")
        # if not method_summary.empty: print(f"  DEBUG_SUMMARY: method_summary after merge head:\n{method_summary.head()}")
    else:
        print("  DEBUG_SUMMARY: method_summary IS EMPTY after merge/construction logic.")
        return pd.DataFrame() # Return empty if it somehow became empty

    # --- Rename Columns ---
    AVG_PREFIX = "Avg_"; TOTAL_PREFIX = "Total_"; PER_FILE_SUFFIX = "_per_file"
    mean_map = {c: f"{AVG_PREFIX}{c}{PER_FILE_SUFFIX}" for c in mean_present}
    sum_map = {c: f"{TOTAL_PREFIX}{c}" for c in sum_present}
    method_summary.rename(columns=mean_map, inplace=True, errors='ignore') # Ignore errors if renaming non-existent cols
    method_summary.rename(columns=sum_map, inplace=True, errors='ignore')

    # --- Calculate Derived Overall Metrics ---
    print("  Calculating derived overall metrics...")
    t_total_methods_col_name = f"{TOTAL_PREFIX}{COL_TOTAL_METHODS}" # Actual renamed column
    t_unique_names_col_name = f"{TOTAL_PREFIX}Unique_Method_Names"
    t_params_count_col_name = f"{TOTAL_PREFIX}Total_Params_Count"
    t_params_w_type_col_name = f"{TOTAL_PREFIX}{COL_PARAMS_WITH_TYPE}"
    t_methods_w_ret_col_name = f"{TOTAL_PREFIX}{COL_METHODS_WITH_RETURN}"
    t_methods_w_uc_col_name = f"{TOTAL_PREFIX}{COL_METHODS_WITH_UC}"

    # Calculate safely, checking column existence
    method_summary[COL_TOTAL_REDUNDANCY] = safe_divide(method_summary.get(t_total_methods_col_name), method_summary.get(t_unique_names_col_name)) if t_total_methods_col_name in method_summary and t_unique_names_col_name in method_summary else np.nan
    method_summary["Total_ParamRichness"] = safe_divide(method_summary.get(t_params_count_col_name), method_summary.get(t_total_methods_col_name), default=0.0) if t_params_count_col_name in method_summary and t_total_methods_col_name in method_summary else np.nan
    method_summary["Total_ParamTypeCompleteness"] = safe_divide(method_summary.get(t_params_w_type_col_name), method_summary.get(t_params_count_col_name)) if t_params_w_type_col_name in method_summary and t_params_count_col_name in method_summary else np.nan
    method_summary["Total_ReturnTypeCompleteness"] = safe_divide(method_summary.get(t_methods_w_ret_col_name), method_summary.get(t_total_methods_col_name)) if t_methods_w_ret_col_name in method_summary and t_total_methods_col_name in method_summary else np.nan
    method_summary["Total_Percentage_Methods_With_UC"] = safe_divide(method_summary.get(t_methods_w_uc_col_name), method_summary.get(t_total_methods_col_name), default=0.0) if t_methods_w_uc_col_name in method_summary and t_total_methods_col_name in method_summary else np.nan

    # --- Define Final Column Order ---
    t_count_uc_action_col_name = f"{TOTAL_PREFIX}{COL_COUNT_UC_ACTION}"
    t_count_uc_only_col_name = f"{TOTAL_PREFIX}{COL_COUNT_UC_ONLY}"
    t_count_action_only_col_name = f"{TOTAL_PREFIX}{COL_COUNT_ACTION_ONLY}"
    t_count_none_col_name = f"{TOTAL_PREFIX}{COL_COUNT_NONE}"
    t_extra_methods_col_name = f"{TOTAL_PREFIX}{COL_EXTRA_METHODS}"

    final_cols_order = [
        COL_MODEL,
        # Averages per file
        f"{AVG_PREFIX}{COL_REDUNDANCY}{PER_FILE_SUFFIX}", f"{AVG_PREFIX}{COL_PARAM_RICHNESS}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_RETURN_COMPLETENESS}{PER_FILE_SUFFIX}", f"{AVG_PREFIX}{COL_PARAM_TYPE_COMPLETENESS}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_PERCENT_UC}{PER_FILE_SUFFIX}",
        # Overall derived ratios
        COL_TOTAL_REDUNDANCY, 'Total_ParamRichness', 'Total_ReturnTypeCompleteness',
        'Total_ParamTypeCompleteness', 'Total_Percentage_Methods_With_UC',
        # Total counts
        t_total_methods_col_name, t_unique_names_col_name, t_extra_methods_col_name, t_methods_w_uc_col_name,
        t_count_uc_action_col_name, t_count_uc_only_col_name, t_count_action_only_col_name, t_count_none_col_name,
        f"{TOTAL_PREFIX}Total_UC_References_File", f"{TOTAL_PREFIX}Unique_UCs_File", f"{TOTAL_PREFIX}Unique_Actions_File",
        f"{TOTAL_PREFIX}DuplicatedOccurrences_Name", f"{TOTAL_PREFIX}DuplicatedOccurrences_NameParam", f"{TOTAL_PREFIX}DuplicatedOccurrences_FullSig",
        f"{TOTAL_PREFIX}Visibility_Public", f"{TOTAL_PREFIX}Visibility_Private", f"{TOTAL_PREFIX}Visibility_Protected", f"{TOTAL_PREFIX}Visibility_Package",
    ]
    # Filter to columns that actually exist in the dataframe
    final_cols = [col for col in final_cols_order if col in method_summary.columns]
    # If final_cols is empty, it means method_summary might be malformed or empty
    if not final_cols and not method_summary.empty:
        print("  [WARN] No expected columns found in method_summary for final reordering. Returning as is.")
    elif final_cols:
        method_summary = method_summary[final_cols]

    print("  ✔ Method metrics summary calculated.")
    return method_summary
def calculate_variability(cache: MetricCache, counts_df_files_rows: pd.DataFrame):
    """
    Calculates variability metrics (Mean, CV, CI, Slope) for Target_Class_Methods counts across runs.

    Args:
        cache: The MetricCache object (used to get model list if counts fail).
        counts_df_files_rows: DataFrame with method counts per file (files as rows),
                              must include COL_MODEL, COL_RUN, and COL_TOTAL_METHODS.

    Returns:
        A pandas DataFrame with variability metrics per model.
    """
    print("📊 Calculating variability metrics...")

    # --- Input Validation ---
    required_cols = [COL_MODEL, COL_RUN, COL_TOTAL_METHODS]
    if (counts_df_files_rows.empty or
        not all(col in counts_df_files_rows.columns for col in required_cols) or
        counts_df_files_rows[COL_TOTAL_METHODS].isnull().all()): # Check if the target column is all null
        print("  [WARN] Cannot calculate variability: Input DataFrame missing required columns or has no valid data.")
        # Try to get models from cache to return empty structure
        models = []
        if hasattr(cache, 'file_info'):
            models = sorted(list(set(i.get("model", "?") for i in cache.file_info.values() if i.get("model"))))
        if models:
            return pd.DataFrame(
                { COL_MODEL: models, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
                  "CI_low": np.nan, "CI_high": np.nan, "NumRuns": 0 }
            )
        else: # Truly no info
            return pd.DataFrame()

    # --- Prepare Data ---
    # Select relevant columns and ensure correct types
    totals_info = counts_df_files_rows[required_cols].copy()
    totals_info[COL_TOTAL_METHODS] = pd.to_numeric(totals_info[COL_TOTAL_METHODS], errors='coerce')
    totals_info[COL_RUN] = pd.to_numeric(totals_info[COL_RUN], errors='coerce')

    # Drop rows where essential info (run number, method count) is invalid/missing
    totals_info.dropna(subset=[COL_RUN, COL_TOTAL_METHODS], inplace=True)

    # Check if any data remains after cleaning
    if totals_info.empty:
        print("  [WARN] No valid run/count data remaining after cleaning for variability calculation.")
        models = sorted(counts_df_files_rows[COL_MODEL].unique()) # Get models from original df if possible
        return pd.DataFrame(
                { COL_MODEL: models, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
                  "CI_low": np.nan, "CI_high": np.nan, "NumRuns": 0 }
            )

    totals_info[COL_RUN] = totals_info[COL_RUN].astype(int)

    # --- Calculate Variability per Model ---
    variability_results = []
    models_in_data = sorted(totals_info[COL_MODEL].unique())
    print(f"  Calculating variability for {len(models_in_data)} models...")

    for mdl in models_in_data:
        m_data = totals_info[totals_info[COL_MODEL] == mdl].sort_values(COL_RUN)
        v = m_data[COL_TOTAL_METHODS].values # Already cleaned of NaNs
        n_runs = len(v)

        if n_runs == 0: # Should not happen if model is in models_in_data, but safe check
            variability_results.append({
                COL_MODEL: mdl, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
                "CI_low": np.nan, "CI_high": np.nan, "NumRuns": n_runs
            })
            continue

        # Basic Stats
        mu = np.mean(v)
        # Use population std dev (ddof=0) for CV as it describes the variability of the observed runs
        sig_pop = np.std(v, ddof=0)
        # Use sample std dev (ddof=1) for CI calculation (unbiased estimator for population)
        sig_sample = np.std(v, ddof=1) if n_runs > 1 else np.nan
        cv = safe_divide(sig_pop, mu) # Use safe divide for CV

        # Convergence Slope (Cumulative Average)
        slope = np.nan
        if n_runs > 1:
            runs = m_data[COL_RUN].values
            # Cumulative average calculation
            cumulative_sum = np.cumsum(v)
            run_indices = np.arange(1, n_runs + 1)
            cumulative_avg = cumulative_sum / run_indices
            # Fit line to cumulative average vs run number (use actual run numbers)
            try:
                # Ensure we have at least 2 points for polyfit
                if len(runs) >= 2: # Polyfit needs at least 2 points
                    slope = np.polyfit(runs, cumulative_avg, 1)[0]
            except Exception as e:
                print(f"  [WARN] Slope calculation failed for model {mdl}: {e}")
                slope = np.nan # Assign NaN on failure

        # Confidence Interval (using t-distribution for small samples)
        ci_low, ci_high = np.nan, np.nan
        if n_runs > 1 and pd.notna(sig_sample) and sig_sample > 0: # Check if sample std dev is valid
            try:
                # degrees of freedom
                df_t = n_runs - 1
                # t-value for 95% CI
                t_crit = t.ppf(0.975, df=df_t)
                # Margin of error
                margin_error = t_crit * sig_sample / math.sqrt(n_runs)
                ci_low = mu - margin_error
                ci_high = mu + margin_error
            except Exception as e:
                 print(f"  [WARN] CI calculation failed for model {mdl}: {e}")
                 ci_low, ci_high = np.nan, np.nan # Assign NaN on failure

        variability_results.append({
            COL_MODEL: mdl, "Mean": mu, COL_CV: cv, "ConvergenceSlope": slope,
            "CI_low": ci_low, "CI_high": ci_high, "NumRuns": n_runs
        })

    df_variability = pd.DataFrame(variability_results)

    # Ensure all models initially found in the input df are present, even if they had no valid data
    all_models_orig = sorted(counts_df_files_rows[COL_MODEL].unique())
    final_df = pd.DataFrame({COL_MODEL: all_models_orig})
    if not df_variability.empty:
        final_df = pd.merge(final_df, df_variability, on=COL_MODEL, how='left')
        # Fill NaN for NumRuns with 0 for models that had no valid data
        if "NumRuns" in final_df.columns:
             final_df["NumRuns"] = final_df["NumRuns"].fillna(0).astype(int)

    # Reorder columns
    col_order = [COL_MODEL, "Mean", COL_CV, "ConvergenceSlope", "CI_low", "CI_high", "NumRuns"]
    final_df = final_df[[col for col in col_order if col in final_df.columns]]

    print("  ✔ Variability metrics calculated.")
    return final_df

def calculate_diversity(cache: MetricCache, counts_df_files_rows: pd.DataFrame):
    """
    Calculates diversity metrics (Entropy, Gini) based on method distribution
    across target classes, plus counts of exclusive methods per model.

    Args:
        cache: The MetricCache object.
        counts_df_files_rows: DataFrame with method counts per file (files as rows),
                              must include COL_MODEL and target class columns.

    Returns:
        A pandas DataFrame with diversity metrics per model.
    """
    print("📊 Calculating diversity metrics (Entropy, Gini, Exclusives)...")

    # --- Get Exclusives Count ---
    exclusive_methods_count = Counter()
    try:
        exclusive_methods_data = cache.get_exclusive_methods_list()
        for mdl, methods in exclusive_methods_data.items():
            exclusive_methods_count[mdl] = len(methods)
        print(f"  Found exclusive methods for {len(exclusive_methods_data)} models.")
    except Exception as e:
        print(f"  [WARN] Failed to get exclusive methods list: {e}")
        exclusive_methods_data = {} # Ensure it's a dict

    # --- Input Validation for Diversity Metrics ---
    # Get target class columns that are actually present in the DataFrame
    target_class_cols_present = [cls for cls in cache.class_names if cls in counts_df_files_rows.columns]

    models_in_df = []
    if COL_MODEL in counts_df_files_rows.columns:
         models_in_df = sorted(counts_df_files_rows[COL_MODEL].unique())

    if (counts_df_files_rows.empty or
        COL_MODEL not in counts_df_files_rows.columns or
        not target_class_cols_present): # Check if *any* target class cols exist
        print("  [WARN] Cannot calculate Entropy/Gini: Input DataFrame missing required columns or target class columns.")
        # Return structure with models and exclusive counts if possible
        if models_in_df:
             return pd.DataFrame([{
                 COL_MODEL: m, "Entropy": np.nan, COL_NORM_ENTROPY: np.nan, "Gini": np.nan,
                 "ExclusiveMethodsCount": exclusive_methods_count.get(m, 0)
                 } for m in models_in_df])
        else:
             return pd.DataFrame() # Cannot determine models

    # --- Prepare Data for Aggregation ---
    # Ensure class count columns are numeric
    diversity_data = counts_df_files_rows[[COL_MODEL] + target_class_cols_present].copy()
    for col in target_class_cols_present:
        diversity_data[col] = pd.to_numeric(diversity_data[col], errors='coerce')

    # Aggregate counts per model for the present target classes
    try:
        model_sums = (
            diversity_data.groupby(COL_MODEL)[target_class_cols_present]
            .sum(numeric_only=True) # Sum should handle NaNs as 0 by default
            .fillna(0) # Ensure any explicit NaNs become 0
        )
    except Exception as e:
        print(f"  [ERROR] Failed to group/sum counts for diversity: {e}")
        # Return structure with models and exclusive counts if possible
        return pd.DataFrame([{
                 COL_MODEL: m, "Entropy": np.nan, COL_NORM_ENTROPY: np.nan, "Gini": np.nan,
                 "ExclusiveMethodsCount": exclusive_methods_count.get(m, 0)
                 } for m in models_in_df])

    # --- Calculate Diversity Metrics per Model ---
    diversity_results = []
    num_total_target_classes = len(cache.class_names) # Use the full list for normalization
    print(f"  Calculating diversity across {len(target_class_cols_present)} present target classes (out of {num_total_target_classes} total defined).")

    for model_name, class_counts_series in model_sums.iterrows():
        counts = class_counts_series.values.astype(float) # Ensure float for division
        total_methods_model = np.sum(counts) # Sum is safe as NaNs were filled

        entropy = 0.0
        norm_entropy = 0.0
        gini = 0.0 # Gini = 0 indicates perfect equality (methods spread evenly) or only one class

        if total_methods_model > 0 and len(counts) > 0:
            proportions = counts / total_methods_model
            proportions_gt_zero = proportions[proportions > 0] # Filter out zero proportions

            # Entropy
            if len(proportions_gt_zero) > 0:
                entropy = -np.sum(proportions_gt_zero * np.log2(proportions_gt_zero))

            # Normalized Entropy (use total number of defined target classes)
            if num_total_target_classes > 1:
                norm_entropy = safe_divide(entropy, np.log2(num_total_target_classes), default=0.0)
            elif num_total_target_classes == 1 and len(proportions_gt_zero) == 1 : # Only one possible class
                 norm_entropy = 1.0 # Max possible entropy if methods exist
            else: # num_total_target_classes is 0 or 1 with no methods
                 norm_entropy = 0.0

            # Gini Coefficient
            # Create proportions array aligned with ALL target classes
            full_proportions = np.zeros(num_total_target_classes)
            present_class_map = {cls: prop for cls, prop in zip(target_class_cols_present, proportions)}
            for i, cls_name in enumerate(cache.class_names):
                full_proportions[i] = present_class_map.get(cls_name, 0.0)

            if num_total_target_classes > 0:
                sorted_proportions = np.sort(full_proportions)
                index = np.arange(1, num_total_target_classes + 1)
                # Gini formula: 1 - sum(p_i^2) OR (2 * sum(i * p_i_sorted) / n - (n+1)/n) / (1 - 1/n) -> simpler: 1 - sum(p_i^2)
                # Let's use the simpler 1 - sum(p_i^2) = 1 - Herfindahl index
                gini = 1.0 - np.sum(full_proportions**2)


        diversity_results.append({
            COL_MODEL: model_name,
            "Entropy": entropy,
            COL_NORM_ENTROPY: norm_entropy,
            "Gini": gini
        })

    diversity_df = pd.DataFrame(diversity_results)

    # --- Combine with Exclusive Counts ---
    if not diversity_df.empty:
        exclusive_df = pd.DataFrame(exclusive_methods_count.items(), columns=[COL_MODEL, 'ExclusiveMethodsCount'])
        final_diversity_df = pd.merge(diversity_df, exclusive_df, on=COL_MODEL, how='left')
        # Ensure all models from the original df are present if merge dropped some
        final_diversity_df = pd.merge(pd.DataFrame({COL_MODEL: models_in_df}), final_diversity_df, on=COL_MODEL, how='left')
        final_diversity_df['ExclusiveMethodsCount'] = final_diversity_df['ExclusiveMethodsCount'].fillna(0).astype(int)
        # Fill NaN for diversity metrics if a model had 0 methods? Default to 0 might be reasonable.
        for col in ["Entropy", COL_NORM_ENTROPY, "Gini"]:
            if col in final_diversity_df.columns:
                 final_diversity_df[col] = final_diversity_df[col].fillna(0.0)

    else: # Handle case where diversity calcs failed but we have models
        final_diversity_df = pd.DataFrame([{
                 COL_MODEL: m, "Entropy": 0.0, COL_NORM_ENTROPY: 0.0, "Gini": 0.0,
                 "ExclusiveMethodsCount": exclusive_methods_count.get(m, 0)
                 } for m in models_in_df])


    # Final column order
    cols_order = [COL_MODEL, 'Entropy', COL_NORM_ENTROPY, 'Gini', 'ExclusiveMethodsCount']
    final_diversity_df = final_diversity_df[[col for col in cols_order if col in final_diversity_df.columns]]

    print("  ✔ Diversity metrics calculated.")
    return final_diversity_df

def generate_exclusive_methods_report(cache: MetricCache):
    """
    Generates a report listing methods generated exclusively by only one model.

    Args:
        cache: The MetricCache object.

    Returns:
        A pandas DataFrame listing exclusive methods per model, including an example context.
    """
    print("📊 Generating Exclusive Methods Report...")
    rows=[]

    # --- Get Exclusive Methods Data ---
    # Use getter which handles checks internally
    exclusive_list_data = cache.get_exclusive_methods_list()
    if not exclusive_list_data:
        print("  ℹ No exclusive methods found for any model.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[COL_MODEL, COL_METHOD_NAME, 'ExampleFile', 'ExampleClass', 'ExampleSignature', 'HasUCAnnotation'])

    models = sorted(exclusive_list_data.keys())
    print(f"  Found exclusive methods for {len(models)} models.")

    # --- Get Detailed Info for Examples ---
    # Check if details needed for examples are available
    if not hasattr(cache, 'get_method_details_list') or not hasattr(cache, 'file_info'):
        print("  [WARN] Cannot retrieve example context for exclusive methods: Cache data missing.")
        # Proceed without examples if necessary
        details = []
        info = {}
    else:
        details = cache.get_method_details_list()
        info = cache.file_info

    # --- Build Report Rows ---
    methods_processed_count = 0
    for mdl in models:
         # Sort methods alphabetically within each model
         methods = sorted(exclusive_list_data.get(mdl, []))
         for name in methods:
              methods_processed_count += 1
              # Find the first occurrence of this method generated by this model
              # (to provide a concrete example file/class/signature)
              example_detail = None
              if details and info: # Only search if data is available
                    example_detail = next((
                        m for m in details
                        if m.get('name') == name and info.get(m.get(COL_FILE), {}).get('model') == mdl
                        ), None) # Find first match or return None

              # Append data for this exclusive method
              rows.append({
                  COL_MODEL: mdl,
                  COL_METHOD_NAME: name,
                  # Safely get example details using .get() with defaults
                  'ExampleFile': example_detail.get(COL_FILE) if example_detail else None,
                  'ExampleClass': example_detail.get('class') if example_detail else None,
                  'ExampleSignature': example_detail.get('full_sig') if example_detail else '', # Use empty string if no sig
                  'HasUCAnnotation': example_detail.get('has_uc_annotation', False) if example_detail else False,
              })

    print(f"  ✔ Generated exclusive methods report for {methods_processed_count} methods.")

    # Create and return DataFrame
    if not rows: # Should not happen if exclusive_list_data was not empty, but safe check
         return pd.DataFrame(columns=[COL_MODEL, COL_METHOD_NAME, 'ExampleFile', 'ExampleClass', 'ExampleSignature', 'HasUCAnnotation'])
    else:
         return pd.DataFrame(rows)
    
def calculate_annotation_metrics(
    cache: MetricCache, per_file_method_metrics_df: pd.DataFrame
):
    """
    Aggregates annotation-specific metrics (UC/Action counts) by model, based
    on per-file metrics derived from target class methods. Calculates overall
    unique UC/Action counts across all runs for each model based on target classes.

    Args:
        cache: The MetricCache object (used for detailed unique counts).
        per_file_method_metrics_df: DataFrame containing per-file metrics.

    Returns:
        A pandas DataFrame with aggregated annotation metrics per model.
    """
    print("📊 Calculating annotation metrics summary...")

    # --- Input Validation ---
    if (per_file_method_metrics_df.empty or
        COL_MODEL not in per_file_method_metrics_df.columns):
        print("  [WARN] Annotation metrics summary skipped - input per-file metrics invalid.")
        cols = [COL_MODEL, f'Total_{COL_METHODS_WITH_UC}', f'Avg_{COL_PERCENT_UC}_per_file',
                'Total_Percentage_Methods_With_UC', 'Total_UC_References', 'Total_Unique_UCs',
                'Avg_Unique_UCs_PerFile', 'Total_Unique_Actions', 'Avg_Unique_Actions_PerFile']
        return pd.DataFrame(columns=cols)

    # --- Aggregation based on Per-File Metrics ---
    # Define aggregation operations
    agg_cols = {
        f'Total_{COL_METHODS_WITH_UC}': (COL_METHODS_WITH_UC, 'sum'),         # Total methods annotated
        f'Avg_{COL_PERCENT_UC}_per_file': (COL_PERCENT_UC, 'mean'),          # Avg % annotated per run
        'Total_UC_References': ('Total_UC_References_File', 'sum'),      # Sum of all UC refs listed
        'Avg_Unique_UCs_PerFile': ('Unique_UCs_File', 'mean'),           # Avg unique UCs *per run*
        'Avg_Unique_Actions_PerFile': ('Unique_Actions_File', 'mean')    # Avg unique Actions *per run*
    }

    # Filter to columns present in the input DataFrame
    valid_agg_cols = {
        k: v for k, v in agg_cols.items()
        if v[0] in per_file_method_metrics_df.columns
    }

    if not valid_agg_cols:
        print("  [WARN] Annotation metrics summary skipped - required columns missing from per-file metrics.")
        cols = [COL_MODEL, f'Total_{COL_METHODS_WITH_UC}', f'Avg_{COL_PERCENT_UC}_per_file',
                'Total_Percentage_Methods_With_UC', 'Total_UC_References', 'Total_Unique_UCs',
                'Avg_Unique_UCs_PerFile', 'Total_Unique_Actions', 'Avg_Unique_Actions_PerFile']
        models_in_df = per_file_method_metrics_df[COL_MODEL].unique() if COL_MODEL in per_file_method_metrics_df else []
        if len(models_in_df)>0: return pd.DataFrame({COL_MODEL: models_in_df, **{c: np.nan for c in cols if c!=COL_MODEL}})
        else: return pd.DataFrame(columns=cols)

    # Perform initial aggregation
    try:
        annotation_agg = (
            per_file_method_metrics_df.groupby(COL_MODEL)
            .agg(**valid_agg_cols)
            .reset_index()
        )
    except Exception as e:
         print(f"  [ERROR] during initial annotation aggregation: {e}")
         return pd.DataFrame() # Return empty on aggregation error

    # --- Calculate OVERALL Unique UCs and Actions per Model (from detailed cache data) ---
    print("  Calculating overall unique UCs/Actions per model (target classes)...")
    model_unique_data = []
    # Check cache prerequisites
    if hasattr(cache, 'class_detailed_methods') and hasattr(cache, 'file_info'):
        class_methods_map = cache.get_class_detailed_methods() # Use getter
        model_files = defaultdict(list)
        # Group files by model
        [model_files[info['model']].append(fname) for fname, info in cache.file_info.items() if info.get('model')]

        for model, files in model_files.items():
            model_total_unique_ucs = set()
            model_total_unique_actions = set()
            for fname in files:
                 for cls_name, method_list in class_methods_map.get(fname, {}).items():
                      for method in method_list:
                        # 'ucs' is list of strings, 'action' is string from Cache parse
                        ucs = method.get('ucs', [])
                        action = method.get('action', '')
                        model_total_unique_ucs.update(ucs)
                        if action: model_total_unique_actions.add(action)

            model_unique_data.append({
                COL_MODEL: model,
                'Total_Unique_UCs': len(model_total_unique_ucs),     # Overall unique UCs for model
                'Total_Unique_Actions': len(model_total_unique_actions) # Overall unique Actions for model
            })
        print(f"  Processed {len(model_files)} models for overall unique counts.")
    else:
         print("  [WARN] Cannot calculate total unique UCs/Actions - cache missing attributes.")

    # --- Merge Overall Unique Counts ---
    if model_unique_data:
        unique_counts_df = pd.DataFrame(model_unique_data)
        annotation_agg = pd.merge(annotation_agg, unique_counts_df, on=COL_MODEL, how='left')
        # Fill NaN for models potentially missing (though unlikely if models come from same source)
        annotation_agg['Total_Unique_UCs'] = annotation_agg['Total_Unique_UCs'].fillna(0).astype(int)
        annotation_agg['Total_Unique_Actions'] = annotation_agg['Total_Unique_Actions'].fillna(0).astype(int)
    else:
        # Add columns with NaN if calculation failed
        if 'Total_Unique_UCs' not in annotation_agg.columns: annotation_agg['Total_Unique_UCs'] = np.nan
        if 'Total_Unique_Actions' not in annotation_agg.columns: annotation_agg['Total_Unique_Actions'] = np.nan

    # --- Calculate Overall Percentage of Methods with UC ---
    print("  Calculating overall percentage of methods with UC...")
    t_methods = COL_TOTAL_METHODS
    t_agg_methods = f'Total_{t_methods}' # Needs total methods per model
    t_agg_uc = f'Total_{COL_METHODS_WITH_UC}' # Calculated in initial agg

    # Calculate total methods per model from the per-file data if not already present
    if t_methods in per_file_method_metrics_df.columns:
        # Avoid recalculating if already present (e.g., from method_summary) - check if needed
        if t_agg_methods not in annotation_agg.columns:
             total_methods_per_model = (
                 per_file_method_metrics_df.groupby(COL_MODEL)[t_methods]
                 .sum(numeric_only=True)
                 .reset_index(name=t_agg_methods) # Name the summed column correctly
             )
             annotation_agg = pd.merge(annotation_agg, total_methods_per_model, on=COL_MODEL, how='left')

        # Now calculate the overall percentage
        if t_agg_uc in annotation_agg.columns and t_agg_methods in annotation_agg.columns:
            annotation_agg[t_agg_uc] = annotation_agg[t_agg_uc].fillna(0)
            annotation_agg[t_agg_methods] = annotation_agg[t_agg_methods].fillna(0)
            annotation_agg['Total_Percentage_Methods_With_UC'] = safe_divide(
                annotation_agg[t_agg_uc], annotation_agg[t_agg_methods], default=0.0
            )
        else:
            annotation_agg['Total_Percentage_Methods_With_UC'] = np.nan
        # Can drop t_agg_methods if not needed later
        # annotation_agg.drop(columns=[t_agg_methods], errors='ignore', inplace=True)
    else:
        annotation_agg['Total_Percentage_Methods_With_UC'] = np.nan

    # --- Finalize Columns ---
    final_order = [
        COL_MODEL,
        t_agg_uc,                           # Total methods with any UC (sum over files)
        f'Avg_{COL_PERCENT_UC}_per_file',    # Avg % of methods with UC per file
        'Total_Percentage_Methods_With_UC', # Overall % of methods with UC
        'Total_UC_References',              # Sum of all UC references listed
        'Total_Unique_UCs',                 # Overall unique UCs for the model
        'Avg_Unique_UCs_PerFile',           # Avg unique UCs found per file
        'Total_Unique_Actions',             # Overall unique Actions for the model
        'Avg_Unique_Actions_PerFile'        # Avg unique Actions found per file
    ]
    # Filter to existing columns and apply order
    final_cols = [c for c in final_order if c in annotation_agg.columns]
    annotation_agg = annotation_agg[final_cols]

    print("  ✔ Annotation metrics summary calculated.")
    return annotation_agg    

def generate_uc_frequency_report(cache: MetricCache):
    """
    Generates a report showing the global frequency of each Use Case ID
    referenced across all methods in all generated files.

    Args:
        cache: The MetricCache object.

    Returns:
        A pandas DataFrame listing UC IDs and their global frequency, sorted.
    """
    print("📊 Generating UC Reference Frequency Report (Global)...")

    # --- Get Global UC Counter ---
    # Use getter which handles checks internally
    uc_counter = cache.get_global_uc_counter()

    if not uc_counter:
        print("  [WARN] Global UC counter is empty. No UC references found.")
        return pd.DataFrame(columns=['UC_ID', COL_GLOBAL_FREQ])

    # --- Create DataFrame ---
    # .most_common() returns list of (element, count) tuples
    df = pd.DataFrame(uc_counter.most_common(), columns=['UC_ID', COL_GLOBAL_FREQ])

    # --- Sort (already sorted by most_common, but can re-sort for consistency) ---
    # Sort primarily by Frequency (descending), then by UC_ID (ascending) as a tie-breaker
    if not df.empty:
        try:
            # Attempt to sort UC_ID as string (common format like 'UC1', 'UC10')
            df.sort_values(by=[COL_GLOBAL_FREQ, 'UC_ID'], ascending=[False, True], inplace=True, key=lambda col: col.astype(str) if col.name == 'UC_ID' else col)
        except Exception as e:
             print(f"  [WARN] Sorting UC frequency report failed: {e}. Report might be unsorted.")


    print(f"  ✔ Generated UC frequency report for {len(df)} unique UC IDs.")
    return df

def generate_action_frequency_report(cache: MetricCache):
    """
    Generates a report showing the global frequency of each unique Action string
    found in the 'uc_action' annotation field across all methods in all
    generated files.

    Args:
        cache: The MetricCache object.

    Returns:
        A pandas DataFrame listing Action strings and their global frequency, sorted.
    """
    print("📊 Generating Action Annotation Frequency Report (Global)...")

    # --- Get Global Action Counter ---
    # Use getter which handles checks internally
    action_counter = cache.get_global_action_counter()

    if not action_counter:
        print("  [WARN] Global Action counter is empty. No 'uc_action' annotations found.")
        return pd.DataFrame(columns=['Action', COL_GLOBAL_FREQ])

    # --- Create DataFrame ---
    # .most_common() returns list of (element, count) tuples
    # Filters out potential empty strings if they were counted, though cache should handle this
    action_data = [(action, count) for action, count in action_counter.most_common() if action] # Ensure action is not empty
    df = pd.DataFrame(action_data, columns=['Action', COL_GLOBAL_FREQ])

    # --- Sort ---
    # Sort primarily by Frequency (descending), then by Action string (ascending)
    if not df.empty:
        try:
            # Sorting by string 'Action' is standard
            df.sort_values(by=[COL_GLOBAL_FREQ, 'Action'], ascending=[False, True], inplace=True)
        except Exception as e:
             print(f"  [WARN] Sorting Action frequency report failed: {e}. Report might be unsorted.")

    print(f"  ✔ Generated Action frequency report for {len(df)} unique non-empty Actions.")
    return df

def generate_uc_method_report(cache: MetricCache):
    """
    Generates a report listing methods that have associated UC annotations,
    including details like the referenced UCs and the annotated action.

    Args:
        cache: The MetricCache object.

    Returns:
        A pandas DataFrame listing methods with UC annotations and their details.
    """
    print("📊 Generating Report of Methods with UC Annotations...")
    rows=[]
    # --- Get Prerequisite Data ---
    # Use getters which handle checks internally
    details = cache.get_method_details_list()
    info = getattr(cache, 'file_info', {}) # Use getattr for safety

    if not details:
        print("  [WARN] No global method details found in cache. Cannot generate UC method report.")
        return pd.DataFrame()
    if not info:
        print("  [WARN] File info missing from cache. Model/Run information might be incomplete.")
        # Decide if this is critical - for now, proceed but model/run might be None

    # --- Iterate and Collect Annotated Methods ---
    annotated_method_count = 0
    for m in details:
        # Check if the method has the 'has_uc_annotation' flag set to True
        # (This flag is set during the MetricCache._parse method)
        if m.get('has_uc_annotation', False): # Safely check the flag
            annotated_method_count += 1
            # Get model/run info safely
            file_path = m.get(COL_FILE)
            file_info = info.get(file_path, {}) if file_path else {}
            model = file_info.get('model')
            run = file_info.get('run')

            # Get annotation details safely
            ucs = m.get('ucs', []) # Already list of strings from cache parse
            action = m.get('action', '') # Already string

            # Append relevant details to the report rows
            rows.append({
                COL_FILE: file_path,
                COL_MODEL: model,
                COL_RUN: run,
                'Class': m.get('class'),
                COL_METHOD_NAME: m.get('name'),
                'Signature': m.get('full_sig'), # Include signature
                'UCs': ",".join(sorted(ucs)) if ucs else None, # Comma-separated list
                'Action': action if action else None, # Use None if empty
                'Modifier': m.get('visibility', '+'), # Default to public
                'ReturnType': m.get('return_type', 'void') # Default to void
            })

    if annotated_method_count == 0:
         print("  [INFO] No methods with UC annotations were found.")
         # Return empty DataFrame with expected columns
         cols = [COL_FILE, COL_MODEL, COL_RUN, 'Class', COL_METHOD_NAME, 'Signature', 'UCs', 'Action', 'Modifier', 'ReturnType']
         return pd.DataFrame(columns=cols)

    # --- Create and Sort DataFrame ---
    df = pd.DataFrame(rows)
    if not df.empty:
        # Define sort columns and ensure they exist
        s_cols = [COL_MODEL, COL_RUN, COL_FILE, 'Class', COL_METHOD_NAME]
        ex_cols = [c for c in s_cols if c in df.columns]
        if ex_cols:
             try:
                 if COL_RUN in ex_cols: df[COL_RUN] = pd.to_numeric(df[COL_RUN], errors='ignore')
                 df.sort_values(by=ex_cols, inplace=True, na_position='last', kind='stable')
             except Exception as e:
                 print(f"  [WARN] Sort failed in UC method report: {e}. Report may be unsorted.")
        else:
             print("  [WARN] Could not sort UC method report - key columns missing.")

    print(f"  ✔ Generated UC method report for {annotated_method_count} annotated methods.")
    return df

def calculate_added_class_llm_counts(cache: MetricCache):
    """
    Identifies classes present in generated models but not in the baseline,
    and counts how many distinct LLMs added each such class.

    Args:
        cache: The MetricCache object containing parsed baseline and generated data.

    Returns:
        A pandas DataFrame listing added classes and the LLMs that added them,
        sorted by count and then class name, or an empty DataFrame if none found.
        Columns: Added_Class, LLM_Count, LLM_List
    """
    print("📊 Calculating Added Class Counts (vs Baseline)...")

    # --- Input Validation ---
    # Check if baseline structure was parsed and contains class info
    if not hasattr(cache, 'baseline_structure') or not cache.baseline_structure or 'classes' not in cache.baseline_structure:
        print("  [WARN] Baseline structure/classes not available. Cannot calculate added classes.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])

    baseline_classes = cache.baseline_structure.get('classes', set()) # Safely get the set
    print(f"  Baseline contains {len(baseline_classes)} classes.")

    # Check if generated data is available
    if not hasattr(cache, 'json_data') or not cache.json_data or not hasattr(cache, 'files') or not cache.files:
        print("  [WARN] No generated JSON data found in cache. Cannot calculate added classes.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])
    if not hasattr(cache, 'file_info'):
        print("  [WARN] File info missing from cache. Cannot map added classes to models.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])


    # --- Track Added Classes per Model ---
    added_class_tracker = defaultdict(set) # Key: added_class_name, Value: set of models adding it
    all_models_processed = set() # Track models we actually processed files for
    files_processed_count = 0
    errors_extracting_classes = 0

    for json_filename in cache.files: # Iterate through successfully loaded files
        model = cache.file_info.get(json_filename, {}).get('model')
        stem = cache.file_info.get(json_filename, {}).get('stem')

        # Ensure we have model info and the corresponding data exists
        if not model or not stem or stem not in cache.json_data:
            continue # Skip if file info or data is inconsistent

        all_models_processed.add(model) # Track models encountered
        file_json = cache.json_data[stem]
        files_processed_count += 1

        # Extract classes generated in this specific file
        try:
            # Use extract_structure_from_json for consistency, though direct parsing is okay too
            # generated_structure = extract_structure_from_json(file_json)
            # gen_classes_in_file = generated_structure.get('classes', set())
            # --- OR direct extraction: ---
            gen_classes_in_file = set(
                c.get('name')
                for c in file_json.get('classes', [])
                if c.get('name') # Ensure name exists and is not empty/None
            )
        except Exception as e:
             print(f"  [ERROR] Failed to extract classes from {json_filename}: {e}")
             errors_extracting_classes += 1
             continue # Skip file if class extraction fails

        # Find classes added in this file compared to baseline
        added_in_file = gen_classes_in_file - baseline_classes

        # Add the current model to the tracker set for each added class
        for added_class in added_in_file:
            if added_class: # Ensure added class name is not empty
                 added_class_tracker[added_class].add(model)

    print(f"  Processed {files_processed_count} generated files for added classes.")
    if errors_extracting_classes > 0:
        print(f"  [WARN] Encountered errors extracting classes from {errors_extracting_classes} files.")

    # --- Format Results ---
    output_data = []
    for added_class, models_set in added_class_tracker.items():
        output_data.append({
            'Added_Class': added_class,
            'LLM_Count': len(models_set),
            'LLM_List': ", ".join(sorted(list(models_set))) # Comma-separated sorted list
        })

    if not output_data:
        print("  ✔ No added classes (classes not in baseline) found in any generated model.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])

    # --- Create DataFrame and Sort ---
    results_df = pd.DataFrame(output_data).sort_values(
        by=['LLM_Count', 'Added_Class'],
        ascending=[False, True] # Sort by count descending, then name ascending
    )
    print(f"  ✔ Found {len(results_df)} classes added by at least one model.")
    return results_df

def calculate_overlap(cache: MetricCache, counts_df_files_rows: pd.DataFrame):
    """
    Calculates overlap metrics between models:
    1. Bootstrap Overlap: Probability that model A generates more target class
       methods than model B based on resampling runs.
    2. Global Jaccard Similarity: Overlap of the sets of *all* method names
       generated by each model pair.

    Args:
        cache: The MetricCache object.
        counts_df_files_rows: DataFrame with method counts per file (files as rows),
                              must include COL_MODEL and COL_TOTAL_METHODS.

    Returns:
        A tuple containing:
            - bootstrap_overlap_df (pd.DataFrame): Matrix of bootstrap overlap probs.
            - jaccard_global_df (pd.DataFrame): Matrix of global Jaccard similarities.
    """
    print("📊 Calculating overlap metrics (Bootstrap and Global Jaccard)...")

    # --- Get list of unique models ---
    models = []
    if hasattr(cache, 'file_info'): # Get models known to the cache first
        models = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
    if not models:
        print("  [ERROR] No models found in cache info. Cannot calculate overlap.")
        return pd.DataFrame(), pd.DataFrame()
    print(f"  Found {len(models)} models in cache.")

    # --- 1. Bootstrap Overlap Calculation ---
    bootstrap_overlap_df = pd.DataFrame(np.nan, index=models, columns=models)
    bootstrap_skipped = False

    # Validate input DataFrame for bootstrap
    required_bootstrap_cols = [COL_MODEL, COL_TOTAL_METHODS]
    if (counts_df_files_rows.empty or
        not all(col in counts_df_files_rows.columns for col in required_bootstrap_cols) or
        counts_df_files_rows[COL_TOTAL_METHODS].isnull().all()):
        print("  [WARN] Skipping Bootstrap Overlap: Input counts DataFrame is invalid or empty.")
        bootstrap_skipped = True
    else:
        # Prepare data: select relevant columns and convert to numeric
        totals = counts_df_files_rows[required_bootstrap_cols].copy()
        totals[COL_TOTAL_METHODS] = pd.to_numeric(totals[COL_TOTAL_METHODS], errors='coerce')
        totals.dropna(subset=[COL_TOTAL_METHODS], inplace=True) # Drop rows where count couldn't be numeric

        if totals.empty:
            print("  [WARN] Skipping Bootstrap Overlap: No valid numeric method counts found after cleaning.")
            bootstrap_skipped = True
        else:
            print("  Performing Bootstrap resampling...")
            bootstrap_cache = {} # Cache resampled means per model
            n_bootstrap = 1000 # Number of bootstrap samples

            # Check which models from the main list actually have data in totals df
            models_with_counts = sorted(totals[COL_MODEL].unique())
            missing_models = set(models) - set(models_with_counts)
            if missing_models:
                 print(f"  [WARN] Bootstrap: Models missing count data: {missing_models}")

            for mdl in models: # Iterate through ALL models found initially
                if mdl not in models_with_counts:
                    bootstrap_cache[mdl] = np.array([]) # Empty array if model had no count data
                    continue

                # Get the vector of total method counts for this model's runs
                v = totals.loc[totals[COL_MODEL] == mdl, COL_TOTAL_METHODS].values
                n_clean = len(v) # Already valid counts

                if n_clean > 0:
                    # Resample with replacement and calculate mean for each sample
                    # Handle case where n_clean=1 (cannot bootstrap effectively)
                    if n_clean == 1:
                         resampled_means = np.full(n_bootstrap, v[0]) # All samples will be the single value
                    else:
                         resampled_indices = np.random.randint(0, n_clean, size=(n_bootstrap, n_clean))
                         resampled_means = v[resampled_indices].mean(axis=1)
                    bootstrap_cache[mdl] = resampled_means
                else: # Should not happen if mdl in models_with_counts, but safe check
                    bootstrap_cache[mdl] = np.array([])

            # Calculate pairwise overlap probabilities
            for model_a, model_b in it.combinations(models, 2):
                dist_a = bootstrap_cache.get(model_a, np.array([]))
                dist_b = bootstrap_cache.get(model_b, np.array([]))

                prob_a_gt_b = np.nan # Default if calculation fails
                if len(dist_a) > 0 and len(dist_b) > 0:
                    # Calculate the proportion of times resampled mean of A > resampled mean of B
                    prob_a_gt_b = np.mean(dist_a > dist_b)
                    bootstrap_overlap_df.loc[model_a, model_b] = prob_a_gt_b
                    # P(B > A) is approximately 1 - P(A > B)
                    bootstrap_overlap_df.loc[model_b, model_a] = 1.0 - prob_a_gt_b if pd.notna(prob_a_gt_b) else np.nan
                else:
                    # If one model had no data, can't compare
                    bootstrap_overlap_df.loc[model_a, model_b] = np.nan
                    bootstrap_overlap_df.loc[model_b, model_a] = np.nan
            print("  ✔ Bootstrap calculation finished.")

    # --- 2. Global Jaccard Similarity Calculation ---
    print("  Calculating Global Jaccard Similarity (based on all method names)...")
    jaccard_global_df = pd.DataFrame(np.nan, index=models, columns=models)
    jaccard_skipped = False
    model_method_sets = {} # Store sets of all methods per model

    # Check cache prerequisites
    if not hasattr(cache, 'get_method_details_list') or not hasattr(cache, 'file_info'):
         print("  [WARN] Skipping Global Jaccard: Cache missing required attributes.")
         jaccard_skipped = True
    else:
        all_details = cache.get_method_details_list() # Uses ALL methods
        info = cache.file_info
        if not all_details:
             print("  [WARN] Skipping Global Jaccard: No global method details found in cache.")
             jaccard_skipped = True
        else:
            # Create a set of all method names for each model
            temp_sets = defaultdict(set)
            for m_detail in all_details:
                m_name = m_detail.get('name')
                m_file = m_detail.get(COL_FILE)
                if m_name and m_file:
                    model = info.get(m_file, {}).get('model')
                    if model in models: # Only consider models we identified initially
                        temp_sets[model].add(m_name)

            # Ensure sets exist for all models, even if empty
            for mdl in models:
                model_method_sets[mdl] = temp_sets.get(mdl, set())

            # Calculate pairwise Jaccard index
            for model_a, model_b in it.combinations(models, 2):
                set_a = model_method_sets.get(model_a, set())
                set_b = model_method_sets.get(model_b, set())

                intersection = len(set_a & set_b)
                union = len(set_a | set_b)

                jaccard_index = safe_divide(intersection, union, default=0.0)
                jaccard_global_df.loc[model_a, model_b] = jaccard_index
                jaccard_global_df.loc[model_b, model_a] = jaccard_index # Symmetric

            # Fill diagonal with 1.0
            np.fill_diagonal(jaccard_global_df.values, 1.0)
            print("  ✔ Global Jaccard calculation finished.")

    return bootstrap_overlap_df, jaccard_global_df

def calculate_per_class_overlap(cache: MetricCache):
    """
    Calculates the Jaccard similarity of method names between models,
    specifically within each target class.

    Args:
        cache: The MetricCache object.

    Returns:
        A dictionary where keys are target class names and values are
        pandas DataFrames representing the Jaccard similarity matrix for that class.
        Returns an empty dictionary if calculation cannot proceed or no overlap found.
    """
    print("📊 Calculating Per-Class Jaccard overlap...")

    # --- Get list of unique models ---
    models = []
    if hasattr(cache, 'file_info'):
        models = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
    if not models:
        print("  [ERROR] No models found in cache info. Cannot calculate per-class overlap.")
        return {}

    # Check for necessary cache attributes
    if not hasattr(cache, 'class_detailed_methods') or not hasattr(cache, 'class_names'):
         print("  [ERROR] Cache missing required attributes (class_detailed_methods or class_names).")
         return {}

    per_class_jaccard_results = {}
    class_methods_map = cache.get_class_detailed_methods() # Use getter
    info = cache.file_info # Already checked above
    target_classes = cache.class_names # Use list from cache

    if not target_classes:
        print("  [WARN] No target classes defined in CLASS_NAMES configuration.")
        return {}

    print(f"  Processing {len(target_classes)} target classes for {len(models)} models...")
    classes_processed_count = 0
    matrices_generated = 0

    for class_name in target_classes:
        classes_processed_count += 1
        # --- Create set of method names per model FOR THIS CLASS ---
        model_class_method_sets = defaultdict(set)
        methods_found_in_class_for_any_model = False # Track if any model had methods for this class

        for model_name in models:
            # Find all files associated with this model
            model_files = [f for f, i in info.items() if i.get('model') == model_name]
            if not model_files: continue # Skip if model has no files

            # Collect method names from this model's files for the current class_name
            for fname in model_files:
                # Get methods for the specific file and class
                methods_in_file_class = class_methods_map.get(fname, {}).get(class_name, [])
                if methods_in_file_class:
                    methods_found_in_class_for_any_model = True # Mark that this class has data
                    for method in methods_in_file_class:
                        m_name = method.get('name')
                        if m_name: # Ensure method name exists
                            model_class_method_sets[model_name].add(m_name)
            # If a model has no methods for this class, its set will remain empty

        # Skip calculating matrix if no methods were found for this class in *any* model
        if not methods_found_in_class_for_any_model:
            continue

        # --- Calculate Jaccard matrix for this class ---
        # Initialize matrix for all models
        jaccard_matrix = pd.DataFrame(np.nan, index=models, columns=models)

        for model_a, model_b in it.combinations(models, 2):
            set_a = model_class_method_sets.get(model_a, set()) # Use .get() for safety
            set_b = model_class_method_sets.get(model_b, set())

            intersection = len(set_a & set_b)
            union = len(set_a | set_b)

            # Use safe_divide helper function
            jaccard_index = safe_divide(intersection, union, default=0.0)
            jaccard_matrix.loc[model_a, model_b] = jaccard_index
            jaccard_matrix.loc[model_b, model_a] = jaccard_index # Symmetric

        # Fill diagonal with 1.0
        np.fill_diagonal(jaccard_matrix.values, 1.0)

        # Only store matrix if it provides meaningful comparison (more than one model has methods for this class)
        non_empty_sets = sum(1 for s in model_class_method_sets.values() if s)
        if non_empty_sets >= 2:
            per_class_jaccard_results[class_name] = jaccard_matrix
            matrices_generated += 1

    print(f"  ✔ Per-Class Jaccard finished. Processed {classes_processed_count} classes, generated {matrices_generated} matrices.")
    return per_class_jaccard_results   

# (Include FULL definitions for generate_core_methods_report, generate_exclusive_methods_report,
# generate_uc_method_report, generate_uc_frequency_report,
# generate_action_frequency_report, generate_method_annotation_report)
# ... Example: generate_core_methods_report ...
def generate_core_methods_report(cache: MetricCache, top_n: int):
    counter = cache.get_global_method_counter();
    if not counter: print("[generate_core_methods_report] Warning: Global method counter is empty."); return pd.DataFrame(columns=[COL_METHOD_NAME, COL_GLOBAL_FREQ])
    df=pd.DataFrame(counter.most_common(top_n), columns=[COL_METHOD_NAME, COL_GLOBAL_FREQ]); return df
# ... PASTE ALL OTHER REPORTING FUNCTIONS HERE ...


# --- Main Execution ---
def main():
    print("🚀 Starting Metrics Pipeline...")
    print("🔍 Setting up directories...")
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True); print(f"Output reports: '{REPORTS_OUTPUT_DIR.resolve()}'")
    if not JSON_INPUT_DIR.is_dir(): sys.exit(f"❌ Error: Input JSON directory not found: '{JSON_INPUT_DIR.resolve()}'.")

    # --- Load & Preprocess Gold Standard Map ---
    gold_map = None; gold_map_details = None; gold_map_path = Path(GOLD_STANDARD_MAP_FNAME)
    if gold_map_path.is_file():
        try:
            with open(gold_map_path, 'r', encoding='utf-8') as f: gold_map = json.load(f); print(f"✔ Loaded gold map: '{gold_map_path}'")
            gold_map_details = _preprocess_gold_map(gold_map)
            if not gold_map_details: print(f"❌ Error preprocessing gold map. Coverage calculations may fail.")
        except Exception as e: print(f"❌ Error loading/processing gold map '{gold_map_path}': {e}")
    else: print(f"⚠️ Warning: Gold map not found at '{gold_map_path}'. Coverage/Mapping skipped.")

    # --- Initialize Cache ---
    print("⏳ Loading JSON data & caching...")
    cache = None
    try:
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, CLASS_NAMES)
        if not cache.get_all_files(): print("⚠️ Warning: Cache initialized, but no generated JSON files loaded.")
        else: print("✔ Cache ready.")
    except Exception as e: print(f"❌❌❌ CRITICAL ERROR during MetricCache initialization: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Define output paths ---
    structural_csv=REPORTS_OUTPUT_DIR/"StructuralPreservationReport.csv"; counts_csv=REPORTS_OUTPUT_DIR/"Counts_TargetClasses.csv"; metrics_per_file_csv=REPORTS_OUTPUT_DIR/"MethodMetrics_PerFile.csv"; metrics_summary_csv=REPORTS_OUTPUT_DIR/"MethodMetrics_Summary.csv"; variability_csv=REPORTS_OUTPUT_DIR/"VariabilityMetrics.csv"; core_methods_csv=REPORTS_OUTPUT_DIR/"CoreMethods_TopN.csv"; coverage_csv=REPORTS_OUTPUT_DIR/"CoverageMetrics.csv"; consensus_csv=REPORTS_OUTPUT_DIR/"ConsensusStrength.csv"; diversity_csv=REPORTS_OUTPUT_DIR/"DiversityMetrics.csv"; exclusive_csv=REPORTS_OUTPUT_DIR/"ExclusiveMethods.csv"; annot_summary_csv=REPORTS_OUTPUT_DIR/"AnnotationMetrics_Summary.csv"; uc_freq_csv=REPORTS_OUTPUT_DIR/"UC_Frequency_Global.csv"; action_freq_csv=REPORTS_OUTPUT_DIR/"Action_Frequency_Global.csv"; uc_method_csv=REPORTS_OUTPUT_DIR/"UC_Method_Report.csv"; bootstrap_csv=REPORTS_OUTPUT_DIR/"BootstrapOverlap.csv"; jaccard_global_csv=REPORTS_OUTPUT_DIR/"JaccardMatrix_Global.csv"; jaccard_per_class_dir=REPORTS_OUTPUT_DIR/"Jaccard_PerClass"; ranking_csv=REPORTS_OUTPUT_DIR/"LLM_Final_Ranking.csv"; added_cls_report_csv=REPORTS_OUTPUT_DIR/"Added_Classes_LLM_Counts.csv"; method_annot_report_csv=REPORTS_OUTPUT_DIR/"Method_Annotation_Details.csv"; cls_focus_csv=REPORTS_OUTPUT_DIR/"Class_Focus_CoreMethods.csv"; placement_consistency_csv=REPORTS_OUTPUT_DIR/"Method_Placement_Consistency.csv"; stable_core_csv=REPORTS_OUTPUT_DIR/"LLM_Stable_Core_Methods.csv"; core_perc_csv=REPORTS_OUTPUT_DIR/"Core_Method_Percentages.csv"; disagreement_csv=REPORTS_OUTPUT_DIR/"Placement_Disagreement_CoreMethods.csv"; method_action_mapping_csv=REPORTS_OUTPUT_DIR/"Method_Action_Mapping_Context.csv"; annot_uc_coverage_csv=REPORTS_OUTPUT_DIR/"UC_Coverage_Annotation.csv"; annot_action_coverage_csv=REPORTS_OUTPUT_DIR/"Action_Coverage_Annotation.csv"; semantic_uc_coverage_csv=REPORTS_OUTPUT_DIR/"UC_Coverage_Semantic.csv"; semantic_action_coverage_csv=REPORTS_OUTPUT_DIR/"Action_Coverage_Semantic.csv"; combined_uc_coverage_csv=REPORTS_OUTPUT_DIR/"UC_Coverage_Combined.csv"; combined_action_coverage_csv=REPORTS_OUTPUT_DIR/"Action_Coverage_Combined.csv"; core_method_semantic_summary_csv = REPORTS_OUTPUT_DIR / "Core_Method_Semantic_Summary.csv"

    # --- Initialize results ---
    structural_df, counts_df_cls_rows, counts_df_files_rows, metrics_per_file = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    method_summary, variability, coverage, diversity, exclusives = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    annot_summary, uc_freq, action_freq, uc_report = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    added_cls_counts, method_annot, boot_overlap, jaccard_global = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cls_focus, plc_agg, disagree_df, stable_df, core_perc_summary = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mapping_df, semantic_sim_summary_df = pd.DataFrame(), pd.DataFrame()
    annot_uc_coverage_df, annot_action_coverage_df = pd.DataFrame(), pd.DataFrame()
    semantic_uc_coverage_df, semantic_action_coverage_df = pd.DataFrame(), pd.DataFrame()
    combined_uc_coverage_df, combined_action_coverage_df = pd.DataFrame(), pd.DataFrame()
    core_method_semantic_summary_df = pd.DataFrame()
    # Store intermediate results needed for combined coverage
    annot_uc_sets, annot_action_sets = None, None
    sem_uc_sets, sem_action_sets = None, None
    embedding_map = {}; annot_action_matches = {} # Initialize as dict
    per_cls_jaccard = {}; consensus = np.nan; core_methods = []

    # --- Calculations ---
    if cache and cache.get_all_files():
        print("\n--- Starting Core Metric Calculations (Part 1) ---")
        # --- (A) Counts & Per-File Basics ---
        print("📊 Calculating method counts..."); counts_df_cls_rows = calculate_counts(cache);
        if not counts_df_cls_rows.empty: counts_df_cls_rows.to_csv(counts_csv); print(f"✔ Saved: {counts_csv}")
        else: print(f"ℹ Skipping save: {counts_csv}")
        print("⚙️ Creating transposed counts...");
        if not counts_df_cls_rows.empty:
            try: # (Transposition logic)
                counts_df_files_rows=counts_df_cls_rows.T; counts_df_files_rows.index.name=COL_FILE;
                if COL_TOTAL_METHODS not in counts_df_files_rows.columns: cls_cols=[c for c in counts_df_files_rows.columns if c in cache.class_names]; counts_df_files_rows[COL_TOTAL_METHODS]=counts_df_files_rows[cls_cols].sum(axis=1).astype(int) if cls_cols else 0
                else: counts_df_files_rows[COL_TOTAL_METHODS]=counts_df_files_rows[COL_TOTAL_METHODS].fillna(0).astype(int)
                counts_df_files_rows[COL_MODEL]=counts_df_files_rows.index.map(lambda f: cache.file_info.get(f,{}).get('model')); counts_df_files_rows[COL_RUN]=counts_df_files_rows.index.map(lambda f: cache.file_info.get(f,{}).get('run'))
                if COL_MODEL not in counts_df_files_rows.columns: counts_df_files_rows[COL_MODEL] = None;
                if COL_RUN not in counts_df_files_rows.columns: counts_df_files_rows[COL_RUN] = None;
                cls_present=sorted([c for c in counts_df_files_rows.columns if c in cache.class_names]); order=[COL_MODEL,COL_RUN]+cls_present+[COL_TOTAL_METHODS]; order=[c for c in order if c in counts_df_files_rows.columns];
                if order: counts_df_files_rows=counts_df_files_rows[order]
                else: print("Warning: No valid columns for transposed counts ordering.")
                for c in cls_present:
                    if c in counts_df_files_rows.columns: counts_df_files_rows[c]=counts_df_files_rows[c].fillna(0).astype(int)
            except Exception as e: print(f" Error preparing transposed counts: {e}"); counts_df_files_rows=pd.DataFrame()
        else: counts_df_files_rows=pd.DataFrame()
        print("📊 Calculating per-file metrics..."); metrics_per_file = cache.get_per_file_method_metrics();
        if not metrics_per_file.empty: metrics_per_file.to_csv(metrics_per_file_csv, index=False); print(f"✔ Saved: {metrics_per_file_csv}")
        else: print(f"ℹ Skipping save: {metrics_per_file_csv}")

        # --- (B) Annotation Details ---
        print("\n--- Annotation Analysis ---")
        print("📊 Generating Method Annotation Details (Target Classes)..."); method_annot = generate_method_annotation_report(cache, target_classes_only=True)
        if not method_annot.empty: method_annot.to_csv(method_annot_report_csv); print(f"✔ Saved: {method_annot_report_csv}")
        else: print(f"ℹ Skipping save: {method_annot_report_csv}")
        cache.method_annot = method_annot # Store for potential use

        # --- (C) Semantic Mapping & Embeddings ---
        print("\n--- Semantic Analysis ---")
        print("📊 Generating detailed method-to-action mapping (Context-Aware)...")
        if gold_map_details:
            # map_methods_to_actions returns df, embedding_map
            mapping_df, embedding_map = map_methods_to_actions(cache, gold_map_details, NLP_MODEL_NAME)
            if not mapping_df.empty: mapping_df.to_csv(method_action_mapping_csv, index=False, float_format="%.4f"); print(f"✔ Saved: {method_action_mapping_csv}")
            else: print(f"ℹ Skipping saving mapping file.")
            cache.mapping_df = mapping_df # Store for potential use
        else: print(f"ℹ Skipping method-to-action mapping (Gold map invalid/missing)."); mapping_df = pd.DataFrame(); embedding_map = {}

        # --- (D) Annotation Coverage ---
        print("\n--- Coverage Analysis ---")
        # Check prerequisites including embedding_map which is needed for action coverage
        if not method_annot.empty and gold_map_details:
            if embedding_map: # Check if embeddings were successfully created
                annot_action_matches = precompute_annotated_action_matches(method_annot, gold_map_details, embedding_map, NLP_MODEL_NAME, SEMANTIC_SIMILARITY_THRESHOLD)
                cache.annot_action_matches = annot_action_matches # Store precomputed results if needed later

                annot_action_coverage_df, annot_action_sets = calculate_action_annotation_coverage(method_annot, gold_map_details, annot_action_matches)
                if not annot_action_coverage_df.empty: annot_action_coverage_df.to_csv(annot_action_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {annot_action_coverage_csv}")
                else: print(f"ℹ Skipping save: {annot_action_coverage_csv}")
            else: print("ℹ Skipping Action Annotation Coverage (Embeddings not available)."); annot_action_coverage_df = pd.DataFrame(); annot_action_sets = None # Ensure vars are defined

            # UC Annotation coverage doesn't need embeddings
            annot_uc_coverage_df, annot_uc_sets = calculate_uc_annotation_coverage(method_annot, gold_map_details)
            if not annot_uc_coverage_df.empty: annot_uc_coverage_df.to_csv(annot_uc_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {annot_uc_coverage_csv}")
            else: print(f"ℹ Skipping save: {annot_uc_coverage_csv}")
        else: print(f"ℹ Skipping Annotation Coverage calculations (prerequisites missing)."); annot_uc_coverage_df = pd.DataFrame(); annot_action_coverage_df = pd.DataFrame(); annot_uc_sets = None; annot_action_sets = None # Ensure vars are defined

        # --- (E) Semantic Coverage ---
        if not mapping_df.empty and gold_map_details:
             semantic_action_coverage_df, sem_action_sets = calculate_action_semantic_coverage(mapping_df, gold_map_details, SEMANTIC_SIMILARITY_THRESHOLD)
             if not semantic_action_coverage_df.empty: semantic_action_coverage_df.to_csv(semantic_action_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {semantic_action_coverage_csv}")
             else: print(f"ℹ Skipping save: {semantic_action_coverage_csv}")

             semantic_uc_coverage_df, sem_uc_sets = calculate_uc_semantic_coverage(mapping_df, gold_map_details, SEMANTIC_SIMILARITY_THRESHOLD)
             if not semantic_uc_coverage_df.empty: semantic_uc_coverage_df.to_csv(semantic_uc_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {semantic_uc_coverage_csv}")
             else: print(f"ℹ Skipping save: {semantic_uc_coverage_csv}")
        else: print(f"ℹ Skipping Semantic Coverage calculations (prerequisites missing)."); semantic_action_coverage_df = pd.DataFrame(); semantic_uc_coverage_df = pd.DataFrame(); sem_action_sets = None; sem_uc_sets = None # Ensure vars defined

        # --- (F) Combined Coverage ---
        print("📊 Calculating Combined Action/UC Coverage...")
        if annot_action_sets is not None and sem_action_sets is not None and gold_map_details:
             combined_action_coverage_df, _ = calculate_action_combined_coverage(annot_action_sets, sem_action_sets, gold_map_details)
             if not combined_action_coverage_df.empty: combined_action_coverage_df.to_csv(combined_action_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {combined_action_coverage_csv}")
             else: print(f"ℹ Skipping save: {combined_action_coverage_csv}")
        else: print(f"ℹ Skipping Combined Action Coverage calculation (prerequisites missing)."); combined_action_coverage_df = pd.DataFrame() # Ensure defined

        if annot_uc_sets is not None and sem_uc_sets is not None and gold_map_details:
             combined_uc_coverage_df, _ = calculate_uc_combined_coverage(annot_uc_sets, sem_uc_sets, gold_map_details)
             if not combined_uc_coverage_df.empty: combined_uc_coverage_df.to_csv(combined_uc_coverage_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {combined_uc_coverage_csv}")
             else: print(f"ℹ Skipping save: {combined_uc_coverage_csv}")
        else: print(f"ℹ Skipping Combined UC Coverage calculation (prerequisites missing)."); combined_uc_coverage_df = pd.DataFrame() # Ensure defined

        # --- (G) Average Semantic Similarity Summary ---
        print("⚙️ Deriving average semantic similarity per model from mapping...")
        semantic_sim_summary_df = pd.DataFrame() # Initialize before try/if
        if not mapping_df.empty and 'Max_Similarity_Score' in mapping_df.columns:
             try:
                 mapping_df['Score_Numeric'] = pd.to_numeric(mapping_df['Max_Similarity_Score'], errors='coerce')
                 if COL_MODEL in mapping_df.columns:
                     semantic_sim_summary_df = mapping_df.groupby(COL_MODEL)['Score_Numeric'].mean().reset_index().rename(columns={'Score_Numeric': COL_AVG_SIM_GOLD})
                     if not semantic_sim_summary_df.empty: print("✔ Derived average semantic similarity scores.")
                     else: print("ℹ Info: Semantic similarity summary empty after grouping.")
                 else: print(f"  [ERROR] Cannot derive average similarity: '{COL_MODEL}' column missing.")
             except Exception as e: print(f"  [ERROR] Failed derive average similarity: {e}"); traceback.print_exc()
        else: print("ℹ Skipping derivation of average similarity (mapping empty or scores missing).")


        # --- (H) Core Methods & Consensus ---
        print("\n--- Core Method & Consensus Analysis ---")
        core_methods = cache.get_core_methods_list(TOP_N_CORE)
        # Initialize dependent dfs
        core_method_semantic_summary_df = pd.DataFrame()
        coverage = pd.DataFrame(); consensus = np.nan
        cls_focus=pd.DataFrame(); plc_agg=pd.DataFrame(); disagree_df=pd.DataFrame(); stable_df=pd.DataFrame(); core_perc_summary=pd.DataFrame()

        if core_methods: # Check if core methods were identified
            print(f"  DEBUG: Entered 'if core_methods:' block (Processing {len(core_methods)} core methods).") # Confirm entry
            # Generate Core Methods List Report
            core_methods_report_df = generate_core_methods_report(cache, TOP_N_CORE)
            if not core_methods_report_df.empty: core_methods_report_df.to_csv(core_methods_csv, index=False); print(f"✔ Saved: {core_methods_csv}")
            else: print(f"ℹ Skipping save: {core_methods_csv}")

            # Calculate Core Method Name Coverage
            coverage, consensus = calculate_coverage(cache, core_methods)
            if not coverage.empty: coverage.to_csv(coverage_csv, index=False); print(f"✔ Saved: {coverage_csv}")
            else: print(f"ℹ Skipping save: {coverage_csv}")
            pd.DataFrame([{"ConsensusStrength": consensus}]).to_csv(consensus_csv, index=False); print(f"✔ Saved: {consensus_csv}")

            # Method Consensus Analysis (Placement etc.) - Depends on method_annot
            # Method Consensus Analysis (Placement etc.) - Depends on method_annot
            if not method_annot.empty:
                core_details = method_annot[method_annot[COL_METHOD_NAME].isin(set(core_methods))].copy()
                if not core_details.empty:
                    print("  Analyzing class focus...")
                    cls_focus = core_details.groupby('Class')[COL_METHOD_NAME].count().reset_index(name='CoreMethodInstances').sort_values('CoreMethodInstances', ascending=False)
                    cls_focus.to_csv(cls_focus_csv, index=False); print(f"✔ Saved: {cls_focus_csv}")

                    print("  Analyzing placement consistency...")
                    plc_agg = core_details.groupby(COL_METHOD_NAME)['Class'].agg(
                        [('Classes_Found_In', lambda x: sorted(list(x.unique()))),
                         ('Num_Unique_Classes', 'nunique'),
                         ('Total_Occurrences', 'count')]
                    ).reset_index()
                    plc_agg['Consistency_Score'] = safe_divide(1, plc_agg['Num_Unique_Classes'], default=0.0)
                    plc_agg.sort_values(['Consistency_Score', COL_METHOD_NAME], ascending=[False, True], inplace=True)
                    plc_agg.to_csv(placement_consistency_csv, index=False); print(f"✔ Saved: {placement_consistency_csv}")

                    print("  Analyzing placement disagreement...")
                    disagree = plc_agg[plc_agg['Num_Unique_Classes'] > 1].copy()
                    details_disagree = []
                    if not disagree.empty:
                        for _, row in disagree.iterrows():
                            m_name = row[COL_METHOD_NAME]
                            instances = core_details[core_details[COL_METHOD_NAME] == m_name]
                            cls_counts = instances['Class'].value_counts()
                            p_cls = cls_counts.idxmax()
                            p_count = cls_counts.max()
                            llms = sorted(list(instances[COL_MODEL].unique()))
                            details_disagree.append({
                                COL_METHOD_NAME: m_name,
                                'Num_Unique_Classes': row['Num_Unique_Classes'],
                                'Classes_Found_In': ", ".join(row['Classes_Found_In']),
                                'Primary_Class': p_cls,
                                'Primary_Class_Count': p_count,
                                'Total_Occurrences': row['Total_Occurrences'],
                                'Involved_LLM_Count': len(llms),
                                'Involved_LLMs': ", ".join(llms)
                            })
                        disagree_df = pd.DataFrame(details_disagree).sort_values(['Num_Unique_Classes', COL_METHOD_NAME], ascending=[False, True])
                        disagree_df.to_csv(disagreement_csv, index=False); print(f"✔ Saved: {disagreement_csv}")
                    else:
                        print(f"ℹ Skipping save: {disagreement_csv} (No disagreements).")

                    print("  Analyzing stable core methods...")
                    stable_core = defaultdict(list)
                    EXPECTED_RUNS = 10 # Should be a global constant ideally
                    run_counts = core_details.groupby([COL_MODEL, COL_METHOD_NAME])[COL_RUN].nunique().reset_index()
                    stable = run_counts[run_counts[COL_RUN] >= EXPECTED_RUNS]
                    for model, grp in stable.groupby(COL_MODEL)[COL_METHOD_NAME]:
                        stable_core[model] = sorted(list(grp))
                    stable_df = pd.DataFrame([{'Model': k, 'Stable_Core_Methods': ', '.join(v)} for k, v in stable_core.items()])
                    if not stable_df.empty:
                        stable_df.to_csv(stable_core_csv, index=False); print(f"✔ Saved: {stable_core_csv}")
                    else:
                        print(f"ℹ Skipping save: {stable_core_csv} (No stable methods).")

                    print("  Calculating core method percentages...")
                    # Ensure coverage (df) is not empty before using set_index
                    print(f"DEBUG: metrics_per_file empty? {metrics_per_file.empty} | Rows: {len(metrics_per_file)}")
                    print(f"DEBUG: coverage empty? {coverage.empty} | Rows: {len(coverage)}")
                    print(f"DEBUG: method_summary empty? {method_summary.empty} | Rows: {len(method_summary)}") # This is the one showing empty
                    if not metrics_per_file.empty and not coverage.empty and not method_summary.empty:
                        core_per_run = core_details.groupby(COL_FILE)[COL_METHOD_NAME].count().reset_index(name='CoreMethodsRunCount')
                        perc_df = pd.merge(metrics_per_file, core_per_run, on=COL_FILE, how='left')
                        perc_df['CoreMethodsRunCount'] = perc_df['CoreMethodsRunCount'].fillna(0).astype(int)
                        perc_df['Percent_Core_Per_Run'] = safe_divide(perc_df['CoreMethodsRunCount'], perc_df[COL_TOTAL_METHODS], default=0.0) * 100
                        t_total_methods_col = f'Total_{COL_TOTAL_METHODS}'
                        if 'Total_Core_Occurrences' in coverage.columns and t_total_methods_col in method_summary.columns:
                             coverage_indexed = coverage.set_index(COL_MODEL)
                             summary_indexed = method_summary.set_index(COL_MODEL)
                             core_sum, total_sum = coverage_indexed['Total_Core_Occurrences'].align(summary_indexed[t_total_methods_col], join='left', fill_value=0)
                             overall_perc = safe_divide(core_sum, total_sum, default=0.0) * 100; overall_perc.name = 'Percent_Core_Overall'
                             avg_perc = perc_df.groupby(COL_MODEL)['Percent_Core_Per_Run'].mean().rename('Avg_Percent_Core_Per_Run')
                             avg_perc, overall_perc = avg_perc.align(overall_perc, join='outer', fill_value=np.nan)
                             core_perc_summary = pd.concat([avg_perc, overall_perc], axis=1).reset_index()
                             if not core_perc_summary.empty:
                                 core_perc_summary.to_csv(core_perc_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {core_perc_csv}")
                             else:
                                 print(f"ℹ Skipping save: {core_perc_csv}")
                        else:
                            print(f"ℹ Skipping core percentage calculation (missing columns).")
                    else:
                        print(f"ℹ Skipping core percentage calculation (prerequisites missing - ID:XYZ).") # THIS IS THE MESSAGE YOU SEE
                else:
                    print("ℹ Skipping detailed consensus analysis (No instances of core methods found in annotation report).")
            else:
                print("ℹ Skipping detailed consensus analysis (Annotation report missing or empty).")

            # Core Method Semantic Summary (Derived from Mapping)
            if not mapping_df.empty:
                print("📊 Analyzing semantic mapping results for Core Methods...")
                core_methods_set = set(core_methods) # Already defined above
                core_mapping_df = mapping_df[mapping_df[COL_METHOD_NAME].isin(core_methods_set)].copy()
                if not core_mapping_df.empty:
                    core_mapping_df['Score_Numeric'] = pd.to_numeric(core_mapping_df['Max_Similarity_Score'], errors='coerce')
                    def get_most_frequent(series): modes = series.mode(); return modes.iloc[0] if not modes.empty else None
                    agg_dict = {
                        'AvgSimilarity': ('Score_Numeric', 'mean'),
                        'Most_Frequent_Best_Match_Action': ('Best_Match_Action', get_most_frequent),
                        'Avg_Score_For_Most_Frequent': ('Score_Numeric',lambda g: g.loc[g['Best_Match_Action'] == (g['Best_Match_Action'].mode().iloc[0] if not g['Best_Match_Action'].mode().empty else None)].mean() if not g['Best_Match_Action'].mode().empty else np.nan),
                        'Generated_In_Classes': ('Class', lambda x: sorted(list(x.unique()))),
                        'Count': ('Max_Similarity_Score', 'size')
                    }
                    try:
                         # Use apply instead of agg with complex lambda accessing outer scope
                         def agg_func(group):
                             res = {}
                             res['AvgSimilarity'] = group['Score_Numeric'].mean()
                             modes = group['Best_Match_Action'].mode()
                             most_freq = modes.iloc[0] if not modes.empty else None
                             res['Most_Frequent_Best_Match_Action'] = most_freq
                             if most_freq is not None:
                                 res['Avg_Score_For_Most_Frequent'] = group.loc[group['Best_Match_Action'] == most_freq, 'Score_Numeric'].mean()
                             else:
                                 res['Avg_Score_For_Most_Frequent'] = np.nan
                             res['Generated_In_Classes'] = sorted(list(group['Class'].unique()))
                             res['Count'] = len(group) # More direct count
                             return pd.Series(res)

                         core_method_semantic_summary_df = core_mapping_df.groupby(
                             [COL_MODEL, COL_METHOD_NAME],
                             observed=True, dropna=False # Groupby handles dropna
                         ).apply(agg_func, include_groups=False).reset_index() # include_groups=False in newer pandas

                         if 'Generated_In_Classes' in core_method_semantic_summary_df.columns:
                              core_method_semantic_summary_df['Generated_In_Classes'] = core_method_semantic_summary_df['Generated_In_Classes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                         core_method_semantic_summary_df.sort_values(by=[COL_MODEL, 'AvgSimilarity'], ascending=[True, False], inplace=True, na_position='last')
                         core_method_semantic_summary_df.to_csv(core_method_semantic_summary_csv, index=False, float_format="%.4f")
                         print(f"✔ Saved: {core_method_semantic_summary_csv}")
                    except Exception as agg_error:
                         print(f"  [ERROR] Aggregation for core method semantic summary failed: {agg_error}"); traceback.print_exc(); core_method_semantic_summary_df = pd.DataFrame()
                else: print(f"ℹ Info: No instances of core methods found in semantic mapping results."); core_method_semantic_summary_df = pd.DataFrame()
            else: print(f"ℹ Skipping Core Method Semantic Summary (mapping_df missing or empty)."); core_method_semantic_summary_df = pd.DataFrame()
        else: # This else corresponds to 'if core_methods:'
            print(f"ℹ Skipping Core Method analysis section (no core methods identified).") # This message should not print based on logs

        # --- (I) Remaining Metrics & Reports ---
        print("\n--- Calculating Remaining Standard Metrics & Reports ---")
        # ... (Aggregated Metrics Summary) ...
        print("📊 Aggregating method metrics...");
        if not metrics_per_file.empty: 
            method_summary = calculate_method_metrics_summary(cache, metrics_per_file)
            print(f"DEBUG_AFTER_CALC: method_summary empty? {method_summary.empty} | Rows: {len(method_summary)}")
        else:
            print(f"ℹ Skipping aggregation: {metrics_per_file_csv} was empty.")
            method_summary = pd.DataFrame()
        if not method_summary.empty: method_summary.to_csv(metrics_summary_csv, index=False); print(f"✔ Saved: {metrics_summary_csv}")
        else: 
            print(f"ℹ Skipping save: {metrics_summary_csv}")
        print(f"DEBUG_MAIN: State of method_summary after its calculation block: Empty? {method_summary.empty}, Rows: {len(method_summary)}")
        
        
        # ... (Variability) ...
        print("📊 Calculating variability...");
        if not counts_df_files_rows.empty: variability = calculate_variability(cache, counts_df_files_rows)
        if not variability.empty: variability.to_csv(variability_csv, index=False); print(f"✔ Saved: {variability_csv}")
        else: print(f"ℹ Skipping save: {variability_csv}")
        # ... (Diversity & Exclusives) ...
        print("📊 Calculating diversity & exclusives...");
        if not counts_df_files_rows.empty: diversity = calculate_diversity(cache, counts_df_files_rows)
        if not diversity.empty: diversity.to_csv(diversity_csv, index=False); print(f"✔ Saved: {diversity_csv}")
        else: print(f"ℹ Skipping save: {diversity_csv}")
        exclusives = generate_exclusive_methods_report(cache);
        if not exclusives.empty: exclusives.to_csv(exclusive_csv, index=False); print(f"✔ Saved: {exclusive_csv}")
        else: print(f"ℹ Skipping save: {exclusive_csv}")
        # ... (Annotation Usage Summary) ...
        print("📊 Calculating annotation metrics summary...");
        if not metrics_per_file.empty: annot_summary = calculate_annotation_metrics(cache, metrics_per_file)
        if not annot_summary.empty: annot_summary.to_csv(annot_summary_csv, index=False); print(f"✔ Saved: {annot_summary_csv}")
        else: print(f"ℹ Skipping save: {annot_summary_csv}")
        # ... (Frequency Reports) ...
        print("📊 Generating frequency reports...");
        uc_freq = generate_uc_frequency_report(cache); action_freq = generate_action_frequency_report(cache); uc_report = generate_uc_method_report(cache)
        if not uc_freq.empty: uc_freq.to_csv(uc_freq_csv, index=False); print(f"✔ Saved: {uc_freq_csv}")
        else: print(f"ℹ Skipping save: {uc_freq_csv}")
        if not action_freq.empty: action_freq.to_csv(action_freq_csv, index=False); print(f"✔ Saved: {action_freq_csv}")
        else: print(f"ℹ Skipping save: {action_freq_csv}")
        if not uc_report.empty: uc_report.to_csv(uc_method_csv, index=False); print(f"✔ Saved: {uc_method_csv}")
        else: print(f"ℹ Skipping save: {uc_method_csv}")
        # ... (Added Classes) ...
        print("📊 Calculating Added Class Counts...");
        added_cls_counts = calculate_added_class_llm_counts(cache)
        if not added_cls_counts.empty: added_cls_counts.to_csv(added_cls_report_csv, index=False); print(f"✔ Saved: {added_cls_report_csv}")
        else: print(f"ℹ Skipping save: {added_cls_report_csv}")
        # ... (Overlap Matrices) ...
        print("📊 Calculating overlap matrices...");
        boot_overlap, jaccard_global = pd.DataFrame(), pd.DataFrame() # Initialize
        if 'calculate_overlap' in globals() and callable(calculate_overlap):
            if not counts_df_files_rows.empty:
                try: boot_overlap, jaccard_global = calculate_overlap(cache, counts_df_files_rows)
                except Exception as e: print(f"  [ERROR] Failed during calculate_overlap: {e}")
                if not boot_overlap.empty and not boot_overlap.isnull().all(axis=None): boot_overlap.to_csv(bootstrap_csv, float_format="%.3f"); print(f"✔ Saved: {bootstrap_csv}")
                else: print(f"ℹ Skipping save: {bootstrap_csv} (Result empty/NaN/Error).")
                if not jaccard_global.empty and not jaccard_global.isnull().all(axis=None): jaccard_global.to_csv(jaccard_global_csv, float_format="%.3f"); print(f"✔ Saved: {jaccard_global_csv}")
                else: print(f"ℹ Skipping save: {jaccard_global_csv} (Result empty/NaN/Error).")
            else: print(f"ℹ Skipping overlap calculations (counts_df_files_rows empty).")
        else: print(" Error: calculate_overlap function not defined.")
        print("📊 Calculating per-class overlap matrices...");
        if 'calculate_per_class_overlap' in globals() and callable(calculate_per_class_overlap):
            per_cls_jaccard = calculate_per_class_overlap(cache);
            if per_cls_jaccard:
                jaccard_per_class_dir.mkdir(exist_ok=True); saved=False
                for cls, matrix in sorted(per_cls_jaccard.items()):
                    if isinstance(matrix, pd.DataFrame) and not matrix.empty and not matrix.isnull().all(axis=None): matrix.to_csv(jaccard_per_class_dir/f"JaccardMatrix_{cls}.csv", float_format="%.3f"); saved=True
                if saved: print(f"✔ Per-class Jaccard matrices saved in: '{jaccard_per_class_dir}'")
                else: print(f"ℹ No valid per-class Jaccard matrices generated.")
            else: print(f"ℹ Skipping per-class Jaccard matrices.")
        else: print(" Error: calculate_per_class_overlap not defined.")

    else: # Cache initialization failed or no files loaded
         print("⚠️ Skipping main calculations block because Cache object is invalid or no JSON files were loaded.")

    # --- Final Ranking ---
    print("\n--- Final Ranking ---")
    print("🏆 Calculating final ranking...")
    dfs_for_rank = { # Gather all potential summary DFs
        'variability': locals().get('variability'), 'method_summary': locals().get('method_summary'),
        'diversity': locals().get('diversity'), 'coverage': locals().get('coverage'),
        'semantic_sim': locals().get('semantic_sim_summary_df'),
        'annot_action_coverage': locals().get('annot_action_coverage_df'),
        'annot_uc_coverage': locals().get('annot_uc_coverage_df'),
        'semantic_action_coverage': locals().get('semantic_action_coverage_df'),
        'semantic_uc_coverage': locals().get('semantic_uc_coverage_df'),
        'combined_action_coverage': locals().get('combined_action_coverage_df'),
        'combined_uc_coverage': locals().get('combined_uc_coverage_df')
    }
    # Define metrics TO USE in ranking
    metrics_for_rank = {
        COL_CV: 'lower', COL_TOTAL_REDUNDANCY: 'lower', COL_NORM_ENTROPY: 'higher',
        COL_UNIQUE_CORE: 'higher', COL_AVG_SIM_GOLD: 'higher',
        'Comb_Action_Coverage_Percent': 'higher',
        #'Avg_Per_UC_Comb_Action_Coverage': 'higher', # Can use this instead of overall % if desired
        'Comb_UC_Hit_Coverage_Percent': 'higher' # Using the hit rate from combined UC coverage
    }
    available_metrics = {}; all_models_found = set()
    for name, df_obj in dfs_for_rank.items(): # Collect available metric data
        if isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
            for metric_col in metrics_for_rank:
                 if metric_col in df_obj.columns and COL_MODEL in df_obj.columns:
                     df_filtered = df_obj[[COL_MODEL, metric_col]].dropna(subset=[metric_col]).copy()
                     if not df_filtered.empty: available_metrics[metric_col] = df_filtered; all_models_found.update(df_filtered[COL_MODEL].unique());
    if len(available_metrics) >= 1 and all_models_found: # Proceed if we have models and at least one metric
        rank_df = pd.DataFrame({COL_MODEL: sorted(list(all_models_found))})
        for metric_col, df_to_merge in available_metrics.items(): rank_df = pd.merge(rank_df, df_to_merge, on=COL_MODEL, how='left')
        norm_cols_present=[]
        for col, direction_simple in metrics_for_rank.items(): # Normalize available metrics
            if col in rank_df.columns:
                norm_col=f"{col}_Norm"; rank_df[col] = pd.to_numeric(rank_df[col], errors='coerce'); vals=rank_df[col].dropna()
                if len(vals)>1: min_v, max_v = vals.min(), vals.max(); rank_df[norm_col]=(np.where(rank_df[col].notna(), 0.5, np.nan) if min_v==max_v else (rank_df[col]-min_v)/(max_v-min_v)); rank_df[norm_col]=rank_df[norm_col].apply(lambda x: 1.0-x if direction_simple=='lower' and pd.notna(x) else x); norm_cols_present.append(norm_col)
                elif len(vals)==1: rank_df[norm_col]=np.where(rank_df[col].notna(), 0.5, np.nan); norm_cols_present.append(norm_col)
                else: rank_df[norm_col]=np.nan
        if norm_cols_present: rank_df[COL_FINAL_SCORE]=rank_df[norm_cols_present].mean(axis=1, skipna=True); print(f"✔ Calculated final score using: {norm_cols_present}")
        else: print("ℹ Info: No metrics normalized."); rank_df[COL_FINAL_SCORE]=np.nan
        f_cols_order=[COL_MODEL]+[c for c in metrics_for_rank if c in rank_df.columns]+norm_cols_present+[COL_FINAL_SCORE]; f_cols_present=[c for c in f_cols_order if c in rank_df.columns]; rank_df=rank_df[f_cols_present].sort_values(COL_FINAL_SCORE, ascending=False, na_position='last')
        rank_df.to_csv(ranking_csv, index=False, float_format="%.3f"); print(f"✔ Saved: {ranking_csv}")
    else: print(f"ℹ Skipping composite ranking ({ranking_csv}): No valid metrics data or models found.")

    print("\n✨ Metrics pipeline execution finished.")


if __name__ == "__main__":
    main()