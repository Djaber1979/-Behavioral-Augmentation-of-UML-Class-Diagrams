# fmt: off
from __future__ import annotations # Must be the first non-comment line
# fmt: on

import json
import math
import os
import re # Added for name splitting
import sys
import traceback
import inspect
import itertools as it

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np # Added for NaN handling
import pandas as pd
from scipy.stats import t
import torch # Added for tensor operations in mapping

# --- NLP Imports ---
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim # Direct import
except ImportError:
    print("Error: sentence-transformers library not found.")
    print("Please install it: pip install sentence-transformers torch")
    # Depending on your system, you might need a different backend like tensorflow or jax instead of torch
    sys.exit(1)

# --- Configuration ---
BASELINE_JSON_FNAME = "methodless.json"  # Assumes baseline is converted and in JSON_INPUT_DIR
GOLD_STANDARD_MAP_FNAME = "uc_action_method_map.json"  # Ground truth mapping file (expected in root)
TOP_N_CORE = 35  # Use top 35 core methods
CLASS_NAMES = [
    "ValidationResult", "Coordinates", "Address", "TimeRange", "OpeningHours",
    "UserAccount", "UserProfile", "UserCredentials", "RoleManager",
    "ServiceRequest", "CollectionRequest", "TransportRequest",
    "RoutePlan", "WasteJourney", "Product",
    "SearchFilter", "CollectionPoint", "NotificationTemplate",
    "NotificationService",
    # Added classes mentioned in gold map but maybe missing from baseline
    "ServiceProvider", "PlatformService"
]  # Baseline classes to focus metrics on

# --- Directory Paths ---
JSON_INPUT_DIR = Path("JSON")  # Directory containing *.json files (including baseline)
REPORTS_OUTPUT_DIR = Path("reports")  # Directory to save CSV results
# Ensure reports directory exists
REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- NLP Model ---
NLP_MODEL_NAME = "all-mpnet-base-v2" # Using the preferred model

# --- Constants for Column Names ---
COL_FILE = "file"
COL_MODEL = "Model"
COL_RUN = "Run"
COL_CV = "CV"
COL_TOTAL_REDUNDANCY = "Total_Redundancy"
COL_NORM_ENTROPY = "NormalizedEntropy"
COL_UNIQUE_CORE = "Unique_Core_Methods"
COL_FINAL_SCORE = "Final_Score"
COL_TOTAL_METHODS = "Target_Class_Methods"  # Methods ONLY in target classes
COL_REDUNDANCY = "Redundancy"
COL_PARAM_RICHNESS = "ParamRichness"
COL_RETURN_COMPLETENESS = "ReturnTypeCompleteness"
COL_PARAM_TYPE_COMPLETENESS = "ParamTypeCompleteness"
COL_PERCENT_UC = "Percentage_Methods_With_UC"
COL_METHODS_WITH_RETURN = "Methods_With_ReturnType_Count"
COL_PARAMS_WITH_TYPE = "Params_With_Type_Count"
COL_METHOD_NAME = "MethodName"
COL_GLOBAL_FREQ = "GlobalFrequency"
COL_METHODS_WITH_UC = "Methods_With_Any_UC"
COL_COUNT_UC_ACTION = "Count_UC_Action"
COL_COUNT_UC_ONLY = "Count_UC_Only"
COL_COUNT_ACTION_ONLY = "Count_Action_Only"
COL_COUNT_NONE = "Count_None"
COL_EXTRA_METHODS = "Extra_Methods_Count"
COL_AVG_SIM_GOLD = "Avg_Similarity_To_Gold" # Average derived from detailed mapping

# --- Random Seed ---
np.random.seed(42)


# --- Helper Functions ---

def split_method_name(name):
    """Splits camelCase or snake_case names into space-separated words."""
    if not name:
        return ""
    # Add space before uppercase letters (but not the first one)
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', name)
    # Add space before uppercase letters followed by uppercase then lowercase
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1)
    # Replace underscores with spaces
    s3 = s2.replace('_', ' ')
    # Convert to lower case and remove potential multiple spaces
    return ' '.join(s3.lower().split())

def model_key(filename_or_stem):
    """Extracts the model identifier by removing the '_run<N>' suffix."""
    if isinstance(filename_or_stem, Path):
        stem = filename_or_stem.stem
    elif isinstance(filename_or_stem, str):
        stem = Path(filename_or_stem).stem
    else:
        print(
            f"[model_key WARNING] Unexpected type: {type(filename_or_stem)}"
        )
        return str(filename_or_stem)
    model_name = re.sub(r"_run\d+$", "", stem)
    return model_name


def compare_structures(baseline_elements, enriched_elements):
    """Compares two structure element dictionaries (sets)."""
    report = {}
    total_baseline = 0
    preserved = 0
    all_keys = set(baseline_elements.keys()) | set(enriched_elements.keys())
    for key in sorted(list(all_keys)):
        base_set = baseline_elements.get(key, set())
        enrich_set = enriched_elements.get(key, set())
        pres_count = len(base_set & enrich_set)
        miss_count = len(base_set - enrich_set)
        add_count = len(enrich_set - base_set)
        total_cat = len(base_set)
        if key in baseline_elements:
            total_baseline += total_cat
            preserved += pres_count
        report[f"{key}_preserved"] = pres_count
        report[f"{key}_missing"] = miss_count
        report[f"{key}_added"] = add_count
        report[f"{key}_total_baseline"] = total_cat
    pres_pct = (
        round((preserved / total_baseline) * 100, 2)
        if total_baseline > 0
        else 100.0
    )
    report["Overall_Preservation_%"] = pres_pct
    report["Total_Baseline_Elements"] = total_baseline
    report["Total_Preserved_Elements"] = preserved
    report["Total_Added_Elements"] = sum(
        len(enriched_elements.get(k, set()) - baseline_elements.get(k, set()))
        for k in enriched_elements
    )
    return report


def extract_structure_from_json(json_data):
    """Extracts structural elements into sets from the parsed JSON data."""
    elements = {
        "packages": set(),
        "enums": set(),
        "enum_values": set(),
        "classes": set(),
        "attributes": set(),
        "relationships": set(),
    }
    if not json_data or not isinstance(json_data, dict):
        return elements
    elements["packages"] = set(json_data.get("packages", []))
    for enum in json_data.get("enums", []):
        name = enum.get("name")
        if name:
            elements["enums"].add(name)
            [
                elements["enum_values"].add(f"{name}::{v}")
                for v in enum.get("values", [])
            ]
    for cls in json_data.get("classes", []):
        cls_name = cls.get("name")
        if cls_name:
            elements["classes"].add(cls_name)
            for attr in cls.get("attributes", []):
                a_name = attr.get("name")
                a_type = attr.get("type")
                if a_name and a_type is not None:
                    norm_type = (
                        " ".join(a_type.split())
                        if isinstance(a_type, str)
                        else a_type
                    )
                    elements["attributes"].add(
                        f"{cls_name}::{a_name}: {norm_type}"
                    )
    for rel in json_data.get("relationships", []):
        src = rel.get("source")
        tgt = rel.get("target")
        if src and tgt:
            sym = rel.get("type_symbol", "--")
            lbl = rel.get("label")
            s_card = rel.get("source_cardinality")
            t_card = rel.get("target_cardinality")
            rel_str = (
                f"{src}"
                + (f" {s_card}" if s_card else "")
                + f" {sym}"
                + (f" {t_card}" if t_card else "")
                + f" {tgt}"
                + (f" : {lbl}" if lbl else "")
            )
            elements["relationships"].add(rel_str)
    return elements


def safe_divide(numerator_col, denominator_col, df, default=np.nan):
    """Safely divides two columns of a DataFrame, handling NaNs and zeros in denominator."""
    if numerator_col in df.columns and denominator_col in df.columns:
        num = df[numerator_col]
        den = df[denominator_col]
        den_num = pd.to_numeric(den, errors="coerce")
        return np.where(
            den_num.notna() & (den_num != 0), num / den_num, default
        )
    else:
        print(
            f"Warn: safe_divide missing {numerator_col} or {denominator_col}."
        )
        return pd.Series(default, index=df.index)


# --- Metric Cache Class ---
class MetricCache:
    """Caches data parsed from JSON files for metric calculation."""

    def __init__(
        self,
        json_files_dir: Path,
        baseline_json_fname: str,
        class_names: list[str],
    ):
        self.json_files_dir = json_files_dir
        self.baseline_fname = baseline_json_fname
        self.class_names = class_names
        self.target_class_set = set(self.class_names)
        self.baseline_json_data = None
        baseline_path = self.json_files_dir / self.baseline_fname
        if baseline_path.is_file():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    self.baseline_json_data = json.load(f)
            except Exception as e:
                print(f"❌ Error loading baseline JSON '{baseline_path}': {e}.")
        else:
            print(f"Warning: Baseline JSON not found: {baseline_path}")

        self.files_paths = sorted(
            [
                p
                for p in self.json_files_dir.glob("*.json")
                if p.name != self.baseline_fname
            ],
            key=lambda p: p.name,
        )
        self.files = [p.name for p in self.files_paths]
        if not self.files:
             # Make this non-fatal, allow script to run but warn heavily
             print(f"⚠️⚠️⚠️ WARNING: No generated JSON files found in '{self.json_files_dir}'. Most calculations will be skipped.")
             self.json_data = {}
             self.files_paths = []
        else:
            self.json_data = {}
            loaded = []
            print(f"Attempting to load {len(self.files_paths)} generated JSON files...")
            for p in self.files_paths:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self.json_data[p.stem] = json.load(f)
                        loaded.append(p.name)
                except Exception as e:
                    print(f"❌ Error loading generated JSON {p.name}: {e}. Skipping.")
            self.files = sorted(loaded) # Only include successfully loaded files
            self.files_paths = [self.json_files_dir / f for f in self.files] # Update paths too
            if not self.files:
                print("⚠️⚠️⚠️ WARNING: No generated JSON files were successfully loaded. Most calculations will be skipped.")

        self.file_info = {}
        for f in self.files: # Iterate only over successfully loaded files
            stem = Path(f).stem
            if stem in self.json_data: # Check ensures we only process loaded files
                m = model_key(stem)
                r_m = re.search(r"_run(\d+)", stem)
                run = r_m.group(1) if r_m else "1"
                self.file_info[f] = {"stem": stem, "model": m, "run": run}

        # Initialize attributes
        self.structure_reports = {}
        self.baseline_structure = {}
        self.detailed_methods = {} # All methods per file
        self.class_detailed_methods = {} # Only target class methods per file/class
        self.extra_method_counts = {}
        self._global_method_details_list = []
        self._global_method_counter = Counter()
        self._global_uc_counter = Counter()
        self._global_action_counter = Counter()
        self._global_unique_method_names = set()

        # Parse data only if files were loaded
        if self.files:
            try:
                self._parse()
                self._post_parse_aggregations()
            except Exception as e:
                print(f"❌❌❌ CRITICAL ERROR during cache parsing/aggregation: {e}")
                traceback.print_exc()
                # Mark cache as invalid? For now, rely on empty attributes.

    def _parse(self):
        """ Parses loaded JSON data to populate cache attributes and count extra methods. """
        print("Parsing baseline structure...")
        if self.baseline_json_data:
            self.baseline_structure = extract_structure_from_json(
                self.baseline_json_data
            )
        else:
            self.baseline_structure = extract_structure_from_json(None)
            print("Warning: Baseline structure is empty.")

        print(f"Parsing {len(self.files)} successfully loaded generated JSON files...")
        if not self.json_data:
             print("Warning: No generated JSON data available to parse.")
             return # Exit early if no data

        for json_filename in self.files: # Iterate only over successfully loaded files
            stem = self.file_info.get(json_filename, {}).get('stem')
            if not stem or stem not in self.json_data:
                print(f"Warning: Skipping file '{json_filename}' - stem/data mismatch.")
                continue
            file_json_data = self.json_data[stem]

            # --- Structure Extraction & Comparison ---
            try:
                generated_structure = extract_structure_from_json(file_json_data)
                if self.baseline_structure:
                    self.structure_reports[json_filename] = compare_structures(
                        self.baseline_structure, generated_structure
                    )
                else:
                     self.structure_reports[json_filename] = {} # Empty report if no baseline
            except Exception as e:
                 print(f"  [ERROR] Structure extraction/comparison failed for {json_filename}: {e}")
                 self.structure_reports[json_filename] = {}

            # --- Method Extraction & Extra Method Counting ---
            file_all_methods = []
            file_target_class_methods = defaultdict(list)
            extra_methods_in_file = 0
            try:
                for class_info in file_json_data.get("classes", []):
                    class_name = class_info.get("name")
                    if not class_name: continue
                    is_target = class_name in self.target_class_set
                    for m_info in class_info.get("methods", []):
                        m_details = m_info.copy()
                        # Ensure essential keys exist even if method object is minimal
                        m_details.setdefault("name", None)
                        m_details.setdefault("parameters", [])
                        m_details.setdefault("return_type", "void")
                        m_details.setdefault("annotation", {})
                        m_details.setdefault("signature", "")
                        m_details.setdefault("visibility", "+")

                        if not m_details.get("name"):
                             print(f"  [WARN] Skipping method without name in {json_filename}, class {class_name}")
                             continue

                        m_details[COL_FILE] = json_filename
                        m_details["class"] = class_name
                        params = m_details.get("parameters", [])
                        m_details["param_count"] = len(params)
                        m_details["params_str"] = ", ".join(
                            [f"{p.get('name','?')}:{p.get('type','?')}" for p in params]
                        )
                        ret_type = m_details.get("return_type")
                        m_details["has_return_type"] = bool(ret_type and ret_type.lower() != "void")

                        annot = m_details.get("annotation") # Already ensured dict by setdefault
                        m_details["has_uc_annotation"] = bool(annot and annot.get("uc_references"))
                        raw_ucs = annot.get("uc_references", [])
                        if isinstance(raw_ucs, list):
                            m_details["ucs"] = [str(uc) for uc in raw_ucs if uc is not None] # Ensure string, skip None
                        elif raw_ucs is not None: # Handle single non-list value if present
                            m_details["ucs"] = [str(raw_ucs)]
                        else: # Handle None or empty
                            m_details["ucs"] = []

                        m_details["action"] = annot.get("uc_action", "")
                        m_details["full_sig"] = m_details["signature"]

                        file_all_methods.append(m_details)
                        if is_target:
                            file_target_class_methods[class_name].append(m_details)
                        else:
                            extra_methods_in_file += 1
            except Exception as e:
                 print(f"  [ERROR] Method extraction failed for {json_filename}: {e}")
                 traceback.print_exc()
                 file_all_methods = []
                 file_target_class_methods = defaultdict(list)
                 extra_methods_in_file = 0

            self.detailed_methods[json_filename] = file_all_methods
            self.class_detailed_methods[json_filename] = dict(file_target_class_methods)
            self.extra_method_counts[json_filename] = extra_methods_in_file

    def _post_parse_aggregations(self):
        """ Aggregate global counts after parsing all files. """
        print("Aggregating global metrics...")
        temp_list = []
        if hasattr(self, 'detailed_methods') and isinstance(self.detailed_methods, dict):
            for m_list in self.detailed_methods.values():
                if isinstance(m_list, list):
                    for m_detail in m_list:
                        if isinstance(m_detail, dict) and all(k in m_detail for k in [COL_FILE, 'name', 'class']):
                            temp_list.append(m_detail)
        self._global_method_details_list = temp_list

        if self._global_method_details_list:
            self._global_method_counter = Counter(m['name'] for m in self._global_method_details_list)
            self._global_uc_counter = Counter(uc for m in self._global_method_details_list for uc in m.get('ucs', []))
            self._global_action_counter = Counter(m['action'] for m in self._global_method_details_list if m.get('action'))
            self._global_unique_method_names = set(self._global_method_counter.keys())
            print(f"Aggregated {len(self._global_method_details_list)} methods globally.")
        else:
            print("Warning: No global method details found for aggregation.")
            self._global_method_counter = Counter()
            self._global_uc_counter = Counter()
            self._global_action_counter = Counter()
            self._global_unique_method_names = set()

    # --- Public Methods ---
    def get_structure_reports_df(self):
        if not hasattr(self, 'structure_reports') or not self.structure_reports:
            print("Warning: No structure reports available.")
            return pd.DataFrame()
        reports = [self.structure_reports.get(f, {}) for f in self.files]
        if not any(reports): return pd.DataFrame()
        df = pd.DataFrame(reports).fillna(0)
        df[COL_FILE] = self.files
        df[COL_MODEL] = df[COL_FILE].map(lambda f: self.file_info.get(f, {}).get('model'))
        df[COL_RUN] = df[COL_FILE].map(lambda f: self.file_info.get(f, {}).get('run'))
        if COL_MODEL not in df.columns: df[COL_MODEL] = None
        if COL_RUN not in df.columns: df[COL_RUN] = None
        for c in df.columns:
            if any(k in c for k in ['preserved','missing','added','_total_baseline','Total_Baseline','Total_Preserved','Total_Added']):
                 try: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                 except: pass
        cols = [COL_FILE, COL_MODEL, COL_RUN] + sorted([c for c in df.columns if c not in [COL_FILE, COL_MODEL, COL_RUN]])
        cols = [c for c in cols if c in df.columns]
        return df[cols] if cols else pd.DataFrame()

    def get_per_file_method_metrics(self):
        """Calculates per-file metrics based ONLY on methods within TARGET classes."""
        if not hasattr(self, 'class_detailed_methods') or not self.class_detailed_methods:
             print("Warning: Target class methods not parsed, cannot calculate per-file metrics.")
             return pd.DataFrame()
        if not hasattr(self, 'extra_method_counts'): self.extra_method_counts = {}

        rows = []
        for fname in self.files:
            methods = [m for m_list in self.class_detailed_methods.get(fname, {}).values() for m in m_list]
            total_target = len(methods)
            extra = self.extra_method_counts.get(fname, 0)
            row = {COL_FILE: fname, COL_TOTAL_METHODS: total_target, COL_EXTRA_METHODS: extra}
            if total_target == 0:
                 row.update({
                     "Unique_Method_Names": 0, COL_REDUNDANCY: np.nan,
                     "DuplicatedOccurrences_Name": 0, "DuplicatedOccurrences_NameParam": 0,
                     "DuplicatedOccurrences_FullSig": 0, COL_PARAM_RICHNESS: 0.0,
                     COL_RETURN_COMPLETENESS: 0.0, COL_METHODS_WITH_RETURN: 0,
                     "Visibility_Public": 0, "Visibility_Private": 0,
                     "Visibility_Protected": 0, "Visibility_Package": 0,
                     COL_METHODS_WITH_UC: 0, COL_PERCENT_UC: 0.0,
                     "Total_UC_References_File": 0, "Unique_UCs_File": 0,
                     "Unique_Actions_File": 0,
                     COL_COUNT_UC_ACTION: 0, COL_COUNT_UC_ONLY: 0,
                     COL_COUNT_ACTION_ONLY: 0, COL_COUNT_NONE: 0,
                     COL_PARAMS_WITH_TYPE: 0, "Total_Params_Count": 0,
                     COL_PARAM_TYPE_COMPLETENESS: np.nan
                 })
            else:
                names=[m.get('name','') for m in methods]; name_params=[f"{m.get('name','')}({m.get('params_str','')})" for m in methods]; sigs=[m.get('full_sig','') for m in methods]
                name_ctr=Counter(names); param_ctr=Counter(name_params); sig_ctr=Counter(sigs); unique_names=set(names)
                row["DuplicatedOccurrences_Name"]=sum(c-1 for c in name_ctr.values() if c>1); row["DuplicatedOccurrences_NameParam"]=sum(c-1 for c in param_ctr.values() if c>1); row["DuplicatedOccurrences_FullSig"]=sum(c-1 for c in sig_ctr.values() if c>1)
                row["Unique_Method_Names"]=len(unique_names); row[COL_REDUNDANCY]=total_target/len(unique_names) if unique_names else np.nan
                p_counts=[m.get('param_count',0) for m in methods]; row[COL_PARAM_RICHNESS]=np.mean(p_counts) if p_counts else 0.0
                m_ret=sum(m.get('has_return_type',False) for m in methods); row[COL_METHODS_WITH_RETURN]=m_ret; row[COL_RETURN_COMPLETENESS]=m_ret/total_target
                vis=Counter(m.get('visibility','+') for m in methods); row.update({"Visibility_Public":vis.get('+',0),"Visibility_Private":vis.get('-',0),"Visibility_Protected":vis.get('#',0),"Visibility_Package":vis.get('~',0)})
                c_uc_act,c_uc_only,c_act_only,c_none,t_uc_refs,m_any_uc=0,0,0,0,0,0; u_ucs,u_acts=set(),set()
                for m in methods:
                    has_ucs=bool(m.get('ucs')); has_action=bool(m.get('action'))
                    if has_ucs and has_action: c_uc_act+=1
                    elif has_ucs: c_uc_only+=1
                    elif has_action: c_act_only+=1
                    else: c_none+=1
                    if has_ucs:
                        m_any_uc+=1
                        ucs=m.get('ucs',[]) # Already list of strings
                        t_uc_refs+=len(ucs)
                        u_ucs.update(ucs)
                    if has_action: u_acts.add(m['action'])
                row.update({COL_METHODS_WITH_UC:m_any_uc, COL_PERCENT_UC:m_any_uc/total_target, COL_COUNT_UC_ACTION:c_uc_act, COL_COUNT_UC_ONLY:c_uc_only, COL_COUNT_ACTION_ONLY:c_act_only, COL_COUNT_NONE:c_none, "Total_UC_References_File":t_uc_refs, "Unique_UCs_File":len(u_ucs), "Unique_Actions_File":len(u_acts)})
                p_w_type=sum(len([p for p in m.get('parameters',[]) if p.get('type')]) for m in methods); t_params=sum(m.get('param_count',0) for m in methods)
                row[COL_PARAMS_WITH_TYPE]=p_w_type; row["Total_Params_Count"]=t_params; row[COL_PARAM_TYPE_COMPLETENESS]=p_w_type/t_params if t_params>0 else np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
             df[COL_MODEL]=df[COL_FILE].map(lambda f:self.file_info.get(f,{}).get('model'))
             df[COL_RUN]=df[COL_FILE].map(lambda f:self.file_info.get(f,{}).get('run'))
             if COL_MODEL not in df.columns: df[COL_MODEL] = None
             if COL_RUN not in df.columns: df[COL_RUN] = None
             base_cols=[COL_FILE,COL_MODEL,COL_RUN]; metric_cols=sorted([c for c in df.columns if c not in base_cols]); cols=base_cols+metric_cols;
             cols = [c for c in cols if c in df.columns]
             df=df[cols]
        return df

    def get_global_method_counter(self): return getattr(self, '_global_method_counter', Counter())
    def get_global_uc_counter(self): return getattr(self, '_global_uc_counter', Counter())
    def get_global_action_counter(self): return getattr(self, '_global_action_counter', Counter())
    def get_core_methods_list(self, top_n):
        counter = self.get_global_method_counter()
        return [m for m, _ in counter.most_common(top_n)] if counter else []
    def get_exclusive_methods_list(self):
        excl=defaultdict(list);
        ctr = self.get_global_method_counter(); details = self.get_method_details_list(); info = getattr(self, 'file_info', {})
        if not ctr or not details or not info:
             print("Warning: Cannot generate exclusive methods list, prerequisite data missing.")
             return {}
        for m in details:
             m_name = m.get('name'); m_file = m.get(COL_FILE)
             if m_name and m_file and ctr.get(m_name, 0) == 1:
                 model=info.get(m_file,{}).get('model')
                 if model: excl[model].append(m_name)
        return dict(excl)
    def get_method_details_list(self): return getattr(self, '_global_method_details_list', [])
    def get_class_detailed_methods(self): return getattr(self, 'class_detailed_methods', {})
    def get_all_files(self): return getattr(self, 'files', []) # Return loaded files

# --- End of MetricCache Class ---


# --- Metric Calculation Functions --- (Assume largely unchanged unless noted)

def calculate_counts(cache: MetricCache):
    """Calculates method counts per class per file (only methods within target classes)."""
    if not hasattr(cache, 'class_detailed_methods'):
        print("[calculate_counts] Error: MetricCache missing 'class_detailed_methods'. Cannot proceed.")
        return pd.DataFrame()

    sorted_files = cache.get_all_files()
    class_names = cache.class_names
    if not sorted_files:
        print("[calculate_counts] Warning: No files loaded in cache to calculate counts for.")
        return pd.DataFrame(index=class_names) # Return empty DF with index

    df = pd.DataFrame(0, index=class_names, columns=sorted_files, dtype=int)
    df.index.name = "Classes"
    file_totals = {f: 0 for f in sorted_files}
    found_any = False
    class_methods_data = cache.get_class_detailed_methods()

    for fname in sorted_files:
        cls_data = class_methods_data.get(fname, {})
        if fname not in df.columns: continue
        f_total = 0
        for cls_name, m_list in cls_data.items():
            if cls_name in df.index:
                count = len(m_list)
                if count > 0:
                    df.loc[cls_name, fname] = count; found_any = True
                f_total += count
        file_totals[fname] = f_total

    if not found_any: print("[calculate_counts] Warning: No target class methods found in any loaded file.")
    if not df.columns.empty:
        totals = pd.Series(file_totals, index=df.columns, name=COL_TOTAL_METHODS).fillna(0).astype(int)
        df = pd.concat([df, totals.to_frame().T])
    else: print("[calculate_counts] Warning: DataFrame has no columns (no files processed).")

    try:
        num_cols = df.select_dtypes(include=np.number).columns;
        if not num_cols.empty: df[num_cols] = df[num_cols].fillna(0).astype(int)
    except Exception as e: print(f"[calculate_counts] Warning: Int conversion failed: {e}")
    return df

def calculate_method_metrics_summary(
    cache: MetricCache, per_file_method_metrics_df: pd.DataFrame
):
    """Aggregates per-file method metrics (calculated on target classes) by model."""
    if per_file_method_metrics_df.empty:
        print(
            "Warning: Input per_file_method_metrics_df is empty for summary calculation."
        )
        return pd.DataFrame()
    mean_cols = [
        COL_REDUNDANCY,
        COL_PARAM_RICHNESS,
        COL_RETURN_COMPLETENESS,
        COL_PARAM_TYPE_COMPLETENESS,
        COL_PERCENT_UC,
    ]
    sum_cols = [
        COL_TOTAL_METHODS,
        "Unique_Method_Names",
        "DuplicatedOccurrences_Name",
        "DuplicatedOccurrences_NameParam",
        "DuplicatedOccurrences_FullSig",
        "Visibility_Public",
        "Visibility_Private",
        "Visibility_Protected",
        "Visibility_Package",
        COL_METHODS_WITH_UC,
        "Total_UC_References_File",
        "Unique_UCs_File",
        "Unique_Actions_File",
        COL_PARAMS_WITH_TYPE,
        "Total_Params_Count",
        COL_METHODS_WITH_RETURN,
        COL_EXTRA_METHODS,
        COL_COUNT_UC_ACTION,
        COL_COUNT_UC_ONLY,
        COL_COUNT_ACTION_ONLY,
        COL_COUNT_NONE,
    ]
    mean_present = [
        c for c in mean_cols if c in per_file_method_metrics_df.columns
    ]
    sum_present = [
        c for c in sum_cols if c in per_file_method_metrics_df.columns
    ]
    summary_mean = pd.DataFrame()
    summary_sum = pd.DataFrame()

    # Ensure COL_MODEL exists before grouping
    if COL_MODEL not in per_file_method_metrics_df.columns:
         print(f"Error: Missing '{COL_MODEL}' column in per_file_method_metrics_df. Cannot calculate summary.")
         return pd.DataFrame()

    if mean_present:
        summary_mean = (
            per_file_method_metrics_df.groupby(COL_MODEL)[mean_present]
            .mean(numeric_only=True)
            .reset_index()
        )
    if sum_present:
        summary_sum = (
            per_file_method_metrics_df.groupby(COL_MODEL)[sum_present]
            .sum(numeric_only=True)
            .reset_index()
        )

    # Check if grouping produced results
    if summary_mean.empty and summary_sum.empty:
         # If input df wasn't empty but grouping yielded nothing (e.g., only one model, no variance)
         # Still return structure based on unique models found
        models_in_df = per_file_method_metrics_df[COL_MODEL].unique()
        if len(models_in_df) > 0:
             print("Warning: Method summary aggregation yielded empty results, returning structure with NaNs.")
             # Create an empty DF with the right columns and models
             all_summary_cols = [COL_MODEL] + [f"Avg_{c}_per_file" for c in mean_cols] + [f"Total_{c}" for c in sum_cols] + [
                 COL_TOTAL_REDUNDANCY, 'Total_ParamRichness', 'Total_ParamTypeCompleteness',
                 'Total_ReturnTypeCompleteness', 'Total_Percentage_Methods_With_UC'
             ]
             method_summary = pd.DataFrame(columns=all_summary_cols)
             method_summary[COL_MODEL] = models_in_df
             return method_summary # Return structure with NaNs
        else:
             print("Warning: No models found in per_file_method_metrics_df.")
             return pd.DataFrame() # Truly empty

    # Merge or construct the summary dataframe
    if summary_mean.empty:
        method_summary = summary_sum
        for c in reversed(mean_present): # Add placeholder columns for missing averages
             method_summary.insert(1, f"Avg_{c}_per_file", np.nan)
    elif summary_sum.empty:
        method_summary = summary_mean
        for c in sum_present: # Add placeholder columns for missing totals
             method_summary[f"Total_{c}"] = 0 if "Count" in c or "Occurrences" in c or "Visibility" in c else np.nan
    else:
        method_summary = pd.merge(summary_mean, summary_sum, on=COL_MODEL, how="outer")

    # Rename columns
    AVG_PREFIX = "Avg_"
    TOTAL_PREFIX = "Total_"
    PER_FILE_SUFFIX = "_per_file"
    mean_map = {c: f"{AVG_PREFIX}{c}{PER_FILE_SUFFIX}" for c in mean_present}
    sum_map = {c: f"{TOTAL_PREFIX}{c}" for c in sum_present}
    method_summary.rename(columns=mean_map, inplace=True)
    method_summary.rename(columns=sum_map, inplace=True)

    # Calculate derived total metrics
    t_total_methods = f"{TOTAL_PREFIX}{COL_TOTAL_METHODS}"
    t_unique_names = f"{TOTAL_PREFIX}Unique_Method_Names"
    t_params_count = f"{TOTAL_PREFIX}Total_Params_Count"
    t_params_w_type = f"{TOTAL_PREFIX}{COL_PARAMS_WITH_TYPE}"
    t_methods_w_ret = f"{TOTAL_PREFIX}{COL_METHODS_WITH_RETURN}"
    t_methods_w_uc = f"{TOTAL_PREFIX}{COL_METHODS_WITH_UC}"

    # Check if necessary columns exist before calculation
    if t_total_methods in method_summary.columns and t_unique_names in method_summary.columns:
        method_summary[COL_TOTAL_REDUNDANCY] = safe_divide(t_total_methods, t_unique_names, method_summary)
    else: method_summary[COL_TOTAL_REDUNDANCY] = np.nan

    if t_params_count in method_summary.columns and t_total_methods in method_summary.columns:
        method_summary["Total_ParamRichness"] = safe_divide(t_params_count, t_total_methods, method_summary, default=0.0)
    else: method_summary["Total_ParamRichness"] = np.nan

    if t_params_w_type in method_summary.columns and t_params_count in method_summary.columns:
        method_summary["Total_ParamTypeCompleteness"] = safe_divide(t_params_w_type, t_params_count, method_summary)
    else: method_summary["Total_ParamTypeCompleteness"] = np.nan

    if t_methods_w_ret in method_summary.columns and t_total_methods in method_summary.columns:
        method_summary["Total_ReturnTypeCompleteness"] = safe_divide(t_methods_w_ret, t_total_methods, method_summary)
    else: method_summary["Total_ReturnTypeCompleteness"] = np.nan

    if t_methods_w_uc in method_summary.columns and t_total_methods in method_summary.columns:
        method_summary["Total_Percentage_Methods_With_UC"] = safe_divide(t_methods_w_uc, t_total_methods, method_summary, default=0.0)
    else: method_summary["Total_Percentage_Methods_With_UC"] = np.nan


    # Define final column order (include derived columns)
    t_count_uc_action = f"{TOTAL_PREFIX}{COL_COUNT_UC_ACTION}"
    t_count_uc_only = f"{TOTAL_PREFIX}{COL_COUNT_UC_ONLY}"
    t_count_action_only = f"{TOTAL_PREFIX}{COL_COUNT_ACTION_ONLY}"
    t_count_none = f"{TOTAL_PREFIX}{COL_COUNT_NONE}"
    t_extra_methods = f"{TOTAL_PREFIX}{COL_EXTRA_METHODS}"

    final_cols_order = [
        COL_MODEL,
        # Averages per file
        f"{AVG_PREFIX}{COL_REDUNDANCY}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_PARAM_RICHNESS}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_RETURN_COMPLETENESS}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_PARAM_TYPE_COMPLETENESS}{PER_FILE_SUFFIX}",
        f"{AVG_PREFIX}{COL_PERCENT_UC}{PER_FILE_SUFFIX}",
        # Overall derived ratios
        COL_TOTAL_REDUNDANCY,
        'Total_ParamRichness',
        'Total_ReturnTypeCompleteness',
        'Total_ParamTypeCompleteness',
        'Total_Percentage_Methods_With_UC',
        # Total counts
        t_total_methods,
        t_unique_names,
        t_extra_methods,
        t_methods_w_uc,
        t_count_uc_action,
        t_count_uc_only,
        t_count_action_only,
        t_count_none,
        f"{TOTAL_PREFIX}Total_UC_References_File",
        f"{TOTAL_PREFIX}Unique_UCs_File",
        f"{TOTAL_PREFIX}Unique_Actions_File",
        f"{TOTAL_PREFIX}DuplicatedOccurrences_Name",
        f"{TOTAL_PREFIX}DuplicatedOccurrences_NameParam",
        f"{TOTAL_PREFIX}DuplicatedOccurrences_FullSig",
        f"{TOTAL_PREFIX}Visibility_Public",
        f"{TOTAL_PREFIX}Visibility_Private",
        f"{TOTAL_PREFIX}Visibility_Protected",
        f"{TOTAL_PREFIX}Visibility_Package",
    ]
    # Filter to columns that actually exist in the dataframe
    final_cols = [col for col in final_cols_order if col in method_summary.columns]
    method_summary = method_summary[final_cols]

    return method_summary

def calculate_diversity(cache: MetricCache, counts_df: pd.DataFrame):
    """Calculates diversity metrics based on method distribution across target classes."""
    # Check input dataframe validity
    if (counts_df.empty or
        COL_MODEL not in counts_df.columns or
        not any(cls in counts_df.columns for cls in cache.class_names)):
        print("Warning: Counts DataFrame invalid for diversity calculation.")
        # Attempt to get models from cache if possible, otherwise return fully empty
        models_in_cache = []
        if hasattr(cache, 'file_info'):
             models_in_cache = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
        if models_in_cache:
            return pd.DataFrame([{COL_MODEL: m, "Entropy": np.nan, COL_NORM_ENTROPY: np.nan, "Gini": np.nan, "ExclusiveMethodsCount": 0} for m in models_in_cache])
        else:
            return pd.DataFrame()


    diversity_rows = []
    exclusive_methods_count = Counter()
    # Use getter for exclusive methods
    exclusive_methods_data = cache.get_exclusive_methods_list()
    for mdl, methods in exclusive_methods_data.items():
        exclusive_methods_count[mdl] = len(methods)

    # Get models present in the counts dataframe
    models_in_counts_df = sorted(counts_df[COL_MODEL].unique())

    # Filter counts_df columns to only include target classes
    class_cols = [c for c in cache.class_names if c in counts_df.columns]
    if not class_cols:
        print("Warning: No target class columns found in counts DataFrame for diversity calculation.")
        # Return structure with models found, but NaN values
        return pd.DataFrame([{COL_MODEL: m, "Entropy": np.nan, COL_NORM_ENTROPY: np.nan, "Gini": np.nan, "ExclusiveMethodsCount": exclusive_methods_count.get(m, 0)} for m in models_in_counts_df])


    # Group by model and sum counts for target classes
    # Ensure numeric conversion happens correctly BEFORE summing
    for col in class_cols:
        counts_df[col] = pd.to_numeric(counts_df[col], errors='coerce')

    model_sums = (
        counts_df.groupby(COL_MODEL)[class_cols]
        .sum(numeric_only=True) # sum should handle NaNs introduced by coerce by default (treat as 0)
        .reindex(models_in_counts_df, fill_value=0) # Ensure all models from counts_df are present
    )

    num_classes = len(cache.class_names) # Use the full target list for normalization denominator

    for mdl, counts in model_sums.iterrows():
        # Ensure counts are float for division, handle potential NaNs from grouping/reindexing if any slipped through
        m_counts = counts.astype(float).fillna(0).values
        total = np.nansum(m_counts) # nansum treats NaN as 0

        if total == 0:
            entropy, norm_entropy, gini = 0.0, 0.0, 0.0
        else:
            p = m_counts / total
            p = p[~np.isnan(p)] # Should not happen if NaNs handled above, but safe check
            if len(p) == 0:
                 entropy, norm_entropy, gini = 0.0, 0.0, 0.0
            else:
                # Entropy Calculation
                p_pos = p[p > 0]
                entropy = -np.sum(p_pos * np.log2(p_pos)) if len(p_pos) > 0 else 0.0
                # Normalized Entropy Calculation (using total number of target classes)
                norm_entropy = entropy / np.log2(num_classes) if num_classes > 1 else 0.0
                # Gini Calculation (requires mapping back to full class list)
                p_all = np.zeros(num_classes) # Array for all potential classes
                # Map current model's proportions into the full array
                p_map = dict(zip(model_sums.columns, p)) # Proportions for classes present in this model's sum
                for i, cls_name in enumerate(cache.class_names):
                     p_all[i] = p_map.get(cls_name, 0.0) # Fill with 0 if class had no methods for this model

                p_sort = np.sort(p_all)
                idx = np.arange(1, num_classes + 1)
                gini = ( (2 * np.sum(idx * p_sort)) / num_classes - (num_classes + 1) / num_classes ) if num_classes > 0 else 0.0

        diversity_rows.append(
            {
                COL_MODEL: mdl,
                "Entropy": entropy,
                COL_NORM_ENTROPY: norm_entropy,
                "Gini": gini,
            }
        )
    # Create initial dataframe from calculated rows
    diversity_df = pd.DataFrame(diversity_rows)

    # Add exclusive counts and ensure all models are present
    if not diversity_df.empty:
        # Create a DF of exclusive counts mapped to models
        exclusive_df = pd.DataFrame(exclusive_methods_count.items(), columns=[COL_MODEL, 'ExclusiveMethodsCount'])
        # Merge diversity metrics with exclusive counts
        diversity_df = pd.merge(diversity_df, exclusive_df, on=COL_MODEL, how='left')
        # Fill NaN for exclusive count (if a model had no exclusives) with 0
        diversity_df['ExclusiveMethodsCount'] = diversity_df['ExclusiveMethodsCount'].fillna(0).astype(int)
        # Ensure final ordering
        cols_order = [COL_MODEL, 'Entropy', COL_NORM_ENTROPY, 'Gini', 'ExclusiveMethodsCount']
        cols = [c for c in cols_order if c in diversity_df.columns] # Filter to existing
        diversity_df = diversity_df[cols]
    else:
        # If diversity calculation failed but we know models, return structure
        if models_in_counts_df:
             diversity_df = pd.DataFrame([{
                 COL_MODEL: m, "Entropy": np.nan, COL_NORM_ENTROPY: np.nan, "Gini": np.nan,
                 "ExclusiveMethodsCount": exclusive_methods_count.get(m, 0)
             } for m in models_in_counts_df])
        # else: diversity_df remains the empty DataFrame initialized at start

    return diversity_df

def calculate_coverage(cache: MetricCache, core_methods: list):
    """
    Calculates coverage metrics: how many unique core methods and total core method
    occurrences are found per model within the target classes.
    Also calculates a basic consensus score.

    Args:
        cache: The MetricCache object.
        core_methods: A list of core method names.

    Returns:
        A tuple containing:
            - coverage_df (pd.DataFrame): DataFrame with coverage per model.
            - consensus_strength (float): Basic consensus score.
    """
    core_set = set(core_methods)
    coverage = []
    model_methods = defaultdict(list) # Stores list of method names per model (target classes only)

    # Check prerequisites
    if not hasattr(cache, 'file_info') or not hasattr(cache, 'class_detailed_methods'):
         print("[calculate_coverage] Error: Cache missing required attributes. Returning empty.")
         return pd.DataFrame(), np.nan

    # --- Collect method names per model from target classes ---
    for fname in cache.get_all_files(): # Iterate through successfully loaded files
        mdl = cache.file_info.get(fname, {}).get('model')
        if not mdl: continue # Skip if model name not found

        # Get target class methods for this file
        cls_methods = cache.class_detailed_methods.get(fname, {})
        for method_list in cls_methods.values():
            model_methods[mdl].extend([m['name'] for m in method_list if 'name' in m])

    # --- Calculate coverage per model ---
    models_with_methods = sorted(list(model_methods.keys()))
    if not models_with_methods:
         print("[calculate_coverage] Warning: No target methods found for any model.")
         # Return structure based on all models known to the cache, if any
         all_models_in_cache = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
         if all_models_in_cache:
              empty_coverage_df = pd.DataFrame([
                  {COL_MODEL: m, COL_UNIQUE_CORE: 0, "Total_Core_Occurrences": 0}
                  for m in all_models_in_cache
              ])
              return empty_coverage_df, np.nan
         else:
              return pd.DataFrame(), np.nan # Truly empty


    for mdl in models_with_methods:
        m_list = model_methods.get(mdl, []) # Get collected target method names
        # Count unique core methods found in this model's list
        unique_core_count = len(set(m_list) & core_set)
        # Count total occurrences of core methods in this model's list
        total_core_occurrences = sum(m in core_set for m in m_list)
        coverage.append(
            {
                COL_MODEL: mdl,
                COL_UNIQUE_CORE: unique_core_count,
                "Total_Core_Occurrences": total_core_occurrences,
            }
        )

    coverage_df = pd.DataFrame(coverage)

    # --- Basic Consensus Strength Calculation (Example) ---
    # Counts how many core methods appear in at least half the models that generated any methods.
    # Note: This uses ALL methods parsed by the cache, not just target class methods,
    #       as it reflects the overall tendency of models to generate a core method name.
    consensus_strength = np.nan # Default
    global_method_counter = cache.get_global_method_counter()
    num_models_total = len(cache.file_info.keys()) # Could also use len(models_in_cache) if preferred

    if global_method_counter and core_set and num_models_total > 0:
        methods_in_models = defaultdict(set)
        all_details = cache.get_method_details_list()
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
            if len(methods_in_models.get(core_method, set())) >= consensus_threshold:
                consensus_method_count += 1

        # Normalize by the total number of core methods considered
        consensus_strength = consensus_method_count / len(core_set) if core_set else 0.0

    # Ensure columns exist and have the right order in the returned DataFrame
    if not coverage_df.empty:
        cols_order = [COL_MODEL, COL_UNIQUE_CORE, "Total_Core_Occurrences"]
        # Add missing columns if necessary (e.g., if coverage list was empty but models existed)
        for col in cols_order:
            if col not in coverage_df.columns:
                coverage_df[col] = 0 if col != COL_MODEL else None

        coverage_df = coverage_df[cols_order] # Apply final order

    # Ensure all models known to the cache are present, even if they had 0 coverage
    all_models_in_cache = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
    if not all_models_in_cache: # If no models found at all
        return pd.DataFrame(), np.nan

    final_coverage_df = pd.DataFrame({COL_MODEL: all_models_in_cache})
    if not coverage_df.empty:
        final_coverage_df = pd.merge(final_coverage_df, coverage_df, on=COL_MODEL, how='left')
        # Fill NaNs introduced by merge for models with 0 coverage
        final_coverage_df[COL_UNIQUE_CORE] = final_coverage_df[COL_UNIQUE_CORE].fillna(0).astype(int)
        final_coverage_df['Total_Core_Occurrences'] = final_coverage_df['Total_Core_Occurrences'].fillna(0).astype(int)
    else: # If coverage_df was empty (e.g., no target methods)
        final_coverage_df[COL_UNIQUE_CORE] = 0
        final_coverage_df['Total_Core_Occurrences'] = 0


    return final_coverage_df, consensus_strength

def calculate_annotation_metrics(
    cache: MetricCache, per_file_method_metrics_df: pd.DataFrame
):
    """
    Aggregates annotation-specific metrics (UC/Action counts) by model, based
    on per-file metrics derived from target class methods. Calculates overall
    unique UC/Action counts across all runs for each model based on target classes.

    Args:
        cache: The MetricCache object.
        per_file_method_metrics_df: DataFrame containing per-file metrics.

    Returns:
        A pandas DataFrame with aggregated annotation metrics per model.
    """
    # Check input validity
    if (per_file_method_metrics_df.empty or
        COL_MODEL not in per_file_method_metrics_df.columns):
        print("Warning: Annotation metrics skipped - input per-file metrics invalid.")
        # Define expected columns for an empty return structure
        cols = [COL_MODEL, f'Total_{COL_METHODS_WITH_UC}', f'Avg_{COL_PERCENT_UC}_per_file',
                'Total_Percentage_Methods_With_UC', 'Total_UC_References', 'Total_Unique_UCs',
                'Avg_Unique_UCs_PerFile', 'Total_Unique_Actions', 'Avg_Unique_Actions_PerFile']
        return pd.DataFrame(columns=cols)

    # Define aggregation operations for existing per-file columns
    agg_cols = {
        # Sum of methods with any UC annotation per file
        f'Total_{COL_METHODS_WITH_UC}': (COL_METHODS_WITH_UC, 'sum'),
        # Average percentage of methods with UC per file
        f'Avg_{COL_PERCENT_UC}_per_file': (COL_PERCENT_UC, 'mean'),
        # Sum of total UC references per file
        'Total_UC_References': ('Total_UC_References_File', 'sum'),
         # Average number of unique UCs per file
        'Avg_Unique_UCs_PerFile': ('Unique_UCs_File', 'mean'),
         # Average number of unique Actions per file
        'Avg_Unique_Actions_PerFile': ('Unique_Actions_File', 'mean')
    }

    # Filter aggregations to only include columns present in the input DataFrame
    valid_agg_cols = {
        k: v for k, v in agg_cols.items()
        if v[0] in per_file_method_metrics_df.columns
    }

    if not valid_agg_cols:
        print("Warning: Annotation metrics skipped - required columns missing from per-file metrics.")
        # Define expected columns for an empty return structure
        cols = [COL_MODEL, f'Total_{COL_METHODS_WITH_UC}', f'Avg_{COL_PERCENT_UC}_per_file',
                'Total_Percentage_Methods_With_UC', 'Total_UC_References', 'Total_Unique_UCs',
                'Avg_Unique_UCs_PerFile', 'Total_Unique_Actions', 'Avg_Unique_Actions_PerFile']
        # Get unique models if possible to return structure
        models_in_df = per_file_method_metrics_df[COL_MODEL].unique() if COL_MODEL in per_file_method_metrics_df else []
        if models_in_df.size > 0:
             return pd.DataFrame({COL_MODEL: models_in_df, **{c: np.nan for c in cols if c != COL_MODEL}})
        else:
             return pd.DataFrame(columns=cols) # Truly empty

    # Perform initial aggregation based on per-file metrics
    try:
        annotation_agg = (
            per_file_method_metrics_df.groupby(COL_MODEL)
            .agg(**valid_agg_cols)
            .reset_index()
        )
    except Exception as e:
         print(f"Error during initial annotation aggregation: {e}")
         return pd.DataFrame() # Return empty on aggregation error


    # --- Calculate TOTAL Unique UCs and Actions per Model (Target Classes Only) ---
    # This requires iterating through the detailed cache data again
    model_unique_data = []
    if hasattr(cache, 'class_detailed_methods') and hasattr(cache, 'file_info'):
        class_methods_map = cache.get_class_detailed_methods() # Use getter
        model_files = defaultdict(list)
        # Group files by model using cache.file_info
        [model_files[info['model']].append(fname) for fname, info in cache.file_info.items() if info.get('model')]

        for model, files in model_files.items():
            model_total_unique_ucs = set()
            model_total_unique_actions = set()
            for fname in files:
                 # Iterate only through target class methods for this file
                 for cls_name, method_list in class_methods_map.get(fname, {}).items():
                      for method in method_list:
                        # 'ucs' is guaranteed to be a list of strings by MetricCache._parse
                        ucs = method.get('ucs', [])
                        action = method.get('action', '') # Guaranteed string
                        # Update sets with UCs and non-empty actions
                        model_total_unique_ucs.update(ucs)
                        if action: model_total_unique_actions.add(action)

            model_unique_data.append({
                COL_MODEL: model,
                'Total_Unique_UCs': len(model_total_unique_ucs),
                'Total_Unique_Actions': len(model_total_unique_actions)
            })
    else:
         print("Warning: Cannot calculate total unique UCs/Actions - cache missing attributes.")


    # Merge total unique counts into the aggregated dataframe
    if model_unique_data:
        unique_counts_df = pd.DataFrame(model_unique_data)
        annotation_agg = pd.merge(annotation_agg, unique_counts_df, on=COL_MODEL, how='left')
        # Fill NaN for models potentially missing from unique_counts_df (if cache was inconsistent)
        annotation_agg['Total_Unique_UCs'] = annotation_agg['Total_Unique_UCs'].fillna(0).astype(int)
        annotation_agg['Total_Unique_Actions'] = annotation_agg['Total_Unique_Actions'].fillna(0).astype(int)
    else:
        # Add columns with NaN if calculation failed
        if 'Total_Unique_UCs' not in annotation_agg.columns:
             annotation_agg['Total_Unique_UCs'] = np.nan
        if 'Total_Unique_Actions' not in annotation_agg.columns:
             annotation_agg['Total_Unique_Actions'] = np.nan


    # --- Calculate Overall Percentage of Methods with UC ---
    t_methods = COL_TOTAL_METHODS
    t_agg_methods = f'Total_{t_methods}' # This needs to be calculated first
    t_agg_uc = f'Total_{COL_METHODS_WITH_UC}' # This comes from agg_cols

    # Calculate total methods per model from the per-file data
    if t_methods in per_file_method_metrics_df.columns:
        total_methods_per_model = (
            per_file_method_metrics_df.groupby(COL_MODEL)[t_methods]
            .sum(numeric_only=True)
            .reset_index(name=t_agg_methods) # Name the summed column correctly
        )
        # Merge this total count into the aggregation result
        annotation_agg = pd.merge(annotation_agg, total_methods_per_model, on=COL_MODEL, how='left')

        # Now calculate the overall percentage
        if t_agg_uc in annotation_agg.columns and t_agg_methods in annotation_agg.columns:
            # Fill NaNs that might result from merges or calculations before division
            annotation_agg[t_agg_uc] = annotation_agg[t_agg_uc].fillna(0)
            annotation_agg[t_agg_methods] = annotation_agg[t_agg_methods].fillna(0)
            # Use safe_divide or manual check for zero denominator
            annotation_agg['Total_Percentage_Methods_With_UC'] = np.where(
                annotation_agg[t_agg_methods] > 0,
                (annotation_agg[t_agg_uc] / annotation_agg[t_agg_methods]),
                0.0 # Assign 0 if total methods is 0
            )
        else:
            annotation_agg['Total_Percentage_Methods_With_UC'] = np.nan
        # Can drop t_agg_methods if not needed later
        # annotation_agg.drop(columns=[t_agg_methods], errors='ignore', inplace=True)
    else:
        # Cannot calculate total percentage if per-file total methods is missing
        annotation_agg['Total_Percentage_Methods_With_UC'] = np.nan


    # Define final column order and filter to existing columns
    final_order = [
        COL_MODEL,
        t_agg_uc, # Total methods with any UC (sum over files)
        f'Avg_{COL_PERCENT_UC}_per_file', # Avg % of methods with UC per file
        'Total_Percentage_Methods_With_UC', # Overall % of methods with UC
        'Total_UC_References', # Sum of all UC references across all methods/files
        'Total_Unique_UCs', # Overall unique UCs for the model
        'Avg_Unique_UCs_PerFile', # Avg unique UCs found per file
        'Total_Unique_Actions', # Overall unique Actions for the model
        'Avg_Unique_Actions_PerFile' # Avg unique Actions found per file
    ]
    final_cols = [c for c in final_order if c in annotation_agg.columns]
    annotation_agg = annotation_agg[final_cols]

    return annotation_agg
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
    print("Identifying added classes compared to baseline...")

    # Check if baseline structure was parsed
    if not hasattr(cache, 'baseline_structure') or not cache.baseline_structure:
        print("Warning: Baseline structure not available. Cannot calculate added classes.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])

    baseline_classes = cache.baseline_structure.get('classes', set())
    print(f"  Baseline contains {len(baseline_classes)} classes.")

    added_class_tracker = defaultdict(set) # Key: added_class_name, Value: set of models adding it
    all_models = set()

    # Check if generated data is available
    if not hasattr(cache, 'json_data') or not cache.json_data or not hasattr(cache, 'files') or not cache.files:
        print("Warning: No generated JSON data found in cache. Cannot calculate added classes.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])

    files_processed = 0
    for json_filename in cache.files: # Iterate through successfully loaded files
        model = cache.file_info.get(json_filename, {}).get('model')
        stem = cache.file_info.get(json_filename, {}).get('stem')

        # Ensure we have model info and the corresponding data exists
        if not model or not stem or stem not in cache.json_data:
            # print(f"  Skipping file {json_filename}: Missing model info or data.") # Optional verbose skip
            continue

        all_models.add(model) # Track all models encountered
        file_json = cache.json_data[stem]

        # Extract classes generated in this specific file
        try:
            gen_classes_in_file = set(
                c.get('name')
                for c in file_json.get('classes', [])
                if c.get('name') # Ensure name exists and is not empty
            )
        except Exception as e:
             print(f"  [ERROR] Failed to extract classes from {json_filename}: {e}")
             continue # Skip file if class extraction fails

        # Find classes added in this file compared to baseline
        added_in_file = gen_classes_in_file - baseline_classes

        # Add the current model to the tracker set for each added class
        for added_class in added_in_file:
            added_class_tracker[added_class].add(model)
        files_processed += 1

    print(f"  Processed {files_processed} generated files for added classes.")

    # Format the results into a list of dictionaries for the DataFrame
    output_data = []
    for added_class, models_set in added_class_tracker.items():
        output_data.append({
            'Added_Class': added_class,
            'LLM_Count': len(models_set),
            'LLM_List': ", ".join(sorted(list(models_set))) # Comma-separated sorted list
        })

    if not output_data:
        print("Info: No added classes (classes not in baseline) found in any generated model.")
        return pd.DataFrame(columns=['Added_Class', 'LLM_Count', 'LLM_List'])

    # Create DataFrame and sort
    results_df = pd.DataFrame(output_data).sort_values(
        by=['LLM_Count', 'Added_Class'],
        ascending=[False, True] # Sort by count descending, then name ascending
    )
    print(f"  Found {len(results_df)} classes added by at least one model.")
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
    print("Calculating overlap metrics (Bootstrap and Global Jaccard)...")

    # --- Get list of unique models ---
    # Prioritize models present in the counts dataframe if available and valid
    models = []
    if (not counts_df_files_rows.empty and
        COL_MODEL in counts_df_files_rows.columns and
        COL_TOTAL_METHODS in counts_df_files_rows.columns and
        counts_df_files_rows[COL_TOTAL_METHODS].notnull().any()):
        models = sorted(counts_df_files_rows[COL_MODEL].unique())
        print(f"  Using {len(models)} models found in counts data for Bootstrap.")
    else:
        # Fallback to models known by the cache if counts data is insufficient
        if hasattr(cache, 'file_info'):
            models = sorted(list(set(i.get("model") for i in cache.file_info.values() if i.get("model"))))
            print(f"  Using {len(models)} models found in cache info (counts data insufficient for Bootstrap).")
        else:
            models = []

    if not models:
        print("  [ERROR] No models found to calculate overlap for.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 1. Bootstrap Overlap Calculation ---
    bootstrap_overlap_df = pd.DataFrame(np.nan, index=models, columns=models)
    bootstrap_skipped = False

    if (counts_df_files_rows.empty or
        COL_MODEL not in counts_df_files_rows.columns or
        COL_TOTAL_METHODS not in counts_df_files_rows.columns or
        counts_df_files_rows[COL_TOTAL_METHODS].isnull().all()):
        print("  Skipping Bootstrap Overlap: Input counts DataFrame is invalid or empty.")
        bootstrap_skipped = True
    else:
        # Prepare data: select relevant columns and convert to numeric
        totals = counts_df_files_rows[[COL_MODEL, COL_TOTAL_METHODS]].copy()
        totals[COL_TOTAL_METHODS] = pd.to_numeric(totals[COL_TOTAL_METHODS], errors='coerce')
        totals.dropna(subset=[COL_TOTAL_METHODS], inplace=True) # Drop rows where count couldn't be numeric

        if totals.empty:
            print("  Skipping Bootstrap Overlap: No valid numeric method counts found.")
            bootstrap_skipped = True
        else:
            print("  Performing Bootstrap resampling...")
            bootstrap_cache = {} # Cache resampled means per model
            n_bootstrap = 1000 # Number of bootstrap samples

            for mdl in models:
                # Get the vector of total method counts for this model's runs
                v = totals.loc[totals[COL_MODEL] == mdl, COL_TOTAL_METHODS].values
                n_clean = len(v) # Already dropped NaNs above

                if n_clean > 0:
                    # Resample with replacement and calculate mean for each sample
                    resampled_indices = np.random.randint(0, n_clean, size=(n_bootstrap, n_clean))
                    resampled_means = v[resampled_indices].mean(axis=1)
                    bootstrap_cache[mdl] = resampled_means
                else:
                    bootstrap_cache[mdl] = np.array([]) # Empty array if model had no valid runs/counts
                    # print(f"    Model '{mdl}' has no valid runs/counts for bootstrap.") # Optional warning

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
                    bootstrap_overlap_df.loc[model_b, model_a] = 1.0 - prob_a_gt_b
                else:
                    # If one model had no data, can't compare
                    bootstrap_overlap_df.loc[model_a, model_b] = np.nan
                    bootstrap_overlap_df.loc[model_b, model_a] = np.nan
            print("  ✔ Bootstrap calculation finished.")


    # --- 2. Global Jaccard Similarity Calculation ---
    print("  Calculating Global Jaccard Similarity (based on all method names)...")
    model_method_sets = {}
    jaccard_global_df = pd.DataFrame(np.nan, index=models, columns=models)
    jaccard_skipped = False

    # Check cache prerequisites
    if not hasattr(cache, 'get_method_details_list') or not hasattr(cache, 'file_info'):
         print("  Skipping Global Jaccard: Cache missing required attributes.")
         jaccard_skipped = True
    else:
        all_details = cache.get_method_details_list()
        info = cache.file_info
        if not all_details:
             print("  Skipping Global Jaccard: No global method details found in cache.")
             jaccard_skipped = True
        else:
            # Create a set of all method names for each model
            temp_sets = defaultdict(set)
            for m_detail in all_details:
                m_name = m_detail.get('name')
                m_file = m_detail.get(COL_FILE)
                if m_name and m_file:
                    model = info.get(m_file, {}).get('model')
                    if model:
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

                jaccard_index = float(intersection / union) if union > 0 else 0.0
                jaccard_global_df.loc[model_a, model_b] = jaccard_index
                jaccard_global_df.loc[model_b, model_a] = jaccard_index # Symmetric

            # Fill diagonal with 1.0 (optional, some prefer NaN or 0)
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
        Returns an empty dictionary if calculation cannot proceed.
    """
    print("Calculating Per-Class Jaccard overlap...")

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

    print(f"  Processing {len(target_classes)} target classes for {len(models)} models...")
    classes_processed_count = 0

    for class_name in target_classes:
        classes_processed_count += 1
        # --- Create set of method names per model FOR THIS CLASS ---
        model_class_method_sets = defaultdict(set)
        methods_found_in_class = False

        for model_name in models:
            # Find all files associated with this model
            model_files = [f for f, i in info.items() if i.get('model') == model_name]
            if not model_files: continue # Skip if model has no files (shouldn't happen ideally)

            # Collect method names from this model's files for the current class_name
            for fname in model_files:
                # Get methods for the specific file and class
                methods_in_file_class = class_methods_map.get(fname, {}).get(class_name, [])
                if methods_in_file_class:
                    methods_found_in_class = True
                    for method in methods_in_file_class:
                        m_name = method.get('name')
                        if m_name:
                            model_class_method_sets[model_name].add(m_name)
            # If a model has no methods for this class, its set will remain empty in the defaultdict

        # Skip calculating matrix if no methods were found for this class in *any* model
        if not methods_found_in_class:
            # print(f"    Skipping class '{class_name}': No methods found in any model.") # Optional info
            continue

        # --- Calculate Jaccard matrix for this class ---
        jaccard_matrix = pd.DataFrame(np.nan, index=models, columns=models)

        for model_a, model_b in it.combinations(models, 2):
            set_a = model_class_method_sets[model_a] # Relies on defaultdict returning empty set if key missing
            set_b = model_class_method_sets[model_b]

            intersection = len(set_a & set_b)
            union = len(set_a | set_b)

            jaccard_index = float(intersection / union) if union > 0 else 0.0
            jaccard_matrix.loc[model_a, model_b] = jaccard_index
            jaccard_matrix.loc[model_b, model_a] = jaccard_index # Symmetric

        # Fill diagonal (optional)
        np.fill_diagonal(jaccard_matrix.values, 1.0)

        # Check if matrix contains any non-NaN values before storing
        # (It should unless only one model exists)
        if not jaccard_matrix.isnull().all().all():
            per_class_jaccard_results[class_name] = jaccard_matrix
        # else: print(f"    Skipping saving matrix for '{class_name}': Matrix is all NaN (e.g., only one model).")


    print(f"  ✔ Per-Class Jaccard calculation finished. Processed {classes_processed_count} classes, generated {len(per_class_jaccard_results)} matrices.")
    return per_class_jaccard_results

def calculate_variability(cache: MetricCache, counts_df: pd.DataFrame):
    """Calculates variability metrics for Target_Class_Methods."""
    if (
        counts_df.empty
        or COL_TOTAL_METHODS not in counts_df.columns
        or counts_df[COL_TOTAL_METHODS].isnull().all()
    ):
        print("Warning: Cannot calculate variability.")
        models = sorted(list(set(i.get("model", "?") for i in cache.file_info.values())))
        return pd.DataFrame(
            { COL_MODEL: models, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
              "CI_low": np.nan, "CI_high": np.nan, "NumRuns": 0 }
        )
    totals_info = counts_df[[COL_MODEL, COL_RUN, COL_TOTAL_METHODS]].copy()
    totals_info[COL_TOTAL_METHODS] = pd.to_numeric(totals_info[COL_TOTAL_METHODS], errors="coerce")
    totals_info[COL_RUN] = pd.to_numeric(totals_info[COL_RUN], errors="coerce")
    totals_info.dropna(subset=[COL_RUN], inplace=True)
    totals_info[COL_RUN] = totals_info[COL_RUN].astype(int)
    variability = []
    models = sorted(totals_info[COL_MODEL].unique())
    for mdl in models:
        m_data = totals_info[totals_info[COL_MODEL] == mdl].sort_values(COL_RUN)
        v = m_data[COL_TOTAL_METHODS].values; n = len(v)
        if n == 0:
            variability.append({ COL_MODEL: mdl, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
                                 "CI_low": np.nan, "CI_high": np.nan, "NumRuns": n }); continue
        v_clean = v[~np.isnan(v)]; n_clean = len(v_clean)
        if n_clean == 0:
            variability.append({ COL_MODEL: mdl, "Mean": np.nan, COL_CV: np.nan, "ConvergenceSlope": np.nan,
                                 "CI_low": np.nan, "CI_high": np.nan, "NumRuns": n }); continue
        mu=np.mean(v_clean); sig_pop=np.std(v_clean, ddof=0); sig_sample=np.std(v_clean, ddof=1) if n_clean > 1 else np.nan; cv=(sig_pop / mu) if mu != 0 else np.nan; slope = np.nan
        if n > 1:
            runs = m_data[COL_RUN].values; s_idx = np.argsort(runs); v_s = v[s_idx]; r_s = runs[s_idx]
            c_sum = np.nancumsum(v_s); counts_avg = np.arange(1, n + 1, dtype=float)
            try: fvi = np.where(~np.isnan(v_s))[0][0]; counts_avg[:fvi] = np.nan
            except IndexError: counts_avg[:] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'): c_avg = c_sum / counts_avg
            mask = ~np.isnan(c_avg)
            if np.sum(mask) > 1:
                 try: slope = np.polyfit(r_s[mask], c_avg[mask], 1)[0]
                 except Exception as e: print(f"Warn: Slope failed {mdl}: {e}"); slope = np.nan
        ci_h = t.ppf(0.975, df=n_clean-1)*sig_sample/math.sqrt(n_clean) if n_clean>1 and not np.isnan(sig_sample) else np.nan; ci_l = mu - ci_h if not np.isnan(mu + ci_h) else np.nan; ci_h_val = mu + ci_h if not np.isnan(mu + ci_h) else np.nan
        variability.append({ COL_MODEL: mdl, "Mean": mu, COL_CV: cv, "ConvergenceSlope": slope,
                             "CI_low": ci_l, "CI_high": ci_h_val, "NumRuns": n })
    df = pd.DataFrame(variability);
    if not df.empty: df = df[[COL_MODEL] + [col for col in df.columns if col != COL_MODEL]]
    return df

# --- NEW Semantic Mapping Function (with Name Splitting) ---

def map_methods_to_actions(cache: MetricCache, gold_map: dict, nlp_model_name: str):
    """
    Maps each generated method (in target classes) to the best-matching gold standard
    action based on maximum cosine similarity, using context (UC annotation or Class).

    Args:
        cache: The MetricCache object containing parsed data.
        gold_map: The dictionary loaded from uc_action_method_map.json.
        nlp_model_name: The name of the sentence-transformer model to use.

    Returns:
        A pandas DataFrame mapping generated methods to their best action match,
        including the similarity score and match type, or an empty DataFrame on error.
        Columns: Model, Run, file, Class, MethodName, SplitMethodName,
                 Best_Match_Action, Max_Similarity_Score, Match_Type ('UC' or 'Class')
    """
    print(f"🚀 Starting method-to-action mapping using model '{nlp_model_name}' (Context-Aware)...")

    # --- 1. Load Sentence Transformer Model ---
    print("  Initializing Sentence Transformer model...")
    try:
        model = SentenceTransformer(nlp_model_name)
        print(f"  ✔ Model '{nlp_model_name}' loaded. Device: {model.device}")
    except Exception as e:
        print(f"  [ERROR] Failed to load Sentence Transformer model '{nlp_model_name}': {e}")
        return pd.DataFrame()

    # --- 2. Pre-process Gold Map ---
    # Store actions by UC ID AND build a reverse map from assigned_class to relevant UC IDs
    gold_actions_by_uc = defaultdict(list)
    class_to_uc_ids = defaultdict(set)
    all_gold_actions = set()
    if not isinstance(gold_map, dict):
        print("  [ERROR] Gold map is not a dictionary.")
        return pd.DataFrame()

    for uc_id, entries in gold_map.items():
        uc_id_str = str(uc_id) # Ensure string key
        if isinstance(entries, list):
            for entry in entries:
                action = entry.get("action")
                assigned_class = entry.get("assigned_class")
                if action and isinstance(action, str):
                    gold_actions_by_uc[uc_id_str].append(action)
                    all_gold_actions.add(action)
                    # Map class back to UC ID if class is provided
                    if assigned_class and isinstance(assigned_class, str):
                        class_to_uc_ids[assigned_class].add(uc_id_str)

    if not all_gold_actions:
         print("  [ERROR] No valid 'action' strings found in the provided gold_map.")
         return pd.DataFrame()
    list_all_gold_actions = sorted(list(all_gold_actions))
    print(f"  Found {len(list_all_gold_actions)} unique gold actions across {len(gold_actions_by_uc)} UCs.")
    print(f"  Mapped {len(class_to_uc_ids)} classes to relevant UC IDs.")

    # --- 3. Extract All Generated Method Details AND Split Names (Target Classes Only) ---
    generated_methods_details = []
    unique_split_names = set()
    print("  Extracting generated method details and splitting names...")
    if not hasattr(cache, 'file_info') or not hasattr(cache, 'class_detailed_methods'):
         print("  [ERROR] Cache object missing required attributes.")
         return pd.DataFrame()

    for fname, info in cache.file_info.items():
        model_name = info.get('model')
        run = info.get('run')
        if not model_name or not run: continue

        class_methods = cache.class_detailed_methods.get(fname, {})
        for class_name, method_list in class_methods.items():
            for method in method_list:
                gen_name = method.get("name")
                if gen_name:
                    split_name = split_method_name(gen_name)
                    details = {
                        COL_MODEL: model_name, COL_RUN: run, COL_FILE: fname,
                        'Class': class_name, COL_METHOD_NAME: gen_name,
                        'SplitMethodName': split_name,
                        'UCs': method.get('ucs', []) # Get associated UCs (guaranteed list by cache)
                    }
                    generated_methods_details.append(details)
                    if split_name: unique_split_names.add(split_name)

    if not generated_methods_details:
         print("  [WARN] No generated methods found in target classes.")
         return pd.DataFrame()
    print(f"  Found {len(generated_methods_details)} generated method instances ({len(unique_split_names)} unique split names) in target classes.")


    # --- 4. Embed All Unique Strings (SPLIT Names + Actions) ---
    strings_to_embed = list(unique_split_names | set(list_all_gold_actions))
    if not strings_to_embed:
        print("  [ERROR] No strings found to embed.")
        return pd.DataFrame()
    print(f"  Embedding {len(strings_to_embed)} unique strings (split names + actions)...")
    try:
        embeddings = model.encode(strings_to_embed, convert_to_tensor=True, show_progress_bar=True)
        embedding_map = {text: emb for text, emb in zip(strings_to_embed, embeddings)}
        print("  ✔ Embeddings created.")
    except Exception as e:
        print(f"  [ERROR] Failed during sentence embedding: {e}"); traceback.print_exc(); return pd.DataFrame()

    # --- 5. Prepare lookup for Gold Action Embeddings ---
    # Instead of one big tensor, keep embeddings mapped by action text for easier lookup
    gold_action_embeddings_map = {}
    missing_embed_actions = []
    for action in list_all_gold_actions:
        emb = embedding_map.get(action)
        if emb is not None:
            gold_action_embeddings_map[action] = emb
        else:
            missing_embed_actions.append(action)
    if missing_embed_actions:
        print(f"  [WARN] Embeddings missing for {len(missing_embed_actions)} gold actions: {missing_embed_actions[:5]}...")
    if not gold_action_embeddings_map:
         print("  [ERROR] No embeddings available for any gold actions.")
         return pd.DataFrame()


    # --- 6. Find Best Match for Each Generated Method (Context-Aware) ---
    results = []
    print("  Mapping generated methods to best matching actions (Context-Aware)...")
    calculation_errors = 0
    methods_processed = 0
    uc_matches = 0
    class_matches = 0
    no_context_matches = 0

    for method_detail in generated_methods_details:
        methods_processed += 1
        split_name = method_detail['SplitMethodName']
        gen_class = method_detail['Class']
        annotated_ucs = method_detail.get('UCs', []) # List of UC IDs (strings)

        gen_emb = embedding_map.get(split_name)
        if gen_emb is None:
            results.append({**method_detail, 'Best_Match_Action': None, 'Max_Similarity_Score': np.nan, 'Match_Type': 'No_Embedding'})
            continue

        gen_emb = gen_emb.to(model.device) # Move embedding to model device

        candidate_actions = set()
        match_type = None

        # Strategy 1: Use UC annotation if available
        if annotated_ucs:
            for uc_num_str in annotated_ucs:
                 uc_id_lookup = f"UC{uc_num_str}"
                 candidate_actions.update(gold_actions_by_uc.get(uc_id_lookup, []))
            if candidate_actions:
                 match_type = 'UC'
                 uc_matches += 1

        # Strategy 2: Use Class alignment if no match from UC annotation
        if not candidate_actions:
            relevant_uc_ids = class_to_uc_ids.get(gen_class, set())
            if relevant_uc_ids:
                for uc_id in relevant_uc_ids:
                     candidate_actions.update(gold_actions_by_uc.get(uc_id, []))
            if candidate_actions:
                 match_type = 'Class'
                 class_matches += 1

        # Find best match among candidate actions
        best_match_action_text = None
        max_similarity_score = -1.0

        if candidate_actions:
            candidate_embeddings = []
            candidate_texts = []
            for action in candidate_actions:
                action_emb = gold_action_embeddings_map.get(action)
                if action_emb is not None:
                    candidate_embeddings.append(action_emb)
                    candidate_texts.append(action) # Keep track of the text

            if candidate_embeddings:
                try:
                    candidate_tensor = torch.stack(candidate_embeddings).to(model.device)
                    all_sims = cos_sim(gen_emb, candidate_tensor).squeeze()

                    if all_sims.numel() > 0:
                        if all_sims.dim() == 0: # Single candidate
                            max_similarity_score = all_sims.item()
                            best_idx = 0
                        else:
                            best_score_tensor, best_idx_tensor = torch.max(all_sims, dim=0)
                            max_similarity_score = best_score_tensor.item()
                            best_idx = best_idx_tensor.item()

                        best_match_action_text = candidate_texts[best_idx] # Use index on candidate_texts list

                except Exception as e:
                    print(f"  [ERROR] Cosine sim failed for method '{method_detail[COL_METHOD_NAME]}' (split: '{split_name}'): {e}")
                    calculation_errors += 1
                    max_similarity_score = np.nan # Indicate error
                    match_type = 'Error' # Mark as error
            # else: No embeddings found for candidate actions
        else:
             no_context_matches += 1
             match_type = 'No_Context' # No candidates found via UC or Class

        results.append({
             **method_detail,
             'Best_Match_Action': best_match_action_text,
             'Max_Similarity_Score': max_similarity_score if max_similarity_score > -1.0 else np.nan,
             'Match_Type': match_type
        })

    print(f"  ✔ Mapping finished. Processed={methods_processed}, UC Matches={uc_matches}, Class Matches={class_matches}, No Context={no_context_matches}, Errors={calculation_errors}")

    # --- 7. Create and Return DataFrame ---
    if not results:
        print("  [WARN] No results generated from mapping.")
        return pd.DataFrame()
    output_df = pd.DataFrame(results)
    # Add/Reorder columns
    cols_order = [
        COL_MODEL, COL_RUN, COL_FILE, 'Class', COL_METHOD_NAME, 'SplitMethodName',
        'Best_Match_Action', 'Max_Similarity_Score', 'Match_Type' # Added Match_Type
    ]
    output_df = output_df[[col for col in cols_order if col in output_df.columns]]
    return output_df


# --- Reporting Functions ---

def generate_core_methods_report(cache: MetricCache, top_n: int):
    counter = cache.get_global_method_counter()
    if not counter:
        print("[generate_core_methods_report] Warning: Global method counter is empty.")
        return pd.DataFrame(columns=[COL_METHOD_NAME, COL_GLOBAL_FREQ])
    df=pd.DataFrame(counter.most_common(top_n), columns=[COL_METHOD_NAME, COL_GLOBAL_FREQ])
    return df

def generate_exclusive_methods_report(cache: MetricCache):
    rows=[]
    exclusive_list_data = cache.get_exclusive_methods_list()
    if not exclusive_list_data:
        print("[generate_exclusive_methods_report] Warning: Exclusive methods list is empty.")
        return pd.DataFrame()
    models=sorted(exclusive_list_data.keys())
    details=cache.get_method_details_list()
    info=getattr(cache, 'file_info', {})
    for mdl in models:
         methods=sorted(exclusive_list_data.get(mdl, []))
         for name in methods:
              d=next((m for m in details if m.get('name')==name and info.get(m.get(COL_FILE),{}).get('model')==mdl), None)
              rows.append({COL_MODEL:mdl, COL_METHOD_NAME:name,
                           'ExampleFile':d.get(COL_FILE) if d else None,
                           'ExampleClass':d.get('class') if d else None,
                           'ExampleSignature':d.get('full_sig') if d else None,
                           'HasUCAnnotation':d.get('has_uc_annotation', False) if d else False,})
    return pd.DataFrame(rows)

def generate_uc_method_report(cache: MetricCache):
    rows=[]; details=cache.get_method_details_list(); info=getattr(cache, 'file_info', {})
    if not details: return pd.DataFrame() # Handle empty
    for m in details:
        if m.get('has_uc_annotation'):
            model=info.get(m.get(COL_FILE),{}).get('model'); ucs=m.get('ucs',[]) # ucs is list from _parse
            rows.append({COL_FILE:m.get(COL_FILE), COL_MODEL:model, 'Class':m.get('class'),
                         COL_METHOD_NAME:m.get('name'), 'Signature':m.get('full_sig'),
                         'UCs':','.join(sorted(ucs)) if ucs else None, 'Action':m.get('action'),
                         'Modifier':m.get('visibility'), 'ReturnType':m.get('return_type')})
    df=pd.DataFrame(rows)
    if not df.empty:
        s_cols=[COL_MODEL, COL_FILE, 'Class', COL_METHOD_NAME]
        ex_cols=[c for c in s_cols if c in df.columns]
        if ex_cols: # Check if columns exist before sorting
            df.sort_values(by=ex_cols, inplace=True, na_position='last')
    return df

def generate_uc_frequency_report(cache: MetricCache):
    counter = cache.get_global_uc_counter()
    if not counter: return pd.DataFrame(columns=['UC_ID', COL_GLOBAL_FREQ]) # Handle empty
    df=pd.DataFrame(counter.most_common(), columns=['UC_ID', COL_GLOBAL_FREQ])
    if not df.empty: df.sort_values(by=[COL_GLOBAL_FREQ, 'UC_ID'], ascending=[False, True], inplace=True)
    return df

def generate_action_frequency_report(cache: MetricCache):
    counter = cache.get_global_action_counter()
    if not counter: return pd.DataFrame(columns=['Action', COL_GLOBAL_FREQ]) # Handle empty
    df=pd.DataFrame(counter.most_common(), columns=['Action', COL_GLOBAL_FREQ])
    if not df.empty: df.sort_values(by=[COL_GLOBAL_FREQ, 'Action'], ascending=[False, True], inplace=True)
    return df

def generate_method_annotation_report(cache: MetricCache, target_classes_only=True):
    """Generates a report listing methods and their associated UC annotation details."""
    rows=[]; methods_to_process=[]
    info=getattr(cache, 'file_info', {}) # Use getattr

    if target_classes_only:
        class_methods_map = getattr(cache, 'class_detailed_methods', {})
        if isinstance(class_methods_map, dict):
            [methods_to_process.extend(m_list) for cls_dict in class_methods_map.values() if isinstance(cls_dict, dict) for m_list in cls_dict.values() if isinstance(m_list, list)]
        else:
             print("[generate_method_annotation_report] Warning: class_detailed_methods not found or not a dict.")
    else:
        _global_list = cache.get_method_details_list()
        methods_to_process = _global_list if isinstance(_global_list, list) else []

    if not methods_to_process:
         print("[generate_method_annotation_report] Warning: No methods found to process.")
         return pd.DataFrame()

    processed=0; skipped=0
    for m in methods_to_process:
        if not isinstance(m, dict): skipped+=1; continue
        f=m.get(COL_FILE); c=m.get('class'); n=m.get('name'); s=m.get('full_sig')
        if not all([f,c,n]): skipped+=1; continue # Ensure essential keys are present
        i=info.get(f,{}); mdl=i.get('model'); run=i.get('run')
        has_a=m.get('has_uc_annotation', False); ucs=m.get('ucs',[]); action=m.get('action','') # ucs guaranteed list
        rows.append({COL_FILE:f, COL_MODEL:mdl, COL_RUN:run, 'Class':c, COL_METHOD_NAME:n, 'Signature':s,
                     'Has_UC_Annotation':has_a, 'UC_References':", ".join(sorted(ucs)) if ucs else None,
                     'UC_Action':action if action else None}); processed+=1

    if skipped > 0: print(f"[generate_method_annotation_report] Skipped {skipped} invalid method entries.")
    df=pd.DataFrame(rows)
    if not df.empty:
        s_cols=[COL_MODEL, COL_RUN, COL_FILE, 'Class', COL_METHOD_NAME]
        ex_cols=[c for c in s_cols if c in df.columns]
        if ex_cols:
             try: df.sort_values(by=ex_cols, inplace=True, na_position='last')
             except Exception as e: print(f"Warn: Sort failed in method annotation report: {e}")
    return df


# --- Main Execution ---
def main():
    print("🚀 Starting Metrics Pipeline...")
    print("🔍 Setting up directories...")
    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output reports: '{REPORTS_OUTPUT_DIR.resolve()}'")

    if not JSON_INPUT_DIR.is_dir():
        sys.exit(f"❌ Error: Input JSON directory not found: '{JSON_INPUT_DIR.resolve()}'.")

    # --- Load Gold Standard Map ---
    gold_map = None
    gold_map_path = Path(GOLD_STANDARD_MAP_FNAME) # Assumes it's in the script's directory
    if gold_map_path.is_file():
        try:
            with open(gold_map_path, 'r', encoding='utf-8') as f:
                gold_map = json.load(f)
            print(f"✔ Loaded gold standard map: '{gold_map_path}'")
        except Exception as e:
            print(f"❌ Error loading gold standard map '{gold_map_path}': {e}. Semantic mapping/ranking may fail.")
    else:
        print(f"⚠️ Warning: Gold standard map not found at '{gold_map_path}'. Semantic mapping/ranking will be skipped.")


    # --- Initialize Cache ---
    print("⏳ Loading JSON data & caching...")
    cache = None # Initialize cache to None
    try:
        cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, CLASS_NAMES)
        if not cache.get_all_files():
             print("⚠️ Warning: Cache initialized, but no generated JSON files were loaded successfully.")
             # Allow script to continue, but subsequent steps will likely be skipped or fail gracefully
        else:
             print("✔ Cache ready.")
    except Exception as e:
        print(f"❌❌❌ CRITICAL ERROR during MetricCache initialization: {e}")
        traceback.print_exc()
        print("💀 Exiting due to cache initialization failure.")
        sys.exit(1)


    # --- Define output paths ---
    structural_csv = REPORTS_OUTPUT_DIR / "StructuralPreservationReport.csv"
    counts_csv = REPORTS_OUTPUT_DIR / "Counts_TargetClasses.csv"
    metrics_per_file_csv = REPORTS_OUTPUT_DIR / "MethodMetrics_PerFile.csv"
    metrics_summary_csv = REPORTS_OUTPUT_DIR / "MethodMetrics_Summary.csv"
    variability_csv = REPORTS_OUTPUT_DIR / "VariabilityMetrics.csv"
    core_methods_csv = REPORTS_OUTPUT_DIR / "CoreMethods_TopN.csv"
    coverage_csv = REPORTS_OUTPUT_DIR / "CoverageMetrics.csv"
    consensus_csv = REPORTS_OUTPUT_DIR / "ConsensusStrength.csv"
    diversity_csv = REPORTS_OUTPUT_DIR / "DiversityMetrics.csv"
    exclusive_csv = REPORTS_OUTPUT_DIR / "ExclusiveMethods.csv"
    annot_summary_csv = REPORTS_OUTPUT_DIR / "AnnotationMetrics_Summary.csv"
    uc_freq_csv = REPORTS_OUTPUT_DIR / "UC_Frequency_Global.csv"
    action_freq_csv = REPORTS_OUTPUT_DIR / "Action_Frequency_Global.csv"
    uc_method_csv = REPORTS_OUTPUT_DIR / "UC_Method_Report.csv"
    bootstrap_csv = REPORTS_OUTPUT_DIR / "BootstrapOverlap.csv"
    jaccard_global_csv = REPORTS_OUTPUT_DIR / "JaccardMatrix_Global.csv"
    jaccard_per_class_dir = REPORTS_OUTPUT_DIR / "Jaccard_PerClass"
    ranking_csv = REPORTS_OUTPUT_DIR / "LLM_Final_Ranking.csv"
    added_cls_report_csv = REPORTS_OUTPUT_DIR / "Added_Classes_LLM_Counts.csv"
    method_annot_report_csv = REPORTS_OUTPUT_DIR / "Method_Annotation_Details.csv"
    cls_focus_csv = REPORTS_OUTPUT_DIR / "Class_Focus_CoreMethods.csv"
    placement_consistency_csv = REPORTS_OUTPUT_DIR / "Method_Placement_Consistency.csv"
    stable_core_csv = REPORTS_OUTPUT_DIR / "LLM_Stable_Core_Methods.csv"
    core_perc_csv = REPORTS_OUTPUT_DIR / "Core_Method_Percentages.csv"
    disagreement_csv = REPORTS_OUTPUT_DIR / "Placement_Disagreement_CoreMethods.csv"
    method_action_mapping_csv = REPORTS_OUTPUT_DIR / "Method_Action_Mapping.csv" # Mapping file path

    # --- Initialize results DataFrames ---
    structural_df, counts_df_cls_rows, counts_df_files_rows, metrics_per_file = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    method_summary, variability, coverage, diversity, exclusives = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    annot_summary, uc_freq, action_freq, uc_report = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    added_cls_counts, method_annot, boot_overlap, jaccard_global = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cls_focus, plc_agg, disagree_df, stable_df, core_perc_summary = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mapping_df = pd.DataFrame()
    semantic_sim_summary_df = pd.DataFrame()
    rank_df = pd.DataFrame()
    per_cls_jaccard = {}
    consensus = np.nan

    # --- Calculations & Saving (Only if cache is valid and files were loaded) ---
    if cache and cache.get_all_files():
        # --- Structure & Counts ---
        print("📊 Calculating structural preservation...")
        structural_df = cache.get_structure_reports_df()
        if not structural_df.empty: structural_df.to_csv(structural_csv, index=False); print(f"✔ Saved: {structural_csv}")
        else: print(f"ℹ Skipping: {structural_csv}")

        print("📊 Calculating method counts...")
        counts_df_cls_rows = calculate_counts(cache)
        if not counts_df_cls_rows.empty: counts_df_cls_rows.to_csv(counts_csv); print(f"✔ Saved: {counts_csv}")
        else: print(f"ℹ Skipping: {counts_csv}")

        print("⚙️ Creating transposed counts...")
        if not counts_df_cls_rows.empty:
            try:
                counts_df_files_rows=counts_df_cls_rows.T;
                counts_df_files_rows.index.name=COL_FILE;
                if COL_TOTAL_METHODS not in counts_df_files_rows.columns:
                    cls_cols=[c for c in counts_df_files_rows.columns if c in cache.class_names];
                    counts_df_files_rows[COL_TOTAL_METHODS]=counts_df_files_rows[cls_cols].sum(axis=1).astype(int) if cls_cols else 0
                else: counts_df_files_rows[COL_TOTAL_METHODS]=counts_df_files_rows[COL_TOTAL_METHODS].fillna(0).astype(int)
                counts_df_files_rows[COL_MODEL]=counts_df_files_rows.index.map(lambda f: cache.file_info.get(f,{}).get('model'))
                counts_df_files_rows[COL_RUN]=counts_df_files_rows.index.map(lambda f: cache.file_info.get(f,{}).get('run'))
                if COL_MODEL not in counts_df_files_rows.columns: counts_df_files_rows[COL_MODEL] = None
                if COL_RUN not in counts_df_files_rows.columns: counts_df_files_rows[COL_RUN] = None
                cls_present=sorted([c for c in counts_df_files_rows.columns if c in cache.class_names]);
                order=[COL_MODEL,COL_RUN]+cls_present+[COL_TOTAL_METHODS];
                order=[c for c in order if c in counts_df_files_rows.columns];
                if order: counts_df_files_rows=counts_df_files_rows[order]
                else: print("Warning: No valid columns for transposed counts ordering.")
                for c in cls_present:
                    if c in counts_df_files_rows.columns: counts_df_files_rows[c]=counts_df_files_rows[c].fillna(0).astype(int)
            except Exception as e: print(f" Error preparing transposed counts: {e}"); counts_df_files_rows=pd.DataFrame()
        else: counts_df_files_rows=pd.DataFrame()


        # --- Per File / Aggregated Metrics ---
        print("📊 Calculating per-file metrics...")
        metrics_per_file = cache.get_per_file_method_metrics()
        if not metrics_per_file.empty: metrics_per_file.to_csv(metrics_per_file_csv, index=False); print(f"✔ Saved: {metrics_per_file_csv}")
        else: print(f"ℹ Skipping: {metrics_per_file_csv}")

        print("📊 Aggregating method metrics...")
        if not metrics_per_file.empty: method_summary = calculate_method_metrics_summary(cache, metrics_per_file)
        if not method_summary.empty: method_summary.to_csv(metrics_summary_csv, index=False); print(f"✔ Saved: {metrics_summary_csv}")
        else: print(f"ℹ Skipping: {metrics_summary_csv}")

        # --- Variability, Diversity ---
        print("📊 Calculating variability...")
        if not counts_df_files_rows.empty and COL_TOTAL_METHODS in counts_df_files_rows.columns and counts_df_files_rows[COL_TOTAL_METHODS].notnull().any():
            variability = calculate_variability(cache, counts_df_files_rows)
        if not variability.empty: variability.to_csv(variability_csv, index=False); print(f"✔ Saved: {variability_csv}")
        else: print(f"ℹ Skipping: {variability_csv}")

        print("📊 Calculating diversity & exclusives...")
        if not counts_df_files_rows.empty and COL_MODEL in counts_df_files_rows.columns and any(c in counts_df_files_rows.columns for c in CLASS_NAMES):
            diversity = calculate_diversity(cache, counts_df_files_rows)
        if not diversity.empty: diversity.to_csv(diversity_csv, index=False); print(f"✔ Saved: {diversity_csv}")
        else: print(f"ℹ Skipping: {diversity_csv}")

        exclusives = generate_exclusive_methods_report(cache)
        if not exclusives.empty: exclusives.to_csv(exclusive_csv, index=False); print(f"✔ Saved: {exclusive_csv}")
        else: print(f"ℹ Skipping: {exclusive_csv}")

        # --- Coverage & Consensus ---
        print("📊 Calculating coverage & consensus...")
        core_methods = cache.get_core_methods_list(TOP_N_CORE)
        if core_methods:
            generate_core_methods_report(cache, TOP_N_CORE).to_csv(core_methods_csv, index=False); print(f"✔ Saved: {core_methods_csv}")
            coverage, consensus = calculate_coverage(cache, core_methods) # consensus is scalar
            if not coverage.empty: coverage.to_csv(coverage_csv, index=False); print(f"✔ Saved: {coverage_csv}")
            else: print(f"ℹ Skipping: {coverage_csv}")
            pd.DataFrame([{"ConsensusStrength": consensus}]).to_csv(consensus_csv, index=False); print(f"✔ Saved: {consensus_csv}")
        else: print(f"ℹ Skipping coverage, core methods, consensus reports (no core methods identified).")

        # --- Annotation Metrics & Reports ---
        print("📊 Calculating annotation metrics & reports...")
        if not metrics_per_file.empty:
            annot_summary = calculate_annotation_metrics(cache, metrics_per_file)
            if not annot_summary.empty: annot_summary.to_csv(annot_summary_csv, index=False); print(f"✔ Saved: {annot_summary_csv}")
            else: print(f"ℹ Skipping: {annot_summary_csv}")

            uc_freq = generate_uc_frequency_report(cache)
            if not uc_freq.empty: uc_freq.to_csv(uc_freq_csv, index=False); print(f"✔ Saved: {uc_freq_csv}")
            else: print(f"ℹ Skipping: {uc_freq_csv}")

            action_freq = generate_action_frequency_report(cache)
            if not action_freq.empty: action_freq.to_csv(action_freq_csv, index=False); print(f"✔ Saved: {action_freq_csv}")
            else: print(f"ℹ Skipping: {action_freq_csv}")

            uc_report = generate_uc_method_report(cache)
            if not uc_report.empty: uc_report.to_csv(uc_method_csv, index=False); print(f"✔ Saved: {uc_method_csv}")
            else: print(f"ℹ Skipping: {uc_method_csv}")
        else: print(f"ℹ Skipping annotation calculations & reports (per-file metrics missing).")

        # --- Added Class Consensus ---
        print("📊 Calculating Added Class Counts...")
        added_cls_counts = calculate_added_class_llm_counts(cache)
        if not added_cls_counts.empty: added_cls_counts.to_csv(added_cls_report_csv, index=False); print(f"✔ Saved: {added_cls_report_csv}")
        else: print(f"ℹ Skipping: {added_cls_report_csv}")

        # --- Detailed Method Annotation Report ---
        print("📊 Generating Method Annotation Details (Target Classes)...")
        method_annot = generate_method_annotation_report(cache, target_classes_only=True)
        if not method_annot.empty: method_annot.to_csv(method_annot_report_csv, index=False); print(f"✔ Saved: {method_annot_report_csv}")
        else: print(f"ℹ Skipping: {method_annot_report_csv}")

        # --- Method Consensus Analysis ---
        print("📊 Analyzing Method Consensus...")
        if not method_annot.empty and core_methods:
            core_details = method_annot[method_annot[COL_METHOD_NAME].isin(set(core_methods))].copy()
            if not core_details.empty:
                 print("  Analyzing class focus...")
                 cls_focus=core_details.groupby('Class')[COL_METHOD_NAME].count().reset_index(name='CoreMethodInstances').sort_values('CoreMethodInstances', ascending=False)
                 cls_focus.to_csv(cls_focus_csv, index=False); print(f"✔ Saved: {cls_focus_csv}")

                 print("  Analyzing placement consistency...")
                 plc_agg=core_details.groupby(COL_METHOD_NAME)['Class'].agg([('Classes_Found_In', lambda x: sorted(list(x.unique()))), ('Num_Unique_Classes', 'nunique'), ('Total_Occurrences', 'count')]).reset_index()
                 plc_agg['Consistency_Score']=1 / plc_agg['Num_Unique_Classes']; plc_agg.sort_values(['Consistency_Score',COL_METHOD_NAME], ascending=[False,True], inplace=True)
                 plc_agg.to_csv(placement_consistency_csv, index=False); print(f"✔ Saved: {placement_consistency_csv}")

                 print("  Analyzing placement disagreement...")
                 disagree=plc_agg[plc_agg['Num_Unique_Classes']>1].copy(); details_disagree=[]
                 if not disagree.empty:
                      for _, row in disagree.iterrows():
                           m_name=row[COL_METHOD_NAME]; instances=core_details[core_details[COL_METHOD_NAME]==m_name]; cls_counts=instances['Class'].value_counts(); p_cls=cls_counts.idxmax(); p_count=cls_counts.max(); llms=sorted(list(instances[COL_MODEL].unique()))
                           details_disagree.append({COL_METHOD_NAME:m_name, 'Num_Unique_Classes':row['Num_Unique_Classes'], 'Classes_Found_In':", ".join(row['Classes_Found_In']), 'Primary_Class':p_cls, 'Primary_Class_Count':p_count, 'Total_Occurrences':row['Total_Occurrences'], 'Involved_LLM_Count':len(llms), 'Involved_LLMs':", ".join(llms)})
                      disagree_df=pd.DataFrame(details_disagree).sort_values(['Num_Unique_Classes',COL_METHOD_NAME], ascending=[False,True]); disagree_df.to_csv(disagreement_csv, index=False); print(f"✔ Saved: {disagreement_csv}")
                 else: print(f"ℹ Skipping: {disagreement_csv} (No disagreements found).")

                 print("  Analyzing stable core methods...")
                 stable_core=defaultdict(list); EXPECTED_RUNS=10
                 run_counts=core_details.groupby([COL_MODEL, COL_METHOD_NAME])[COL_RUN].nunique().reset_index(); stable=run_counts[run_counts[COL_RUN]>=EXPECTED_RUNS]
                 for model, grp in stable.groupby(COL_MODEL)[COL_METHOD_NAME]: stable_core[model] = sorted(list(grp))
                 stable_df=pd.DataFrame([{'Model':k,'Stable_Core_Methods':', '.join(v)} for k,v in stable_core.items()])
                 if not stable_df.empty: stable_df.to_csv(stable_core_csv, index=False); print(f"✔ Saved: {stable_core_csv}")
                 else: print(f"ℹ Skipping: {stable_core_csv} (No stable methods found).")

                 print("  Calculating core method percentages...")
                 if not metrics_per_file.empty and not coverage.empty and not method_summary.empty:
                      core_per_run=core_details.groupby(COL_FILE)[COL_METHOD_NAME].count().reset_index(name='CoreMethodsRunCount'); perc_df=pd.merge(metrics_per_file, core_per_run, on=COL_FILE, how='left'); perc_df['CoreMethodsRunCount']=perc_df['CoreMethodsRunCount'].fillna(0).astype(int)
                      perc_df['Percent_Core_Per_Run']=safe_divide('CoreMethodsRunCount', COL_TOTAL_METHODS, perc_df, default=0.0)*100
                      t_total_methods_col = f'Total_{COL_TOTAL_METHODS}'
                      if 'Total_Core_Occurrences' in coverage.columns and t_total_methods_col in method_summary.columns:
                          core_sum=coverage.set_index(COL_MODEL)['Total_Core_Occurrences']; total_sum=method_summary.set_index(COL_MODEL)[t_total_methods_col]
                          overall_perc=(core_sum/total_sum*100).fillna(0).rename('Percent_Core_Overall'); avg_perc=perc_df.groupby(COL_MODEL)['Percent_Core_Per_Run'].mean().rename('Avg_Percent_Core_Per_Run')
                          core_perc_summary=pd.concat([avg_perc, overall_perc], axis=1).reset_index(); core_perc_summary.to_csv(core_perc_csv, index=False, float_format="%.2f"); print(f"✔ Saved: {core_perc_csv}")
                      else: print(f"ℹ Skipping core percentage calculation (missing columns in coverage or method_summary).")
                 else: print(f"ℹ Skipping core percentage calculation (prerequisite dataframes missing or empty).")
            else: print("ℹ Skipping detailed consensus analysis (No core methods found in target classes).")
        else: print("ℹ Skipping detailed consensus analysis (Prerequisites missing: method_annot or core_methods list).")

        # --- Method to Action Mapping (using new function) ---
        print("📊 Generating detailed method-to-action mapping...")
        if gold_map:
            mapping_df = map_methods_to_actions(cache, gold_map, NLP_MODEL_NAME)
            if not mapping_df.empty:
                mapping_df.to_csv(method_action_mapping_csv, index=False, float_format="%.4f")
                print(f"✔ Saved: {method_action_mapping_csv}")
            else:
                print(f"ℹ Skipping saving mapping file (generation failed or returned empty).")
        else:
            print(f"ℹ Skipping method-to-action mapping (Gold standard map not loaded).")
            mapping_df = pd.DataFrame() # Ensure it's defined as empty

        # --- Derive Average Semantic Similarity for Ranking ---
        print("⚙️ Deriving average semantic similarity per model from mapping...")
        semantic_sim_summary_df = pd.DataFrame() # Default empty
        if not mapping_df.empty and 'Max_Similarity_Score' in mapping_df.columns:
            try:
                mapping_df['Max_Similarity_Score_Numeric'] = pd.to_numeric(mapping_df['Max_Similarity_Score'], errors='coerce')
                semantic_sim_summary_df = mapping_df.groupby(COL_MODEL)['Max_Similarity_Score_Numeric']\
                                                    .mean()\
                                                    .reset_index()\
                                                    .rename(columns={'Max_Similarity_Score_Numeric': COL_AVG_SIM_GOLD})
                if semantic_sim_summary_df.empty:
                    print("ℹ Info: Semantic similarity summary is empty after grouping.")
                else:
                    print("✔ Derived average semantic similarity scores.")
            except Exception as e:
                print(f"  [ERROR] Failed to derive average similarity from mapping: {e}")
                semantic_sim_summary_df = pd.DataFrame() # Reset on error
        else:
             print("ℹ Skipping derivation of average similarity (mapping empty or scores missing).")


        # --- Overlap Matrices ---
        print("📊 Calculating overlap matrices...")
        if 'calculate_overlap' in globals() and callable(calculate_overlap):
            if not counts_df_files_rows.empty:
                 boot_overlap, jaccard_global = calculate_overlap(cache, counts_df_files_rows)
                 if not boot_overlap.empty and not boot_overlap.isnull().all(axis=None): boot_overlap.to_csv(bootstrap_csv, float_format="%.3f"); print(f"✔ Saved: {bootstrap_csv}")
                 else: print(f"ℹ Skipping: {bootstrap_csv}")
                 if not jaccard_global.empty and not jaccard_global.isnull().all(axis=None): jaccard_global.to_csv(jaccard_global_csv, float_format="%.3f"); print(f"✔ Saved: {jaccard_global_csv}")
                 else: print(f"ℹ Skipping: {jaccard_global_csv}")
            else: print(f"ℹ Skipping overlap calculations (transposed counts df empty).")
        else: print(" Error: calculate_overlap not found."); boot_overlap, jaccard_global = pd.DataFrame(), pd.DataFrame()

        print("📊 Calculating per-class overlap matrices...")
        per_cls_jaccard = calculate_per_class_overlap(cache)
        if per_cls_jaccard:
            jaccard_per_class_dir.mkdir(exist_ok=True); saved=False
            for cls, matrix in sorted(per_cls_jaccard.items()):
                if isinstance(matrix, pd.DataFrame) and not matrix.empty and not matrix.isnull().all(axis=None):
                     matrix.to_csv(jaccard_per_class_dir/f"JaccardMatrix_{cls}.csv", float_format="%.3f"); saved=True
            if saved: print(f"✔ Per-class Jaccard matrices saved in: '{jaccard_per_class_dir}'")
            else: print(f"ℹ No valid per-class Jaccard matrices generated.")
        else: print(f"ℹ Skipping per-class Jaccard matrices (calculation returned empty).")

    else: # Cache initialization failed or no files loaded
         print("⚠️ Skipping most calculations because Cache object is invalid or no JSON files were loaded.")

    # --- Final Ranking ---
    print("🏆 Calculating final ranking...")
    # Gather available summary dataframes for ranking
    dfs_for_rank = {
        'variability': locals().get('variability'),
        'method_summary': locals().get('method_summary'),
        'diversity': locals().get('diversity'),
        'coverage': locals().get('coverage'),
        'semantic_sim': locals().get('semantic_sim_summary_df') # Use derived summary
    }
    # Define metrics to use and their direction
    metrics_for_rank = {
        COL_CV: 'lower',
        COL_TOTAL_REDUNDANCY: 'lower',
        COL_NORM_ENTROPY: 'higher',
        COL_UNIQUE_CORE: 'higher',
        COL_AVG_SIM_GOLD: 'higher' # Keep using the average score
    }
    available_metrics = {}
    all_models_found = set() # Keep track of all models across summary dfs

    for name, df_obj in dfs_for_rank.items():
        if isinstance(df_obj, pd.DataFrame) and not df_obj.empty:
            metric_col = next((m for m in metrics_for_rank if m in df_obj.columns), None)
            if metric_col and COL_MODEL in df_obj.columns:
                df_filtered = df_obj[[COL_MODEL, metric_col]].dropna(subset=[metric_col]).copy()
                if not df_filtered.empty:
                    available_metrics[metric_col] = df_filtered
                    all_models_found.update(df_filtered[COL_MODEL].unique())
                else: print(f"ℹ Info for Ranking: Dataframe for {name} ('{metric_col}') is empty after dropping NaNs.")
            else: print(f"ℹ Info for Ranking: Data for {name} ('{metric_col or 'target metric'}' or '{COL_MODEL}') column missing.")
        else: print(f"ℹ Info for Ranking: Dataframe for {name} is missing or empty.")

    if len(available_metrics) >= 1:
        rank_df = pd.DataFrame({COL_MODEL: sorted(list(all_models_found))})
        for metric_col, df_to_merge in available_metrics.items():
             rank_df = pd.merge(rank_df, df_to_merge, on=COL_MODEL, how='left') # Use left merge

        norm_cols_present=[]
        for col, direction_simple in metrics_for_rank.items():
            direction = f"{direction_simple}_is_better" # Reconstruct
            if col in rank_df.columns:
                norm_col=f"{col}_Norm"
                rank_df[col] = pd.to_numeric(rank_df[col], errors='coerce')
                vals=rank_df[col].dropna()
                if len(vals)>1:
                    min_v, max_v = vals.min(), vals.max()
                    if min_v==max_v: rank_df[norm_col]=np.where(rank_df[col].notna(), 0.5, np.nan)
                    else:
                         rank_df[norm_col]=(rank_df[col]-min_v)/(max_v-min_v)
                         if direction=='lower_is_better': rank_df[norm_col]=1.0-rank_df[norm_col]
                    norm_cols_present.append(norm_col)
                elif len(vals)==1:
                    rank_df[norm_col]=np.where(rank_df[col].notna(), 0.5, np.nan)
                    norm_cols_present.append(norm_col)
                else: rank_df[norm_col]=np.nan

        if norm_cols_present:
            rank_df[COL_FINAL_SCORE]=rank_df[norm_cols_present].mean(axis=1, skipna=True)
            print(f"✔ Calculated final score using: {norm_cols_present}")
        else:
            print("ℹ Info for Ranking: No metrics were available/normalized to calculate a final score.")
            rank_df[COL_FINAL_SCORE]=np.nan

        f_cols_order=[COL_MODEL]+[c for c in metrics_for_rank if c in rank_df.columns]+norm_cols_present+[COL_FINAL_SCORE]
        f_cols_present=[c for c in f_cols_order if c in rank_df.columns]
        rank_df=rank_df[f_cols_present].sort_values(COL_FINAL_SCORE, ascending=False, na_position='last')
        rank_df.to_csv(ranking_csv, index=False, float_format="%.3f"); print(f"✔ Saved: {ranking_csv}")
    else: print(f"ℹ Skipping composite ranking ({ranking_csv}): No valid metrics data available.")

    print("\n✨ Metrics pipeline execution finished.")


if __name__ == "__main__":
    main()