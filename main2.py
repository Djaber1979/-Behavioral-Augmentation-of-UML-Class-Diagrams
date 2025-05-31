#!/usr/bin/env python3
"""
main2.py - Consolidated LLM-enhanced class diagram metrics pipeline
Generates an Excel workbook (or CSVs if no Excel engine) with multiple sheets/files summarizing:
- Structural preservation
- Method counts and per-file metrics
- Aggregated method metrics
- Variability, diversity, core & coverage analyses
- Annotation & semantic mapping/coverage
- Overlap (bootstrap & Jaccard) global and per-class
- Final consolidated summary per model
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import t
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# --- Configuration ---
JSON_INPUT_DIR = Path("JSON")
BASELINE_JSON_FNAME = "methodless.json"
GOLD_STANDARD_MAP_FNAME = "uc_action_method_map.json"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TOP_N_CORE = 38
CLASS_NAMES = [
    "ValidationResult", "Coordinates", "Address", "TimeRange", "OpeningHours",
    "UserAccount", "UserProfile", "UserCredentials", "RoleManager",
    "ServiceRequest", "CollectionRequest", "TransportRequest", "RoutePlan", "WasteJourney", "Product",
    "SearchFilter", "CollectionPoint", "NotificationTemplate",
    "NotificationService", "ServiceProvider", "PlatformService",
]
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
NLP_MODEL_NAME = 'all-mpnet-base-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)

# --- Utilities ---
def split_method_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1 \2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1)
    s3 = s2.replace('_', ' ')
    return ' '.join(s3.lower().split())

def safe_divide(numerator, denominator, default=np.nan):
    try:
        result = numerator / denominator
    except Exception:
        return default
    if isinstance(result, (int, float, np.floating, np.integer)):
        val = float(result)
        return default if math.isnan(val) else val
    if hasattr(result, 'fillna'):
        return result.fillna(default)
    try:
        arr = np.array(result, dtype=float)
        arr[np.isnan(arr)] = default
        return arr
    except Exception:
        return result

# --- JSON Structure Extraction ---
def extract_structure_from_json(json_data: dict) -> dict[str, set]:
    elements = {k: set() for k in ['packages','enums','enum_values','classes','attributes','relationships']}
    if not isinstance(json_data, dict):
        return elements
    for pkg in json_data.get('packages', []):
        elements['packages'].add(pkg)
    for enum in json_data.get('enums', []):
        name = enum.get('name')
        if name:
            elements['enums'].add(name)
            for v in enum.get('values', []):
                elements['enum_values'].add(f"{name}::{v}")
    for cls in json_data.get('classes', []):
        cname = cls.get('name')
        if cname:
            elements['classes'].add(cname)
            for attr in cls.get('attributes', []):
                aname = attr.get('name'); atype = attr.get('type')
                if aname and atype is not None:
                    elements['attributes'].add(f"{cname}::{aname}: {atype}")
    for rel in json_data.get('relationships', []):
        src = rel.get('source'); tgt = rel.get('target')
        if src and tgt:
            sym = rel.get('type_symbol','--'); lbl = rel.get('label','')
            s_card = rel.get('source_cardinality',''); t_card = rel.get('target_cardinality','')
            elements['relationships'].add(
                f"{src}{' '+s_card if s_card else ''} {sym}{' '+t_card if t_card else ''} {tgt}{' : '+lbl if lbl else ''}"
            )
    return elements

# --- Structure Comparison ---
def compare_structures(baseline: dict[str, set], enriched: dict[str, set]) -> dict:
    report = {}
    total_baseline = 0
    preserved = 0
    keys = set(baseline.keys()) | set(enriched.keys())
    for k in sorted(keys):
        base = baseline.get(k, set()); enrich = enriched.get(k, set())
        p = len(base & enrich); m = len(base - enrich); a = len(enrich - base)
        report[f"{k}_preserved"] = p
        report[f"{k}_missing"] = m
        report[f"{k}_added"] = a
        report[f"{k}_total_baseline"] = len(base)
        total_baseline += len(base)
        preserved += p
    report['Overall_Preservation_%'] = round((preserved/total_baseline)*100,2) if total_baseline>0 else 100.0
    report['Total_Baseline_Elements'] = total_baseline
    report['Total_Preserved_Elements'] = preserved
    report['Total_Added_Elements'] = sum(len(enriched.get(k,set())-baseline.get(k,set())) for k in enriched)
    return report

# --- Gold Map Preprocessing ---
def _preprocess_gold_map(gold_map: dict) -> dict:
    details = {
        'all_uc_ids': set(), 'all_actions': set(),
        'uc_to_actions': defaultdict(set),
        'action_to_details': {}, 'class_to_uc_ids': defaultdict(set),
        'uc_action_counts': Counter()
    }
    for uc_id, entries in gold_map.items():
        uc = str(uc_id)
        details['all_uc_ids'].add(uc)
        count = 0
        if isinstance(entries, list):
            for e in entries:
                action = e.get('action'); cls = e.get('assigned_class')
                if action and cls:
                    details['all_actions'].add(action)
                    details['uc_to_actions'][uc].add(action)
                    details['action_to_details'][action] = {'uc_id': uc, 'assigned_class': cls}
                    details['class_to_uc_ids'][cls].add(uc)
                    count += 1
        details['uc_action_counts'][uc] = count
    details['total_actions'] = len(details['all_actions'])
    details['total_ucs'] = len(details['all_uc_ids'])
    return details

# --- MetricCache ---
class MetricCache:
    def __init__(self, json_dir: Path, baseline_fname: str, class_names: list[str]):
        baseline = json_dir/baseline_fname
        self.baseline_structure = extract_structure_from_json(
            json.load(open(baseline)) if baseline.is_file() else {}
        )
        self.files = [p.name for p in json_dir.glob("*.json") if p.name!=baseline_fname]
        self.json_data = {f: json.load(open(json_dir/f)) for f in self.files}
        self.file_info = {}
        for f in self.files:
            stem = Path(f).stem
            model = re.sub(r"_run\d+$","",stem)
            run = re.search(r'_run(\d+)$',stem)
            self.file_info[f] = {'model':model,'run':run.group(1) if run else '1'}
        self.global_details = []
        for f,data in self.json_data.items():
            for cls in data.get('classes',[]):
                for m in cls.get('methods',[]):
                    name=m.get('name')
                    if name:
                        self.global_details.append({
                            'name':name,'class':cls.get('name',''),
                            'file':f,'ucs':m.get('annotation',{})
                                          .get('uc_references',[]) or [],
                            'action':m.get('annotation',{}).get('uc_action','')
                        })
        self.global_method_counter = Counter(m['name'] for m in self.global_details)
        self.global_uc_counter     = Counter(uc for m in self.global_details for uc in m['ucs'])
        self.global_action_counter = Counter(m['action'] for m in self.global_details if m['action'])
        # load gold map
        gold_map = json.load(open(GOLD_STANDARD_MAP_FNAME))
        self.gold = _preprocess_gold_map(gold_map)

    def get_structure_reports_df(self) -> pd.DataFrame:
        rows=[]
        for f,data in self.json_data.items():
            enriched = extract_structure_from_json(data)
            rows.append({**compare_structures(self.baseline_structure,enriched),'File':f})
        return pd.DataFrame(rows)

    def get_per_file_method_metrics(self) -> pd.DataFrame:
        rows = []
        # build the top‐methods set once per call
        top_set = {name for name, _ in self.global_method_counter.most_common(TOP_N_CORE)}

        for f, data in self.json_data.items():
            methods = []
            extra = 0
            for cls in data.get('classes', []):
                if cls.get('name') in CLASS_NAMES:
                    methods += cls.get('methods', [])
                else:
                    extra += len(cls.get('methods', []))

            total = len(methods)
            # method names for the "Top_Methods" count
            names = [m.get('name') for m in methods if m.get('name')]

            # annotation flags per method
            has_uc     = [bool(m.get('annotation', {}).get('uc_references')) for m in methods]
            has_action = [bool(m.get('annotation', {}).get('uc_action'))        for m in methods]

            uc_only     = sum(u and not a for u, a in zip(has_uc,     has_action))
            action_only = sum(a and not u for u, a in zip(has_uc,     has_action))
            both        = sum(u and a     for u, a in zip(has_uc,     has_action))
            neither     = sum(not u and not a for u, a in zip(has_uc, has_action))

            top_count = sum(1 for n in names if n in top_set)

            rows.append({
                'Model': self.file_info[f]['model'],
                'Run':   self.file_info[f]['run'],
                'File':  f,
                'Target_Class_Methods': total,
                'Extra_Methods_Count': extra,
                'Unique_Method_Names': len(set(names)),
                'Redundancy': safe_divide(total, len(set(names))),
                'ParamRichness': np.mean([len(m.get('parameters', [])) for m in methods]) if methods else 0,
                'ReturnTypeCompleteness': safe_divide(
                    sum(bool(m.get('return_type') and m['return_type'].lower()!='void') for m in methods),
                    total
                ),
                'Top_Methods': top_count,
                'Percentage_Methods_With_UC':     safe_divide(uc_only,     total),
                'Percentage_Methods_With_Action': safe_divide(action_only, total),
                'Percentage_Methods_With_Both':   safe_divide(both,        total),
                'Percentage_Methods_Without':    safe_divide(neither,     total),
            })

        return pd.DataFrame(rows)

def generate_core_methods_report(cache: MetricCache, top_n: int) -> pd.DataFrame:
    # 1) pick top N methods by global frequency
    common = cache.global_method_counter.most_common(top_n)
    df = pd.DataFrame(common, columns=['MethodName','GlobalFrequency'])
    
    # 2) get every generated method with its cosine‐based best match
    ann     = generate_method_annotation_report(cache)
    mapping = map_methods_to_actions(ann, cache)
    
    # 3) max cosine similarity per core method
    max_sim   = mapping.groupby('MethodName')['SimilarityScore'].max()
    
    # 4) class where each core method appears most often
    cls_counts = mapping.groupby(['MethodName','Class']).size()
    top_class  = cls_counts.groupby(level=0).idxmax().apply(lambda x: x[1])
    
    # 5) LLM that generated the most instances of each core method
    llm_counts = mapping.groupby(['MethodName','Model']).size()
    top_llm    = llm_counts.groupby(level=0).idxmax().apply(lambda x: x[1])
    
    # 6) stitch back into our DataFrame
    df['MaxSimilarity'] = df['MethodName'].map(max_sim).fillna(0)
    df['TopClass']      = df['MethodName'].map(top_class).fillna('')
    df['TopLLM']        = df['MethodName'].map(top_llm).fillna('')
    
    return df


def calculate_counts(cache: MetricCache) -> pd.DataFrame:
    """
    Counts of methods in each target class per run (file). Rows are classes plus a Total row,
    columns are individual JSON filenames (runs).
    """
    # collect counts per file
    data = []
    for f, data_json in cache.json_data.items():
        row = {'File': f}
        for cls in CLASS_NAMES:
            cnt = 0
            for c in data_json.get('classes', []):
                if c.get('name') == cls:
                    cnt = len(c.get('methods', []))
                    break
            row[cls] = cnt
        data.append(row)
    # pivot: classes as rows, files as columns
    df = pd.DataFrame(data).set_index('File').T
    # append total row: total methods per run
    df.loc['Total'] = df.sum()
    # reset index
    df = df.reset_index().rename(columns={'index': 'Class'})
    return df


def calculate_method_metrics_summary(perfile_df: pd.DataFrame) -> pd.DataFrame:
    grp = perfile_df.groupby('Model')
    summary = pd.DataFrame({
        'Avg_Target_Class_Methods': grp['Target_Class_Methods'].mean(),
        'Std_Target_Class_Methods': grp['Target_Class_Methods'].std(),
        'Avg_Redundancy': grp['Redundancy'].mean(),
        'Avg_ParamRichness': grp['ParamRichness'].mean(),
        'Avg_ReturnTypeCompleteness': grp['ReturnTypeCompleteness'].mean(),
        'Avg_Percentage_Methods_With_UC': grp['Percentage_Methods_With_UC'].mean(),
        'Total_Extra_Methods': grp['Extra_Methods_Count'].sum(),
        'Total_Unique_Method_Names': grp['Unique_Method_Names'].sum()
    })
    return summary.reset_index()

def calculate_variability(perfile_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for mdl, grp in perfile_df.groupby('Model'):
        vals = grp['Target_Class_Methods'].values
        n=len(vals); m=vals.mean(); s=vals.std(ddof=1)
        cv = s/m if m else np.nan
        ci = t.interval(0.95, n-1, loc=m, scale=s/math.sqrt(n)) if n>1 else (m,m)
        x=np.arange(1,n+1)
        slope = np.polyfit(x,vals,1)[0]
        rows.append({'Model':mdl,'Mean':m,'CV':cv,'CI_low':ci[0],'CI_high':ci[1],'ConvergenceSlope':slope,'NumRuns':n})
    return pd.DataFrame(rows)

def calculate_diversity(cache: MetricCache) -> pd.DataFrame:
    rows=[]
    model_class_counts=defaultdict(lambda: Counter())
    for f,data in cache.json_data.items():
        mdl=cache.file_info[f]['model']
        for cls in data.get('classes',[]):
            name=cls.get('name')
            if name in CLASS_NAMES:
                model_class_counts[mdl][name]+=len(cls.get('methods',[]))
    model_methods=defaultdict(set)
    for f,data in cache.json_data.items():
        mdl=cache.file_info[f]['model']
        for cls in data.get('classes',[]):
            if cls.get('name') in CLASS_NAMES:
                for m in cls.get('methods',[]):
                    if m.get('name'): model_methods[mdl].add(m['name'])
    for mdl,counts in model_class_counts.items():
        total=sum(counts.values())
        props=[v/total for v in counts.values()] if total>0 else []
        ent = -sum(p*math.log2(p) for p in props if p>0)
        norm_ent = ent/math.log2(len(CLASS_NAMES)) if CLASS_NAMES else np.nan
        gini = 1 - sum(p*p for p in props)
        other_methods = set().union(*(model_methods[m] for m in model_methods if m!=mdl))
        exclusive = len(model_methods[mdl] - other_methods)
        rows.append({'Model':mdl,'Entropy':ent,'NormalizedEntropy':norm_ent,'Gini':gini,'ExclusiveMethodsCount':exclusive})
    return pd.DataFrame(rows)

def calculate_coverage(cache: MetricCache, core_methods: list[str]) -> pd.DataFrame:
    results=[]; core_set=set(core_methods)
    model_methods=defaultdict(list)
    for f,data in cache.json_data.items():
        mdl=cache.file_info[f]['model']
        for cls in data.get('classes',[]):
            if cls.get('name') in CLASS_NAMES:
                for m in cls.get('methods',[]):
                    if m.get('name'): model_methods[mdl].append(m['name'])
    for mdl,names in model_methods.items():
        unique_core=len(set(names)&core_set)
        total_occ=sum(1 for n in names if n in core_set)
        results.append({'Model':mdl,'Unique_Core_Methods':unique_core,'Total_Core_Occurrences':total_occ})
    return pd.DataFrame(results)

# --- Annotation & Semantic Mapping/Coverage Functions ---
def generate_method_annotation_report(cache: MetricCache) -> pd.DataFrame:
    """
    Detailed listing of every method in the target classes, showing its generated annotations
    Columns: Model, Run, File, Class, MethodName, Signature, Has_UC_Annotation, UC_References, UC_Action
    """
    rows = []
    for file_name, data in cache.json_data.items():
        mdl = cache.file_info[file_name]['model']
        run = cache.file_info[file_name]['run']
        for cls_entry in data.get('classes', []):
            cname = cls_entry.get('name')
            if cname not in CLASS_NAMES:
                continue
            for m in cls_entry.get('methods', []):
                name = m.get('name')
                if not name:
                    continue
                ann = m.get('annotation') or {}
                ucs = ann.get('uc_references') if isinstance(ann.get('uc_references'), list) else []
                action = ann.get('uc_action') or ''
                has_uc = bool(ucs)
                signature = m.get('signature') or ''
                rows.append({
                    'Model': mdl,
                    'Run': run,
                    'File': file_name,
                    'Class': cname,
                    'MethodName': name,
                    'Signature': signature,
                    'Has_UC_Annotation': has_uc,
                    'UC_References': ','.join(ucs),
                    'UC_Action': action
                })
    return pd.DataFrame(rows)

def generate_uc_method_report(annotation_df: pd.DataFrame) -> pd.DataFrame:
    return annotation_df[annotation_df['Has_UC_Annotation']].copy()


def calculate_annotation_metrics(annotation_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    total_methods= len(annotation_df)
    for mdl,grp in annotation_df.groupby('Model'):
        methods_with_uc=sum(grp['Has_UC_Annotation'])
        pct_per_file = grp.groupby('File')['Has_UC_Annotation'].mean().mean()
        total_uc_refs = grp['UC_References'].apply(lambda x: len(x.split(',')) if x else 0).sum()
        unique_ucs = set().union(*(set(x.split(',')) for x in grp['UC_References'] if x))
        avg_ucs_per_file = grp.groupby('File')['UC_References'].apply(lambda lst: len(set().union(*(l.split(',') for l in lst if l)))).mean()
        unique_actions = set(grp['UC_Action'][grp['UC_Action']!=''])
        avg_actions_per_file = grp.groupby('File')['UC_Action'].nunique().mean()
        rows.append({
            'Model':mdl,
            'Total_Methods_With_Any_UC':methods_with_uc,
            'Avg_Pct_Methods_With_UC_per_File':pct_per_file,
            'Total_UC_References':total_uc_refs,
            'Total_Unique_UCs':len(unique_ucs),
            'Avg_Unique_UCs_Per_File':avg_ucs_per_file,
            'Total_Unique_Actions':len(unique_actions),
            'Avg_Unique_Actions_Per_File':avg_actions_per_file
        })
    return pd.DataFrame(rows)


def generate_uc_frequency_report(cache: MetricCache) -> pd.DataFrame:
    return pd.DataFrame(cache.global_uc_counter.most_common(),columns=['UC_ID','GlobalFrequency'])


def generate_action_frequency_report(cache: MetricCache) -> pd.DataFrame:
    return pd.DataFrame(cache.global_action_counter.most_common(),columns=['Action','GlobalFrequency'])


def map_methods_to_actions(annotation_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    """
    Maps each generated method to the best-matching gold action using cosine similarity.
    If the method has UC annotations, restrict comparison to actions within those UCs.
    Otherwise, restrict to actions associated with the method's class.
    """
    model = SentenceTransformer(NLP_MODEL_NAME).to(DEVICE)
    rows = []
    for _, row in annotation_df.iterrows():
        split = split_method_name(row['MethodName'])
        emb = model.encode(split, convert_to_tensor=True, device=DEVICE)
        # determine candidate actions
        uc_refs = [uc for uc in row['UC_References'].split(',') if uc]
        if uc_refs:
            # restrict to actions within any of the annotated UCs
            candidates = set().union(*(cache.gold['uc_to_actions'].get(uc, set()) for uc in uc_refs))
        else:
            # restrict to actions for this method's class
            cls = row['Class']
            ucs_for_class = cache.gold['class_to_uc_ids'].get(cls, set())
            candidates = set().union(*(cache.gold['uc_to_actions'].get(uc, set()) for uc in ucs_for_class))
        if not candidates:
            # fallback to all actions
            candidates = cache.gold['all_actions']
        cand_list = list(candidates)
        cand_embs = model.encode(cand_list, convert_to_tensor=True, device=DEVICE)
        sims = cos_sim(emb, cand_embs)[0]
        idx = int(torch.argmax(sims.cpu()))
        best = cand_list[idx]
        score = float(sims[idx].cpu())
        rows.append({**row.to_dict(), 'SplitMethod': split,
                     'Best_Match_Action': best,
                     'SimilarityScore': score})
    return pd.DataFrame(rows)
def calculate_action_annotation_coverage(annotation_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    rows=[]
    total=cache.gold['total_actions']
    for mdl,grp in annotation_df.groupby('Model'):
        covered=set(grp['UC_Action'].dropna().unique())
        rows.append({'Model':mdl,'Annot_Covered_Action_Count':len(covered),'Annot_Action_Coverage_Percent':len(covered)/total*100})
    return pd.DataFrame(rows)


def calculate_action_annotation_coverage(annotation_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    rows=[]
    for mdl,grp in annotation_df.groupby('Model'):
        # unique annotated actions matching gold
        acts = set(grp['UC_Action'][grp['UC_Action'].isin(gold['all_actions'])])
        count = len(acts)
        pct = count/ gold['total_actions'] * 100
        # per UC
        uc_covs=[]
        for uc,acts_in_uc in gold['uc_to_actions'].items():
            annotated = set(grp['UC_Action'][grp['UC_Action'].isin(acts_in_uc)])
            if annotated:
                uc_covs.append(len(annotated)/len(acts_in_uc))
        avg_uc_cov = np.mean(uc_covs) if uc_covs else 0
        rows.append({'Model':mdl,'Annot_Covered_Action_Count':count,'Annot_Action_Coverage_%':pct,'Avg_Per_UC_Annot_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows)


def calculate_uc_annotation_coverage(annotation_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    rows=[]
    for mdl,grp in annotation_df.groupby('Model'):
        ucs_ann = set().union(*(set(x.split(',')) for x in grp['UC_References'] if x))
        count = len(ucs_ann)
        pct = count/ gold['total_ucs'] * 100
        # actions per annotated UC
        uc_covs=[]
        for uc in ucs_ann:
            acts_in_uc = gold['uc_to_actions'].get(uc, set())
            annotated = set(grp['UC_Action'][grp['UC_Action'].isin(acts_in_uc)])
            if acts_in_uc:
                uc_covs.append(len(annotated)/len(acts_in_uc))
        avg_uc_cov = np.mean(uc_covs) if uc_covs else 0
        rows.append({'Model':mdl,'Annot_Covered_UC_Count':count,'Annot_UC_Coverage_%':pct,'Avg_Per_UC_Annot_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows)


def calculate_action_semantic_coverage(map_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    rows=[]
    for mdl,grp in map_df.groupby('Model'):
        valid = grp[grp['Max_Similarity_Score']>=SEMANTIC_SIMILARITY_THRESHOLD]
        acts = set(valid['Best_Match_Action'])
        count=len(acts); pct=count/gold['total_actions']*100
        uc_covs=[]
        for uc,acts_in_uc in gold['uc_to_actions'].items():
            matched = acts & acts_in_uc
            if matched:
                uc_covs.append(len(matched)/len(acts_in_uc))
        avg_uc_cov = np.mean(uc_covs) if uc_covs else 0
        rows.append({'Model':mdl,'Sem_Covered_Action_Count':count,'Sem_Action_Coverage_%':pct,'Avg_Per_UC_Sem_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows)


def calculate_uc_semantic_coverage(map_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    rows=[]
    for mdl,grp in map_df.groupby('Model'):
        valid = grp[grp['Max_Similarity_Score']>=SEMANTIC_SIMILARITY_THRESHOLD]
        ucs=set(gold['action_to_details'][a]['uc_id'] for a in set(valid['Best_Match_Action']))
        count=len(ucs); pct=count/gold['total_ucs']*100
        # per UC action coverage
        uc_covs=[]
        for uc in ucs:
            acts_in_uc = gold['uc_to_actions'].get(uc,set())
            matched = set(valid['Best_Match_Action']) & acts_in_uc
            if acts_in_uc:
                uc_covs.append(len(matched)/len(acts_in_uc))
        avg_uc_cov=np.mean(uc_covs) if uc_covs else 0
        rows.append({'Model':mdl,'Sem_Covered_UC_Count':count,'Sem_UC_Coverage_%':pct,'Avg_Per_UC_Sem_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows)


def calculate_action_combined_coverage(annotation_df: pd.DataFrame, map_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    aa = calculate_action_annotation_coverage(annotation_df,gold).set_index('Model')
    sm = calculate_action_semantic_coverage(map_df,gold).set_index('Model')
    rows=[]
    for mdl in sorted(set(aa.index)|set(sm.index)):
        ann_acts = set(annotation_df[annotation_df['Model']==mdl]['UC_Action']) & gold['all_actions']
        sem_acts = set(map_df[map_df['Model']==mdl].query('Max_Similarity_Score>=@SEMANTIC_SIMILARITY_THRESHOLD')['Best_Match_Action'])
        union = ann_acts|sem_acts
        count=len(union); pct=count/gold['total_actions']*100
        # per UC
        uc_covs=[]
        for uc,acts_in_uc in gold['uc_to_actions'].items():
            if (acts_in_uc & union):
                uc_covs.append(len(acts_in_uc & union)/len(acts_in_uc))
        avg_uc_cov=np.mean(uc_covs) if uc_covs else 0
        rows.append({'Model':mdl,'Comb_Covered_Action_Count':count,'Comb_Action_Coverage_%':pct,'Avg_Per_UC_Comb_Action_Coverage':avg_uc_cov})
    return pd.DataFrame(rows)


def calculate_uc_combined_coverage(annotation_df: pd.DataFrame, map_df: pd.DataFrame, gold: dict) -> pd.DataFrame:
    ann_uc = calculate_uc_annotation_coverage(annotation_df,gold).set_index('Model')
    sem_uc = calculate_uc_semantic_coverage(map_df,gold).set_index('Model')
    rows=[]
    for mdl in sorted(set(ann_uc.index)|set(sem_uc.index)):
        u_anns = set().union(*(set(x.split(',')) for x in annotation_df.query('Model==@mdl')['UC_References'] if x))
        u_sems = set(gold['action_to_details'][a]['uc_id'] for a in map_df.query('Model==@mdl and Max_Similarity_Score>=@SEMANTIC_SIMILARITY_THRESHOLD')['Best_Match_Action'])
        union=u_anns|u_sems
        count=len(union); pct=count/gold['total_ucs']*100
        rows.append({'Model':mdl,'Comb_Covered_UC_Count':count,'Comb_UC_Coverage_%':pct})
    return pd.DataFrame(rows)


def calculate_added_class_llm_counts(cache: MetricCache) -> pd.DataFrame:
    baseline_classes=cache.baseline_structure['classes']
    cls_map=defaultdict(set)
    for f,data in cache.json_data.items():
        mdl=cache.file_info[f]['model']
        for c in data.get('classes',[]):
            name=c.get('name')
            if name and name not in baseline_classes:
                cls_map[name].add(mdl)
    rows=[]
    for cls,models in cls_map.items():
        rows.append({'Added_Class':cls,'LLM_Count':len(models),'LLM_List':','.join(sorted(models))})
    return pd.DataFrame(rows)

# --- Annotation & Semantic Mapping/Coverage ---
def generate_method_annotation_report(cache: MetricCache) -> pd.DataFrame:
    rows=[]
    for f,data in cache.json_data.items():
        mdl,run=cache.file_info[f]['model'],cache.file_info[f]['run']
        for cls in data.get('classes',[]):
            cname=cls.get('name')
            if cname in CLASS_NAMES:
                for m in cls.get('methods',[]):
                    name=m.get('name','')
                    ann=m.get('annotation',{}) or {}
                    ucs=ann.get('uc_references') if isinstance(ann.get('uc_references'),list) else []
                    action=ann.get('uc_action','') if isinstance(ann.get('uc_action',''),str) else ''
                    rows.append({
                        'Model':mdl,'Run':run,'File':f,'Class':cname,
                        'MethodName':name,'Has_UC_Annotation':bool(ucs),
                        'UC_References':','.join(ucs),'UC_Action':action
                    })
    return pd.DataFrame(rows)

def map_methods_to_actions(annotation_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    """
    For each generated method, find the best‐matching gold action by cosine similarity,
    and clamp similarity scores to the [-1, 1] range.
    """
    # split names into sentences
    sentences = [split_method_name(n) for n in annotation_df['MethodName']]
    # load & encode on the correct device
    model = SentenceTransformer(NLP_MODEL_NAME, device=DEVICE)
    emb_methods = model.encode(sentences, convert_to_tensor=True, device=DEVICE)
    # encode gold actions
    gold_actions = list(cache.gold['all_actions'])
    emb_gold     = model.encode(gold_actions,    convert_to_tensor=True, device=DEVICE)
    # compute cosine similarities and clamp
    sims = cos_sim(emb_methods, emb_gold).clamp(-1.0, 1.0)
    # for each method, pick the best match
    best_match = []
    scores     = []
    for row in sims:
        idx = int(torch.argmax(row))
        best_match.append(gold_actions[idx])
        # ensure the float is within [-1,1]
        scores.append(max(min(float(row[idx]), 1.0), -1.0))
    # assemble output
    df = annotation_df.copy()
    df['SplitMethod']       = sentences
    df['Best_Match_Action'] = best_match
    df['SimilarityScore']   = scores
    return df

def calculate_action_annotation_coverage(annotation_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    rows=[]
    for mdl,grp in annotation_df.groupby('Model'):
        covered=set(grp['UC_Action'].dropna().unique())
        total=cache.gold['total_actions']
        rows.append({'Model':mdl,'Annot_Covered_Action_Count':len(covered),
                     'Annot_Action_Coverage_Percent':len(covered)/total*100})
    return pd.DataFrame(rows)

def calculate_uc_annotation_coverage(annotation_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    rows=[]
    for mdl,grp in annotation_df.groupby('Model'):
        ucs=set(sum((u.split(',') for u in grp['UC_References'] if u),[]))
        total_uc=cache.gold['total_ucs']
        rows.append({'Model':mdl,'Annot_Covered_UC_Count':len(ucs),
                     'Annot_UC_Hit_Coverage_Percent':len(ucs)/total_uc*100})
    return pd.DataFrame(rows)

def calculate_action_semantic_coverage(mapping_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    rows=[]
    df=mapping_df.copy()
    df['Match']=df['SimilarityScore']>=SEMANTIC_SIMILARITY_THRESHOLD
    for mdl,grp in df.groupby('Model'):
        cov=set(grp.loc[grp['Match'],'Best_Match_Action'])
        total=cache.gold['total_actions']
        rows.append({'Model':mdl,'Sem_Covered_Action_Count':len(cov),
                     'Sem_Action_Coverage_Percent':len(cov)/total*100})
    return pd.DataFrame(rows)

def calculate_uc_semantic_coverage(mapping_df: pd.DataFrame, cache: MetricCache) -> pd.DataFrame:
    rows=[]
    df=mapping_df.copy()
    df['Match']=df['SimilarityScore']>=SEMANTIC_SIMILARITY_THRESHOLD
    for mdl,grp in df.groupby('Model'):
        actions=set(grp.loc[grp['Match'],'Best_Match_Action'])
        ucs=set(cache.gold['action_to_details'][a]['uc_id'] for a in actions)
        total_uc=cache.gold['total_ucs']
        rows.append({'Model':mdl,'Sem_Covered_UC_Count':len(ucs),
                     'Sem_UC_Hit_Coverage_Percent':len(ucs)/total_uc*100})
    return pd.DataFrame(rows)

def calculate_bootstrap_overlap(perfile_df: pd.DataFrame, n_iter:int=1000) -> pd.DataFrame:
    # probability Model A > Model B
    vals = {mdl: grp['Target_Class_Methods'].values for mdl, grp in perfile_df.groupby('Model')}
    models = list(vals.keys())
    M = len(models)
    mat = pd.DataFrame(np.zeros((M, M)), index=models, columns=models)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if a == b:
                mat.loc[a, b] = np.nan
            else:
                greater = 0
                for _ in range(n_iter):
                    val_a = np.mean(np.random.choice(vals[a], size=len(vals[a]), replace=True))
                    val_b = np.mean(np.random.choice(vals[b], size=len(vals[b]), replace=True))
                    if val_a > val_b:
                        greater += 1
                mat.loc[a, b] = greater / n_iter
    return mat


def calculate_jaccard_global(cache: MetricCache) -> pd.DataFrame:
    # unique methods per model
    mm=defaultdict(set)
    for f,data in cache.json_data.items():
        mdl=cache.file_info[f]['model']
        for c in data.get('classes',[]):
            if c.get('name') in CLASS_NAMES:
                for m in c.get('methods',[]):
                    if m.get('name'): mm[mdl].add(m['name'])
    models=list(mm.keys())
    M=len(models)
    mat=pd.DataFrame(np.zeros((M,M)),index=models,columns=models)
    for i,a in enumerate(models):
        for j,b in enumerate(models):
            if a==b: mat.loc[a,b]=1.0
            else:
                ia, ib = mm[a], mm[b]
                mat.loc[a,b] = len(ia&ib)/len(ia|ib) if ia|ib else 0
    return mat


def calculate_per_class_jaccard(cache: MetricCache) -> dict[str,pd.DataFrame]:
    results={}
    for cls in CLASS_NAMES:
        mm=defaultdict(set)
        for f,data in cache.json_data.items():
            mdl=cache.file_info[f]['model']
            for c in data.get('classes',[]):
                if c.get('name')==cls:
                    for m in c.get('methods',[]):
                        if m.get('name'): mm[mdl].add(m['name'])
        models=list(mm.keys())
        M=len(models)
        mat=pd.DataFrame(np.zeros((M,M)),index=models,columns=models)
        for i,a in enumerate(models):
            for j,b in enumerate(models):
                if a==b: mat.loc[a,b]=1.0
                else:
                    ia, ib = mm[a], mm[b]
                    mat.loc[a,b]=len(ia&ib)/len(ia|ib) if ia|ib else 0
        results[cls]=mat
    return results


def calculate_global_ranking(cache: MetricCache,
                             perfile_df: pd.DataFrame,
                             annotation_df: pd.DataFrame,
                             mapping_df: pd.DataFrame) -> pd.DataFrame:
    
    # 1) pick off each metric and index by Model
    cov_act = calculate_action_combined_coverage(annotation_df, mapping_df, cache.gold) \
                .set_index('Model')['Comb_Action_Coverage_%']
    cov_uc  = calculate_uc_combined_coverage(annotation_df, mapping_df, cache.gold) \
                .set_index('Model')['Comb_UC_Coverage_%']
    sem    = mapping_df.groupby('Model')['SimilarityScore'].mean()
    summary = calculate_method_metrics_summary(perfile_df).set_index('Model')
    red    = summary['Avg_Redundancy']
    pr     = summary['Avg_ParamRichness']
    rtc    = summary['Avg_ReturnTypeCompleteness']
    var    = calculate_variability(perfile_df).set_index('Model')
    cv     = var['CV']
    slope  = var['ConvergenceSlope']
    div    = calculate_diversity(cache).set_index('Model')
    ent    = div['NormalizedEntropy']
    excl   = div['ExclusiveMethodsCount']
    # global Jaccard as average row-similarity (excluding self)
    jac_mat = calculate_jaccard_global(cache)
    jaccard = (jac_mat.sum(axis=1) - 1) / (jac_mat.shape[1] - 1)

    # 2) assemble into one DF
    df = pd.DataFrame({
        'cov_act': cov_act,
        'cov_uc':  cov_uc,
        'sem':     sem,
        'red':     red,
        'pr':      pr,
        'rtc':     rtc,
        'cv':      cv,
        'slope':   slope,
        'ent':     ent,
        'excl':    excl,
        'jaccard': jaccard
    }).dropna()

    # 3) normalization helper
    def norm(series: pd.Series, inv: bool=False) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx==mn:
            return pd.Series(1.0, index=series.index)
        scaled = (series - mn) / (mx - mn)
        return 1 - scaled if inv else scaled

    normed = pd.DataFrame({
        'coverage_act': norm(df['cov_act']),
        'coverage_uc':  norm(df['cov_uc']),
        'semantic':     norm(df['sem']),
        'redundancy':   norm(df['red'], inv=True),
        'paramRich':    norm(df['pr']),
        'returnComp':   norm(df['rtc']),
        'vol_cv':       norm(df['cv'], inv=True),
        'vol_slope':    norm(df['slope']),
        'breadth_ent':  norm(df['ent']),
        'breadth_excl': norm(df['excl']),
        'agreement':    norm(df['jaccard']),
    }, index=df.index)

    # 4) weights summing to 1.0
    weights = {
      'coverage_act': 0.30,
      'coverage_uc':  0.10,   # total 40% coverage
      'semantic':     0.20,   # 20% semantic faithfulness
      'redundancy':   0.075,
      'paramRich':    0.075,
      'returnComp':   0.075,  # 15% quality
      'vol_cv':       0.05,
      'vol_slope':    0.05,   # 10% stability
      'breadth_ent':  0.05,
      'breadth_excl': 0.05,   # 10% breadth
      'agreement':    0.05    # 5% agreement
    }

    # 5) compute weighted sum
    scores = (normed * pd.Series(weights)).sum(axis=1)
    ranking = scores.sort_values(ascending=False).to_frame(name='FinalScore')

    return ranking


# --- Main ---

def main():
    cache = MetricCache(JSON_INPUT_DIR, BASELINE_JSON_FNAME, CLASS_NAMES)
    calculate_counts(cache).to_csv(REPORT_DIR/"Counts_TargetClasses.csv", index=False)
    # Structural preservation
    cache.get_structure_reports_df().to_csv(REPORT_DIR/"StructuralPreservationReport.csv",index=False)
    # Per-file metrics
    perfile = cache.get_per_file_method_metrics()
    perfile.to_csv(REPORT_DIR/"MethodMetrics_PerFile.csv",index=False)
    # Summary metrics
    calculate_method_metrics_summary(perfile).to_csv(REPORT_DIR/"MethodMetrics_Summary.csv",index=False)
    calculate_variability(perfile).to_csv(REPORT_DIR/"VariabilityMetrics.csv",index=False)
    calculate_diversity(cache).to_csv(REPORT_DIR/"DiversityMetrics.csv",index=False)
    # Core & coverage
    core = generate_core_methods_report(cache,TOP_N_CORE)
    core.to_csv(REPORT_DIR/"CoreMethods_TopN.csv",index=False)
    calculate_coverage(cache,core['MethodName'].tolist())\
        .to_csv(REPORT_DIR/"CoverageMetrics.csv",index=False)
    # Annotations
    ann = generate_method_annotation_report(cache)
    ann.to_csv(REPORT_DIR/"Method_Annotation_Details.csv",index=False)
    # Semantic mapping
    mapping = map_methods_to_actions(ann,cache)
    mapping.to_csv(REPORT_DIR/"Method_Action_Mapping_Context.csv",index=False)
    combined_ann_map = ann.merge(
        mapping[['Model','Run','File','Class','MethodName','Best_Match_Action','SimilarityScore']],
        on=['Model','Run','File','Class','MethodName'],
        how='left'
    )
    combined_ann_map.to_csv(
        REPORT_DIR/"Annotation_and_Mapping_Combined.csv",
        index=False
    )
    calculate_action_annotation_coverage(ann,cache).to_csv(
        REPORT_DIR/"Action_Coverage_Annotation.csv",index=False)
    calculate_uc_annotation_coverage(ann,cache).to_csv(
        REPORT_DIR/"UC_Coverage_Annotation.csv",index=False)
    calculate_action_semantic_coverage(mapping,cache).to_csv(
        REPORT_DIR/"Action_Coverage_Semantic.csv",index=False)
    calculate_uc_semantic_coverage(mapping,cache).to_csv(
        REPORT_DIR/"UC_Coverage_Semantic.csv",index=False)
    # Overlap
    calculate_bootstrap_overlap(perfile).to_csv(REPORT_DIR/"BootstrapOverlap.csv")
    calculate_jaccard_global(cache).to_csv(REPORT_DIR/"JaccardMatrix_Global.csv")
    for cls,df in calculate_per_class_jaccard(cache).items():
        df.to_csv(REPORT_DIR/f"JaccardMatrix_{cls}.csv")
    # --- COMBINED STRUCTURE + CLASS COUNTS + PER-FILE METRICS ---
    # 1) get the three pieces
    struct_df = cache.get_structure_reports_df()        # File, preservation columns…
    metric_df = perfile                               # File, Model, Run, Target_Class_Methods, etc.
    counts_df = calculate_counts(cache)               # Class × File table

    counts_wide = (
        counts_df
        .set_index('Class')        # now index=Class, columns are Files
        .T                         # transpose → index=File, columns=Classes
        .reset_index()             # bring File back as a column
        .rename(columns={'index':'File'})
    )

    combined = (
        struct_df
        .merge(counts_wide, on='File', how='left')
        .merge(metric_df, on='File', how='left')
    )

    combined.to_csv(REPORT_DIR/"Combined_Struct_Counts_Metrics.csv", index=False)

    summary = calculate_method_metrics_summary(perfile)
    variability = calculate_variability(perfile)
    diversity = calculate_diversity(cache)
    coverage = calculate_coverage(cache, core['MethodName'].tolist())
    uc_ann = calculate_uc_annotation_coverage(ann, cache)
    uc_sem = calculate_uc_semantic_coverage(mapping, cache)

    # merge them all on 'Model'
    combined_Model = (
        summary
        .merge(variability, on='Model', how='outer')
        .merge(diversity,   on='Model', how='outer')
        .merge(coverage,    on='Model', how='outer')
        .merge(uc_ann,      on='Model', how='outer')
        .merge(uc_sem,      on='Model', how='outer')
    )

    # write out the combined sheet
    combined_Model.to_csv(REPORT_DIR/"Model_Summary_CombinedMetrics.csv", index=False)
    calculate_coverage(cache, core['MethodName'].tolist()).to_csv(REPORT_DIR/"CoverageMetrics.csv", index=False)
  
    ranking_df = calculate_global_ranking(cache, perfile, ann, mapping)
    ranking_df.to_csv(REPORT_DIR/"LLM_Final_Ranking_Weighted.csv")

if __name__=='__main__':
    main()