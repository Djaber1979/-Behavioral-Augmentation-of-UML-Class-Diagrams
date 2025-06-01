import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import traceback # Keep traceback import for error handling

# --- Configuration ---
PUML_INPUT_DIR = Path("puml")
JSON_OUTPUT_DIR = Path("JSON") # Output to a JSON directory in the current directory

# Optional: Keep these setup prints or remove them too
# print(f"Input PUML directory: {PUML_INPUT_DIR.resolve()}")
# print(f"Output JSON directory: {JSON_OUTPUT_DIR.resolve()}")

# --- PlantUML Parsing Regexes ---
# Regex for class definitions (captures optional abstract, declaration line, body)
CLASS_BLOCK_RE = re.compile(
    r'^\s*(abstract\s+)?class\s+' # Group 1: Optional 'abstract ' keyword
    r'(.*?)\s*'                   # Group 2: Everything after class/abstract class up to {
    r'\{([\s\S]*?)\}',            # Group 3: Body content inside {}
    re.MULTILINE
)
# Regex for enum definitions
ENUM_BLOCK_RE = re.compile(r'^\s*enum\s+("?[\w]+"?)\s*\{([\s\S]*?)\}', re.MULTILINE) # Added ^ and MULTILINE

# Regex for relationships
RELATION_RE = re.compile(
    r'^\s*'                                  # Start space
    r'([\w."<>]+)'                           # Group 1: Source
    r'(?:\s+("[^"]+"))?'                     # Group 2: Optional source cardinality/role
    r'\s*([*o<]?[-.]{1,2}[->]+[.]?|[-.]{1,2}[.]?[->]+|[*o<]?[-.]{1,2}|[.]>?)' # Group 3: Arrow/Line Type
    r'(?:\s+("[^"]+"))?'                     # Group 4: Optional target cardinality/role
    r'\s*([\w."<>]+)'                           # Group 5: Target
    r'(?:\s*:\s*(.+))?'                      # Group 6: Optional Label
    r'\s*$',                                 # End space/line
    re.MULTILINE
)
# Regex for package definitions
PACKAGE_RE = re.compile(r'^\s*package\s+([\w."]+)', re.MULTILINE)

# Regex for methods (separates signature body from optional trailing comment)
METHOD_DETAIL_RE = re.compile(
    r'^(?P<signature_body>'
      r'[ \t]*'
      r'(?P<visibility>[+#~-])?'
      r'\s*'
      r'(?P<stereotypes><<\s*[\w,\s]+\s*>>)?'
      r'\s*'
      r'(?P<static>\{static\})?'
      r'\s*'
      r'(?P<abstract>\{abstract\})?'
      r'\s*'
      r'(?P<name>[\w]+)'
      r'\s*'
      r'\((?P<params>.*?)\)'
      r'(?:\s*:\s*(?P<return>[\w<>\[\]\s,]+))?'
      r'(?P<suffix>.*?)??'
    r')'
    r'\s*'
    r'(?P<comment>//.*)?'
    r'\s*$'
)

# Regex for parsing UC annotations within comments (format: //UCx //action:Text)
UC_COMMENT_RE = re.compile(
    r'//\s*UC(?P<uc_ids>\d+(?:[\d,/]+)*)'
    r'(?:\s*//\s*action:\s*(?P<action>.+))?'
    r'\s*$'
)

# Regex for attributes
ATTRIBUTE_RE = re.compile(
    r'^[ \t]*'
    r'(?!(?:class|enum|interface)\s)' # Avoid matching class/enum keywords
    r'(?P<visibility>[+#~-])?'
    r'\s*'
    r'(?P<stereotypes><<\s*[\w,\s]+\s*>>)?'
    r'\s*'
    r'(?P<static>\{static\})?'
    r'\s*'
    r'(?P<modifier>\{readOnly\}|final)?'
    r'\s*'
    r'(?P<name>[\w_]+)'
    r'\s*:\s*'
    r'(?P<type>[\w<>\[\]\s,.*]+?)' # Non-greedy type
    r'(?:\s*=\s*(?P<default>.+?))?' # Optional default value
    r'\s*$'
)

# Regex for parameters within method parentheses
PARAMETER_RE = re.compile(r'([\w<>\[\]]+)\s*:\s*([\w<>\[\]\s,]+)')

# --- Helper Functions ---

def parse_parameters(param_string):
    """ Parses parameter string into a list of {'name': name, 'type': type} dicts """
    params = []
    # Split by comma first to handle multiple params
    raw_params = [p.strip() for p in param_string.split(',') if p.strip()]
    for part in raw_params:
        match = PARAMETER_RE.match(part) # Match "name : Type"
        if match:
            name, type_str = match.groups()
            # Normalize spaces in type string
            type_str_normalized = ' '.join(type_str.strip().split())
            params.append({'name': name.strip(), 'type': type_str_normalized})
        elif ':' not in part and part: # Handle type-only parameters?
             params.append({'name': None, 'type': ' '.join(part.strip().split())})
    return params

def parse_method(line):
    """ Parses a method line, separating signature from annotation and normalizing spacing. """
    match = METHOD_DETAIL_RE.match(line.strip())
    if not match: return None
    details = match.groupdict()

    # --- Annotation Extraction from Comment ---
    annotation_data = None
    comment_content = details.get('comment')
    if comment_content:
        uc_match = UC_COMMENT_RE.match(comment_content.strip())
        if uc_match:
            uc_details = uc_match.groupdict()
            uc_ids_str = uc_details.get('uc_ids')
            action_str = uc_details.get('action')
            if uc_ids_str:
                uc_references = [digit for part in re.split('[,/]', uc_ids_str) for digit in re.findall(r'\d+', part)]
                if uc_references:
                     annotation_data = {
                         "uc_references": uc_references,
                         "uc_action": ' '.join(action_str.strip().split()) if action_str is not None else ""
                     }

    # --- Extract & Normalize Core Components ---
    visibility = details.get('visibility') if details.get('visibility') is not None else '+'
    stereotypes = details.get('stereotypes')
    is_static = bool(details.get('static'))
    is_abstract = bool(details.get('abstract'))
    name = details.get('name','').strip()
    params_str = details.get('params','')
    return_type = details.get('return','').strip() if details.get('return') is not None else 'void'
    suffix = details.get('suffix','').strip() if details.get('suffix') else None
    parameters = parse_parameters(params_str)
    norm_return_type = ' '.join(return_type.split())

    # Determine cleaned suffix (exclude annotation part)
    cleaned_suffix = None
    if suffix:
        suffix_norm = ' '.join(suffix.split())
        if not comment_content or not comment_content.strip().startswith(suffix_norm):
             cleaned_suffix = suffix_norm

    # --- Reconstruct Cleaned Signature String ---
    signature_parts = []
    if visibility: signature_parts.append(visibility)
    if stereotypes: signature_parts.append(stereotypes)
    if is_static: signature_parts.append('{static}')
    if is_abstract: signature_parts.append('{abstract}')
    if name: signature_parts.append(name)
    param_strings = [f"{p['name']} : {p.get('type','')}" if p.get('name') else p.get('type','') for p in parameters if p.get('type')]
    signature_parts.append(f"({', '.join(param_strings)})")
    if norm_return_type and norm_return_type.lower() != 'void': signature_parts.append(f": {norm_return_type}")
    if cleaned_suffix: signature_parts.append(cleaned_suffix)
    cleaned_signature = ' '.join(filter(None, signature_parts)) # Filter None before join

    # --- Construct Final JSON ---
    method_json = {
        "signature": cleaned_signature, "visibility": visibility,
        "stereotypes": stereotypes, "is_static": is_static, "is_abstract": is_abstract,
        "name": name, "parameters": parameters, "return_type": norm_return_type,
        "suffix": cleaned_suffix,
    }
    if annotation_data: method_json["annotation"] = annotation_data
    return method_json

def parse_attribute(line):
    """ Parses an attribute line and normalizes spacing in string fields. """
    match = ATTRIBUTE_RE.match(line.strip())
    if not match: return None
    details = match.groupdict()
    visibility = details.get('visibility') if details.get('visibility') else '~'
    stereotypes = details.get('stereotypes')
    is_static = bool(details.get('static'))
    modifier = details.get('modifier')
    name = details.get('name','').strip()
    type_str = details.get('type','').strip()
    default_value = details.get('default','').strip() if details.get('default') else None
    norm_type = ' '.join(type_str.split())
    norm_default = ' '.join(default_value.split()) if default_value else None

    signature_parts = []
    if visibility != '~': signature_parts.append(visibility) # Only show non-default visibility
    if stereotypes: signature_parts.append(stereotypes)
    if is_static: signature_parts.append('{static}')
    if modifier: signature_parts.append(modifier)
    if name: signature_parts.append(name)
    if norm_type: signature_parts.append(f": {norm_type}")
    if norm_default: signature_parts.append(f"= {norm_default}")
    cleaned_signature = ' '.join(filter(None, signature_parts))

    return {
        "signature": cleaned_signature, "visibility": visibility,
        "stereotypes": stereotypes, "is_static": is_static, "modifier": modifier,
        "name": name, "type": norm_type, "default_value": norm_default
    }

def parse_puml_to_structure(puml_content):
    """ Parses PUML text content into a structured dictionary """
    data = { "packages": [], "enums": [], "classes": [], "relationships": [] }

    # Extract packages
    try: data["packages"] = sorted(list(set(pkg.strip('"') for pkg in PACKAGE_RE.findall(puml_content))))
    except Exception as e: print(f"  [ERROR] Package extraction failed: {e}")

    # Extract enums
    try:
        for enum_match in ENUM_BLOCK_RE.finditer(puml_content):
            try:
                name = enum_match.group(1).strip('"'); body = enum_match.group(2)
                values = [val.strip() for val in body.splitlines() if val.strip() and not val.strip().startswith(("//", "'"))]
                data["enums"].append({"name": name, "values": sorted(values)})
            except Exception as e: print(f"  [ERROR] Processing enum '{name if 'name' in locals() else 'Unknown'}': {e}")
        data["enums"].sort(key=lambda x: x['name']) # Sort enums by name
    except Exception as e: print(f"  [ERROR] Enum processing loop failed: {e}")

    # Extract classes and their members
    class_count = 0
    for class_match in CLASS_BLOCK_RE.finditer(puml_content):
        class_count += 1
        final_class_name = f"UNKNOWN_CLASS_{class_count}"; is_abstract = False # Defaults
        try:
            abstract_keyword = class_match.group(1); declaration_line = class_match.group(2).strip(); body = class_match.group(3)
            if not declaration_line or body is None: print(f"[WARN] Skipping class match {class_count}: Invalid capture."); continue
            is_abstract = bool(abstract_keyword)
            alias_name = None; extends = None; implements_str = None; stereotype = None
            alias_match = re.search(r'(.*?)\s+as\s+(\w+)\s*$', declaration_line);
            if alias_match: alias_name = alias_match.group(2).strip(); declaration_line = alias_match.group(1).strip()
            implements_match = re.search(r'(.*?)\s+implements\s+([\w<>, ]+)\s*$', declaration_line);
            if implements_match: implements_str = implements_match.group(2).strip(); declaration_line = implements_match.group(1).strip()
            extends_match = re.search(r'(.*?)\s+extends\s+([\w<>]+)\s*$', declaration_line);
            if extends_match: extends = extends_match.group(2).strip(); declaration_line = extends_match.group(1).strip()
            stereotype_match = re.search(r'(.*?)\s*(<<\s*[\w,\s]+\s*>>)\s*$', declaration_line);
            if stereotype_match: stereotype = stereotype_match.group(2).strip(); declaration_line = stereotype_match.group(1).strip()
            declaration_line = re.sub(r'^(?:abstract|final)\s+', '', declaration_line).strip()
            class_name_parsed = declaration_line.strip().strip('"')
            final_class_name = alias_name if alias_name else class_name_parsed
            if not final_class_name: print(f"[WARN] Skipping class match {class_count}: Could not determine name."); continue

            class_data = {"name": final_class_name, "is_abstract": is_abstract, "extends": extends, "implements": sorted([i.strip() for i in implements_str.split(',')]) if implements_str else [], "attributes": [], "methods": []}
            for line in body.splitlines():
                line_stripped = line.strip();
                if not line_stripped or line_stripped.startswith(("//", "'")): continue
                method_data = parse_method(line_stripped)
                if method_data: class_data["methods"].append(method_data); continue
                attribute_data = parse_attribute(line_stripped)
                if attribute_data: class_data["attributes"].append(attribute_data); continue
            # Sort attributes and methods for consistent output
            class_data["attributes"].sort(key=lambda x: x['name'])
            class_data["methods"].sort(key=lambda x: x['name'])
            data["classes"].append(class_data)
        except Exception as e: print(f"[ERROR] Processing class '{final_class_name}': {e}"); traceback.print_exc()
    data["classes"].sort(key=lambda x: x['name']) # Sort classes by name

    # Extract relationships
    try:
        temp_relationships = []
        for rel_match in RELATION_RE.finditer(puml_content):
            try:
                groups = rel_match.groups()
                source = groups[0].strip('"') if groups[0] else None; target = groups[4].strip('"') if groups[4] else None
                if source and target:
                     temp_relationships.append({"source": source, "source_cardinality": groups[1].strip('"') if groups[1] else None, "type_symbol": groups[2].strip() if groups[2] else None, "target_cardinality": groups[3].strip('"') if groups[3] else None, "target": target, "label": groups[5].strip() if groups[5] else None})
            except IndexError: print(f"  [ERROR] Parsing relationship groups: {rel_match}")
            except Exception as e: print(f"    [ERROR] Processing relationship match: {e}")
        # Sort relationships for consistency
        data["relationships"] = sorted(temp_relationships, key=lambda x: (x['source'], x['target'], x.get('label','')))
    except Exception as e: print(f"  [ERROR] Relationship processing loop failed: {e}")

    return data

# --- Main Conversion Logic ---
def convert_puml_to_json():
    """ Finds PUML files, parses them, and saves as JSON. """
    if not PUML_INPUT_DIR.is_dir(): print(f"Error: Input dir not found: {PUML_INPUT_DIR.resolve()}"); sys.exit(1)
    all_puml_files = list(PUML_INPUT_DIR.glob("*.puml"))
    # Define baseline here if needed, or assume it's in PUML_INPUT_DIR
    baseline_puml_path = Path("methodless.puml")
    if baseline_puml_path.is_file() and baseline_puml_path not in all_puml_files:
         all_puml_files.append(baseline_puml_path)
    puml_files = sorted(all_puml_files, key=lambda p: p.name)
    if not puml_files: print("No .puml files found."); return
    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converted = 0; errors = 0
    print(f"Starting conversion for {len(puml_files)} files...")
    for puml_path in puml_files:
        try:
            content = puml_path.read_text(encoding="utf-8")
            structured_data = parse_puml_to_structure(content)
            json_path = JSON_OUTPUT_DIR / (puml_path.stem + ".json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False) # Use indent=2 for smaller files
            converted += 1
        except Exception as e:
            print(f"  [ERROR] Failed processing {puml_path.name}: {e}"); traceback.print_exc(); errors += 1
    print("-" * 20 + "\nConversion Summary:")
    print(f"  Successfully converted: {converted}"); print(f"  Errors encountered: {errors}")
    print("-" * 20)

if __name__ == "__main__":
    convert_puml_to_json()