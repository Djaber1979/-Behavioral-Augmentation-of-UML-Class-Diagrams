import json
from pathlib import Path
import re
import sys
import traceback
from collections import defaultdict

# --- Configuration ---
# Directory containing the RECONSTRUCTED .puml files (output of the first script)
PUML_TO_REFINE_DIR = Path(".") / "puml" # Or "PUML_Reconstructed"

# Directory containing the corresponding .json files (LLM outputs)
JSON_INPUT_DIR = Path(".") / "json"

# Baseline JSON file (essential for determining missing/added elements)
BASELINE_JSON_FNAME = "methodless.json" # Assumed to be in JSON_INPUT_DIR

# Directory to save the REFINED *.puml files
PUML_OUTPUT_DIR = Path(".") / "puml_refined" # New distinct output directory

# Ensure output directory exists
PUML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Target Files for Refinement (Stems from your image) ---
TARGET_FILE_STEMS = [
    "ChatGPT-o3_run4", "ChatGPT-o3_run7", "Claude3.7_run10",
    "Gemini-2.5-Pro-Prreview-03-25_run4", "Gemini-2.5-Pro-Prreview-03-25_run8",
    "Grok3_run10", "Grok3_run3",
    "Llama4_run1", "Llama4_run10", "Llama4_run2", "Llama4_run3",
    "Llama4_run4", "Llama4_run6", "Llama4_run8", "Llama4_run9",
    "Qwen3_run1", "Qwen3_run4", "Qwen3_run6", "Qwen3_run7", "Qwen3_run9"
]

# --- Helper Functions ---
def get_visibility_symbol(vis_string):
    vis = str(vis_string).lower() if vis_string is not None else '+'
    if vis in ["public", "+"]: return "+"
    if vis in ["private", "-"]: return "-"
    if vis in ["protected", "#"]: return "#"
    if vis in ["package", "package-private", "~"]: return "~"
    return "+"

def extract_structure_from_json(json_data):
    """Extracts structural elements into sets from the parsed JSON data for diffing."""
    elements={'classes': set(), 'enums': set(), 'attributes':set(), 'enum_values':set(), 'relationships':set()}
    if not json_data or not isinstance(json_data, dict): return elements

    # Enums and Values
    enums_data = json_data.get('enums', [])
    if isinstance(enums_data, list):
        for enum in enums_data:
            if not isinstance(enum, dict): continue
            name=enum.get('name')
            if name and isinstance(name, str):
                elements['enums'].add(name.strip())
                enum_vals = enum.get('values',[])
                if isinstance(enum_vals, list):
                    for v in enum_vals:
                        if v and isinstance(v, str): elements['enum_values'].add(f"{name.strip()}::{v.strip()}")
    # Classes and Attributes
    classes_data = json_data.get('classes', [])
    if isinstance(classes_data, list):
        for cls in classes_data:
            if not isinstance(cls, dict): continue
            cls_name=cls.get('name')
            if cls_name and isinstance(cls_name, str):
                elements['classes'].add(cls_name.strip())
                attributes_data = cls.get('attributes', [])
                if isinstance(attributes_data, list):
                    for attr in attributes_data:
                        if not isinstance(attr, dict): continue
                        a_name=attr.get('name'); a_type=attr.get('type')
                        if a_name and isinstance(a_name, str) and a_type is not None:
                             norm_type=' '.join(str(a_type).strip().split()) # Normalize spaces in type
                             elements['attributes'].add(f"{cls_name.strip()}::{a_name.strip()}: {norm_type}")
    # Relationships
    relationships_data = json_data.get('relationships', [])
    if isinstance(relationships_data, list):
        for rel in relationships_data:
             if not isinstance(rel, dict): continue
             src_raw=rel.get('source'); tgt_raw=rel.get('target')
             if src_raw and isinstance(src_raw, str) and tgt_raw and isinstance(tgt_raw, str):
                 src = src_raw.strip(); tgt = tgt_raw.strip()
                 # Only consider relationships if classes are part of the model
                 # (This assumes the json is self-consistent)
                 if src in elements['classes'] and tgt in elements['classes']:
                     sym=rel.get('type_symbol','--'); lbl=rel.get('label')
                     s_card=rel.get('source_cardinality'); t_card=rel.get('target_cardinality')
                     lbl_str = f" : {lbl.strip()}" if lbl and isinstance(lbl, str) and lbl.strip() else ""
                     s_card_str = f' "{s_card.strip()}"' if s_card and isinstance(s_card, str) and s_card.strip() else ""
                     t_card_str = f' "{t_card.strip()}"' if t_card and isinstance(t_card, str) and t_card.strip() else ""
                     elements['relationships'].add(f"{src}{s_card_str} {sym}{t_card_str} {tgt}{lbl_str}")
    return elements

def parse_puml_attribute_line(line_content, class_name):
    """Tries to parse an attribute from a PUML line (class context)."""
    match = re.match(r"^\s*[+\-#~]?\s*([a-zA-Z_][\w]*)\s*:\s*([\w<>\[\]\s,\"]+?)\s*(?:' //.*)?$", line_content)
    if match and class_name:
        attr_name = match.group(1).strip()
        attr_type = ' '.join(match.group(2).strip().split()) # Normalize spaces
        return f"{class_name}::{attr_name}: {attr_type}"
    return None

def parse_puml_enum_value_line(line_content, enum_name):
    """Tries to parse an enum value from a PUML line (enum context)."""
    match = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*(?:,)?\s*(?:' //.*)?$", line_content)
    if match and enum_name:
        val_name = match.group(1)
        if val_name.lower() not in ['enum', 'class', '{', '}']: return f"{enum_name}::{val_name}"
    return None

def parse_puml_relationship_line(line_content):
    """Tries to parse a relationship from a PUML line. This is complex."""
    # Regex to capture: ClassA "card" --|> "card" ClassB : label ' // comment
    # It's hard to make this perfect for all PlantUML variations.
    match = re.match(r"^\s*([\w.\"]+)\s*(?:\"(.*?)\")?\s*([.\*]?-{1,2}[>|]?)\s*(?:\"(.*?)\")?\s*([\w.\"]+)\s*(?::\s*(.*?))?\s*(?:' //.*)?$", line_content.strip())
    if match:
        src, s_card, sym, t_card, tgt, lbl = match.groups()
        src = src.strip('"'); tgt = tgt.strip('"')
        lbl_str = f" : {lbl.strip()}" if lbl and lbl.strip() else ""
        s_card_str = f' "{s_card.strip()}"' if s_card and s_card.strip() else ""
        t_card_str = f' "{t_card.strip()}"' if t_card and t_card.strip() else ""
        return f"{src}{s_card_str} {sym}{t_card_str} {tgt}{lbl_str}"
    return None
def parse_puml_line_enum_value(line_content, enum_name):
    """Tries to parse an enum value from a PUML line (enum context)."""
    match = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*(?:,)?\s*(?:' //.*)?$", line_content) # Common enum value format
    if match and enum_name:
        val_name = match.group(1)
        # Avoid matching keywords if they slip in
        if val_name.lower() not in ['enum', 'class', '{', '}']:
            return f"{enum_name}::{val_name}"
    return None
def format_puml_attribute(attr_str, indent="  "):
    """ 'ClassName::AttrName: Type' -> '  + AttrName: Type' (assumes public for new) """
    match = re.match(r"(\w+)::(\w+):\s*(.*)", attr_str)
    if match: return f"{indent}+ {match.group(2)}: {match.group(3)}"
    return None

def format_puml_enum_value(enum_val_str, indent="  "):
    """ 'EnumName::ValueName' -> '  ValueName' """
    parts = enum_val_str.split('::', 1)
    if len(parts) == 2: return f"{indent}{parts[1]}"
    return None

def format_puml_relationship(rel_str):
    """ Assumes rel_str is already in PlantUML form from extract_structure_from_json """
    return rel_str

# --- Main Modification Logic ---
print("Starting PUML Refinement based on JSON diff from baseline...")

# 1. Load Baseline Structure from its JSON
baseline_json_path = JSON_INPUT_DIR / BASELINE_JSON_FNAME
baseline_structure = None
if baseline_json_path.is_file():
    try:
        with open(baseline_json_path, 'r', encoding='utf-8') as f: baseline_json_content = json.load(f)
        baseline_structure = extract_structure_from_json(baseline_json_content)
        if not baseline_structure: raise ValueError("Baseline structure parsing failed.")
        print(f"✔ Parsed baseline structure from: {baseline_json_path}")
    except Exception as e: print(f"❌ Error reading/parsing baseline JSON {baseline_json_path}: {e}"); sys.exit(1)
else: print(f"❌ Error: Baseline JSON file not found at {baseline_json_path}"); sys.exit(1)

baseline_attributes_set = baseline_structure.get('attributes', set())
baseline_enum_values_set = baseline_structure.get('enum_values', set())
baseline_relationships_set = baseline_structure.get('relationships', set())

# --- Process Each Target File ---
files_processed = 0; files_written = 0; files_skipped = 0

for stem in TARGET_FILE_STEMS:
    print(f"\nRefining target: {stem}...")
    files_processed += 1
    json_path = JSON_INPUT_DIR / f"{stem}.json"; puml_input_path = PUML_TO_REFINE_DIR / f"{stem}.puml"
    puml_output_path = PUML_OUTPUT_DIR / f"{stem}.puml"

    if not json_path.is_file(): print(f"  [WARN] JSON not found: '{json_path}'. Skipping."); files_skipped += 1; continue
    if not puml_input_path.is_file(): print(f"  [WARN] Reconstructed PUML not found: '{puml_input_path}'. Skipping."); files_skipped += 1; continue

    try:
        with open(json_path, 'r', encoding='utf-8') as f: json_content = json.load(f)
        generated_structure = extract_structure_from_json(json_content)
        if not generated_structure: raise ValueError("Generated structure parsing failed for JSON.")
    except Exception as e: print(f"  [ERROR] Parsing generated JSON {json_path.name}: {e}. Skipping."); files_skipped += 1; continue

    gen_attributes_set = generated_structure.get('attributes', set())
    gen_enum_values_set = generated_structure.get('enum_values', set())
    gen_relationships_set = generated_structure.get('relationships', set())
    gen_classes_set = generated_structure.get('classes', set())
    gen_enums_set = generated_structure.get('enums', set())

    # Elements from baseline that are NOT in generated JSON (to be omitted from PUML)
    omitted_attributes = baseline_attributes_set - gen_attributes_set
    omitted_enum_values = baseline_enum_values_set - gen_enum_values_set
    omitted_relationships = baseline_relationships_set - gen_relationships_set

    # Elements in generated JSON that are NOT in baseline (to be added to PUML)
    added_attributes = gen_attributes_set - baseline_attributes_set
    added_enum_values = gen_enum_values_set - baseline_enum_values_set
    added_relationships = gen_relationships_set - baseline_relationships_set

    print(f"  To Omit: {len(omitted_attributes)} attr, {len(omitted_enum_values)} enum_val, {len(omitted_relationships)} rel.")
    print(f"  To Add:  {len(added_attributes)} attr, {len(added_enum_values)} enum_val, {len(added_relationships)} rel.")

    output_puml_lines = []
    try:
        with open(puml_input_path, 'r', encoding='utf-8') as f: source_lines = f.readlines()

        in_class_block = False; in_enum_block = False
        current_context_name = None # Stores class or enum name
        skip_current_block_content = False # True if class/enum itself is omitted

        for line_idx, line in enumerate(source_lines):
            stripped_line = line.strip(); original_indent = line[:len(line)-len(line.lstrip(' '))]
            line_to_write = line # Default to keeping the line

            class_match = re.match(r"^\s*(?:abstract\s+|interface\s+)?class\s+([\w.\"]+)", stripped_line)
            enum_match = re.match(r"^\s*enum\s+([\w.\"]+)", stripped_line)

            if class_match and not in_enum_block: # Start of a class
                potential_class_name = class_match.group(1).strip('"')
                if potential_class_name in gen_classes_set: # Class is in JSON, keep it
                    current_context_name = potential_class_name
                    in_class_block = True; in_enum_block = False; skip_current_block_content = False
                    print(f"    Keeping class: {current_context_name}")
                else: # Class not in JSON, mark for skipping
                    skip_current_block_content = True; current_context_name = potential_class_name
                    in_class_block = True; in_enum_block = False
                    print(f"    Omitting class (and its content from PUML): {current_context_name}")
                    line_to_write = None # Omit the class definition line itself
            elif enum_match and not in_class_block: # Start of an enum
                potential_enum_name = enum_match.group(1).strip('"')
                if potential_enum_name in gen_enums_set:
                    current_context_name = potential_enum_name
                    in_enum_block = True; in_class_block = False; skip_current_block_content = False
                    print(f"    Keeping enum: {current_context_name}")
                else:
                    skip_current_block_content = True; current_context_name = potential_enum_name
                    in_enum_block = True; in_class_block = False
                    print(f"    Omitting enum (and its content from PUML): {current_context_name}")
                    line_to_write = None
            elif stripped_line == "}" and (in_class_block or in_enum_block): # End of a block
                if not skip_current_block_content and current_context_name:
                    # Inject ADDED elements for *kept* classes/enums BEFORE closing brace
                    indent_prefix = original_indent + "  "
                    if in_class_block:
                        class_added_attrs = {a for a in added_attributes if a.startswith(f"{current_context_name}::")}
                        if class_added_attrs:
                            output_puml_lines.append(f"{indent_prefix}__ Attributes Added by LLM (JSON only) __\n")
                            for attr_str in sorted(list(class_added_attrs)):
                                formatted_attr = format_puml_attribute(attr_str, indent_prefix)
                                if formatted_attr: output_puml_lines.append(f"{formatted_attr}\n")
                    elif in_enum_block:
                        enum_added_vals = {v for v in added_enum_values if v.startswith(f"{current_context_name}::")}
                        if enum_added_vals:
                            output_puml_lines.append(f"{indent_prefix}__ Values Added by LLM (JSON only) __\n")
                            for enum_val_str in sorted(list(enum_added_vals)):
                                formatted_val = format_puml_enum_value(enum_val_str, indent_prefix)
                                if formatted_val: output_puml_lines.append(f"{formatted_val}\n")
                # The line itself (closing brace) should be kept if the block was kept, omitted if block was omitted
                if skip_current_block_content: line_to_write = None
                in_class_block = False; in_enum_block = False; current_context_name = None; skip_current_block_content = False
            elif skip_current_block_content: # Inside a block to be omitted
                line_to_write = None
            elif in_class_block and current_context_name: # Inside a kept class block
                puml_attr_canonical = parse_puml_attribute_line(stripped_line, current_context_name)
                if puml_attr_canonical and puml_attr_canonical in omitted_attributes:
                    line_to_write = None; print(f"    Omitting attribute from PUML: {puml_attr_canonical}")
                # Methods from reconstructed PUML are kept unless explicitly handled otherwise
            elif in_enum_block and current_context_name: # Inside a kept enum block
                puml_enum_val_canonical = parse_puml_line_enum_value(stripped_line, current_context_name)
                if puml_enum_val_canonical and puml_enum_val_canonical in omitted_enum_values:
                    line_to_write = None; print(f"    Omitting enum value from PUML: {puml_enum_val_canonical}")
            elif not in_class_block and not in_enum_block and not stripped_line.startswith("@startuml") and not stripped_line.startswith("@enduml") and "direction" not in stripped_line.lower() and "skinparam" not in stripped_line.lower() and "package " not in stripped_line.lower():
                puml_rel_canonical = parse_puml_relationship_line(stripped_line)
                if puml_rel_canonical and puml_rel_canonical in omitted_relationships:
                    line_to_write = None; print(f"    Omitting relationship from PUML: {puml_rel_canonical}")

            if line_to_write is not None: output_puml_lines.append(line_to_write)

        # --- Add ADDED Relationships (if any) at the end, before @enduml ---
        final_lines = []
        found_enduml = False
        for line in output_puml_lines:
            if line.strip().startswith("@enduml"):
                if added_relationships:
                    print(f"  Adding {len(added_relationships)} relationships defined only in JSON...")
                    final_lines.append("\n' --- Relationships Added by LLM (JSON only) ---\n")
                    for rel_str in sorted(list(added_relationships)):
                        final_lines.append(f"{rel_str}\n")
                final_lines.append(line)
                found_enduml = True
            else:
                final_lines.append(line)
        if not found_enduml: # Ensure @enduml if it was somehow removed
            if added_relationships and not final_lines[-1].strip().startswith("' --- Relationships Added by LLM"): # Avoid double insertion point
                 print(f"  Adding {len(added_relationships)} relationships defined only in JSON (end of file)...")
                 final_lines.append("\n' --- Relationships Added by LLM (JSON only) ---\n")
                 for rel_str in sorted(list(added_relationships)): final_lines.append(f"{rel_str}\n")
            final_lines.append("@enduml\n")
        output_puml_lines = final_lines

        # 5. Write Modified PUML File
        with open(puml_output_path, 'w', encoding='utf-8') as f: f.writelines(output_puml_lines)
        print(f"  ✔ Saved refined PUML: {puml_output_path}"); files_written += 1
    except Exception as e: print(f"  [ERROR] Processing/writing PUML for {stem}: {e}"); traceback.print_exc(); files_skipped += 1

print(f"\nPUML Refinement Finished. Processed: {files_processed}, Skipped: {files_skipped}, Written: {files_written}")