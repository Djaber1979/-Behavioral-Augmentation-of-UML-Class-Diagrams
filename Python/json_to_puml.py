import json
from pathlib import Path
import re
import sys
import traceback # Keep this import

# --- Configuration ---
JSON_INPUT_DIR = Path(".") / "json"
BASELINE_PUML_FNAME = "baseline.puml"
PUML_OUTPUT_DIR = Path(".") / "puml"
PUML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---

def format_parameters(params_list):
    if not params_list or not isinstance(params_list, list): return ""
    parts = []
    for p in params_list:
        if not isinstance(p, dict): continue
        p_name = p.get("name", "?"); p_type = p.get("type", "Unknown")
        p_type_str = str(p_type); p_type_clean = f'"{p_type_str}"' if ' ' in p_type_str else p_type_str
        parts.append(f"{p_name}: {p_type_clean}")
    return ", ".join(parts)

def get_visibility_symbol(vis_string):
    vis = str(vis_string).lower() if vis_string is not None else '+'
    if vis in ["public", "+"]: return "+"
    if vis in ["private", "-"]: return "-"
    if vis in ["protected", "#"]: return "#"
    if vis in ["package", "package-private", "~"]: return "~"
    return "+"

def parse_json_structure(json_data):
    structure = {"classes": {}, "relationships": [], "packages": []}
    if not json_data or not isinstance(json_data, dict): return structure
    structure["packages"] = json_data.get('packages', []);
    if not isinstance(structure["packages"], list): structure["packages"] = []
    classes_data = json_data.get('classes', []);
    if not isinstance(classes_data, list): classes_data = []
    for cls_data in classes_data:
        if not isinstance(cls_data, dict): continue
        class_name = cls_data.get('name')
        if not class_name or not isinstance(class_name, str): continue
        methods = []; methods_data = cls_data.get('methods', [])
        if not isinstance(methods_data, list): methods_data = []
        for m_info in methods_data:
            if not isinstance(m_info, dict): continue
            method_name = m_info.get("name")
            if not method_name or not isinstance(method_name, str): continue
            params = m_info.get("parameters", []);
            if not isinstance(params, list): params = []
            ret_type = m_info.get("return_type", "void")
            visibility = get_visibility_symbol(m_info.get("visibility"))
            annotation = m_info.get("annotation", {});
            if not isinstance(annotation, dict): annotation = {}
            ucs = annotation.get("uc_references", [])
            action = annotation.get("uc_action", "")
            if isinstance(ucs, list): ucs = [str(u) for u in ucs if u]
            elif ucs: ucs = [str(ucs)]
            else: ucs = []
            if not isinstance(action, str): action = ""
            methods.append({"name": method_name.strip(), "visibility": visibility, "parameters": params, "return_type": str(ret_type).strip(), "ucs": ucs, "action": action.strip()})
        structure["classes"][class_name] = {"methods": methods}
    return structure

# --- Main Reconstruction Logic ---

print("Starting PUML Reconstruction (Baseline Classes Only)...")

# 1. Read Baseline PUML Template
baseline_puml_path = Path(BASELINE_PUML_FNAME)
baseline_lines = []
if baseline_puml_path.is_file():
    try:
        with open(baseline_puml_path, 'r', encoding='utf-8') as f: baseline_lines = f.readlines()
        print(f"✔ Read baseline template: {baseline_puml_path}")
    except Exception as e: print(f"❌ Error reading baseline PUML {baseline_puml_path}: {e}"); sys.exit(1)
else: print(f"❌ Error: Baseline PUML file not found at {baseline_puml_path}"); sys.exit(1)

# 2. Iterate through JSON files
baseline_json_to_exclude = "methodless.json" # Use the specific name of baseline JSON
json_files = sorted([p for p in JSON_INPUT_DIR.glob("*.json") if p.name != baseline_json_to_exclude])
print(f"Found {len(json_files)} JSON files in '{JSON_INPUT_DIR}' to process.")
if not json_files: print("No JSON files found to process. Exiting."); sys.exit(0)

# --- Process Each JSON File ---
for json_path in json_files:
    print(f"\nProcessing: {json_path.name}...")
    puml_filename = json_path.stem + ".puml"
    puml_output_path = PUML_OUTPUT_DIR / puml_filename

    # 3. Parse JSON structure
    try:
        with open(json_path, 'r', encoding='utf-8') as f: json_content = json.load(f)
        llm_structure = parse_json_structure(json_content)
        if not llm_structure["classes"]: print(f"  [WARN] No classes found/parsed in JSON {json_path.name}. Skipping."); continue
    except Exception as e: print(f"  [ERROR] Failed to parse JSON {json_path.name}: {e}. Skipping."); traceback.print_exc(); continue

    # 4. Generate PUML Content
    output_puml_lines = []
    in_class_block = False
    current_class_name = None

    for line in baseline_lines:
        stripped_line = line.strip()
        original_indent = line[:len(line)-len(line.lstrip(' '))]

        class_match = re.match(r"^\s*(?:abstract\s+|interface\s+)?class\s+([\w.\"]+)", stripped_line)
        if class_match and not in_class_block:
            current_class_name = class_match.group(1).strip('"')
            in_class_block = True
            output_puml_lines.append(line)
            continue

        if in_class_block and stripped_line == "}":
            if current_class_name in llm_structure["classes"]:
                class_data = llm_structure["classes"][current_class_name]
                if class_data.get("methods"):
                    output_puml_lines.append(f"{original_indent}  \n")
                    for method in class_data["methods"]:
                        params_str = format_parameters(method["parameters"])
                        ret_type_str = method.get("return_type", "void")
                        method_name_str = method.get("name", "unknown")
                        visibility_str = method.get("visibility", "+")
                        method_signature = f"{visibility_str} {method_name_str}({params_str}): {ret_type_str}"
                        uc_comment = ""; action_comment = ""

                        # --- MODIFICATION: Ensure "UC" Prefix ---
                        if method.get("ucs"):
                            formatted_ucs = []
                            for uc in method['ucs']:
                                uc_str = str(uc).strip() # Ensure string and strip spaces
                                if uc_str and uc_str.isdigit(): # If it's only digits
                                    formatted_ucs.append(f"UC{uc_str}")
                                elif uc_str and not uc_str.upper().startswith("UC"): # If not starting with UC (case-insensitive)
                                     formatted_ucs.append(f"UC{uc_str}") # Prepend UC
                                elif uc_str: # Already has UC prefix or is non-numeric
                                     formatted_ucs.append(uc_str)

                            if formatted_ucs:
                                 uc_comment = f" // {', '.join(sorted(formatted_ucs))}"
                        # --- END MODIFICATION ---

                        action_text = method.get("action", "")
                        if action_text: action_comment = f" // action: {action_text}"
                        annotation_part = f"{uc_comment}{action_comment}"
                        full_method_line = f"{original_indent}  {method_signature}{annotation_part}\n"
                        output_puml_lines.append(full_method_line)

            output_puml_lines.append(line) # Add closing brace
            in_class_block = False
            current_class_name = None
            continue

        if in_class_block: output_puml_lines.append(line) # Copy content within baseline classes
        else:
            if not stripped_line.startswith("@enduml"): output_puml_lines.append(line) # Copy content outside classes

    # --- REMOVED Section for adding JSON-only classes ---

    # Ensure @enduml is the last line
    if not output_puml_lines or not output_puml_lines[-1].strip().startswith("@enduml"):
        output_puml_lines.append("@enduml\n")

    # 5. Write PUML File
    try:
        with open(puml_output_path, 'w', encoding='utf-8') as f: f.writelines(output_puml_lines)
        print(f"  ✔ Saved reconstructed PUML: {puml_output_path}")
    except Exception as e: print(f"  [ERROR] Failed to write PUML file {puml_output_path}: {e}")

print("\nPUML Reconstruction Finished.")

if __name__ == "__main__":
     # This part is usually for running the script directly,
     # but since the logic is self-contained above, it might not be needed
     # unless you want to add argument parsing etc. later.
     pass