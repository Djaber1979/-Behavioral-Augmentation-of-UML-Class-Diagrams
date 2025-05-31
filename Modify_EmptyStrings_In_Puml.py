#!/usr/bin/env python3
"""
Modify_EmptyStrings_In_Puml.py

Searches for all occurrences of "" (empty string literal) within .puml files
in a specified directory (and its subdirectories, if any) and replaces
them with a single asterisk character '*'.

Excludes specified files like 'methodless.puml'.
"""
import os
from pathlib import Path

# --- Configuration ---
# Directory containing the .puml files to process.
# If the script is in the root of the project and .puml files are in a 'puml' subdirectory:
# PUML_FILES_DIR = Path(__file__).resolve().parent / "puml"
# If the script is in the SAME directory as the .puml files:
SCRIPT_PARENT_DIR = Path(__file__).resolve().parent
PUML_FILES_DIR = SCRIPT_PARENT_DIR / "puml"
# Files to exclude from processing
FILES_TO_EXCLUDE = {"methodless.puml"}

# The string to search for
SEARCH_STRING = '"1" --> *'

# The string to replace with
# Scenario 1: Replace "" with "*" (asterisk inside quotes)
# REPLACE_STRING = '"*"'
# Scenario 2: Replace "" with * (single asterisk character)
REPLACE_STRING = '"1" --> "*"' # Using this based on common placeholder usage

def modify_puml_file(file_path: Path, search_str: str, replace_str: str) -> bool:
    """
    Reads a .puml file, replaces all occurrences of search_str with replace_str,
    and writes the content back to the file.
    Returns True if modifications were made, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if search_str in content:
            modified_content = content.replace(search_str, replace_str)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Modified: {file_path.name}")
            return True
        else:
            # print(f"No changes needed for: {file_path.name}")
            return False
    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return False

def main():
    print(f"Starting modification of .puml files in: {PUML_FILES_DIR}")
    print(f"Searching for: '{SEARCH_STRING}'")
    print(f"Replacing with: '{REPLACE_STRING}'")
    print(f"Excluding files: {FILES_TO_EXCLUDE}\n")

    if not PUML_FILES_DIR.is_dir():
        print(f"Error: Directory not found: {PUML_FILES_DIR}")
        return

    modified_files_count = 0
    processed_files_count = 0

    # Iterate through all .puml files in the directory (not recursive by default with glob)
    # If you need recursive search, use .rglob("*.puml")
    for puml_file_path in PUML_FILES_DIR.glob("*.puml"):
        if puml_file_path.name in FILES_TO_EXCLUDE:
            print(f"Skipping excluded file: {puml_file_path.name}")
            continue

        processed_files_count += 1
        if modify_puml_file(puml_file_path, SEARCH_STRING, REPLACE_STRING):
            modified_files_count += 1
    
    print(f"\nFinished processing.")
    print(f"Total .puml files scanned (excluding ignored): {processed_files_count}")
    print(f"Files modified: {modified_files_count}")

if __name__ == "__main__":
    main()