import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Assumes this script is in the same directory as the main results folder
# and the JSON folder exists within results
JSON_INPUT_DIR = Path("JSON")      # Directory containing the *.json files
OUTPUT_DIR = Path("analysis_results") # Directory for saving analysis results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Function (copied from main.py for consistency) ---
def model_key(filename_or_stem):
    """Extracts the model identifier by removing the '_run<N>' suffix."""
    if isinstance(filename_or_stem, Path): stem = filename_or_stem.stem
    elif isinstance(filename_or_stem, str): stem = Path(filename_or_stem).stem
    else: return str(filename_or_stem)
    model_name = re.sub(r"_run\d+$", "", stem)
    return model_name

# --- Main Analysis Functions ---

def load_and_extract_methods(json_dir: Path, baseline_fname: str) -> dict[str, list[str]]:
    """Loads JSON files and extracts method names, grouped by model."""
    print(f"Loading JSON data from: {json_dir.resolve()}")
    model_method_lists = defaultdict(list) # { model_name: [ [methods_run1], [methods_run2], ... ] }
    model_run_filenames = defaultdict(list) # { model_name: [ fname_run1, fname_run2, ... ] }

    json_files = sorted(
        [p for p in json_dir.glob("*.json") if p.name != baseline_fname],
        key=lambda p: p.name
    )

    if not json_files:
        print("Error: No generated JSON files found (excluding baseline).")
        return {}, {}

    print(f"Found {len(json_files)} generated JSON files.")
    loaded_count = 0
    for f_path in json_files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model = model_key(f_path.stem)
            methods_in_file = [
                method.get('name')
                for cls in data.get('classes', [])
                for method in cls.get('methods', [])
                if method.get('name') # Ensure method has a name
            ]
            model_method_lists[model].append(methods_in_file)
            model_run_filenames[model].append(f_path.name) # Store filename for reference
            loaded_count += 1
        except Exception as e:
            print(f"  [Warning] Failed to load or parse {f_path.name}: {e}")

    print(f"Successfully loaded method data for {loaded_count} files from {len(model_method_lists)} models.")
    return model_method_lists, model_run_filenames

def calculate_cosine_similarity(model_method_lists: dict[str, list[list[str]]],
                                model_run_filenames: dict[str, list[str]]) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Calculates the cosine similarity matrix based on TF-IDF of method names."""
    if not model_method_lists:
        return None, [], []

    # Create flat lists of documents (joined method names) and labels (model+run)
    all_docs = []
    all_doc_labels = [] # e.g., "ModelA_run1"
    all_model_labels = [] # e.g., "ModelA"

    print("\nPreparing documents for TF-IDF...")
    for model, runs_methods_lists in model_method_lists.items():
        for i, method_list in enumerate(runs_methods_lists):
            # Use the stored filename to ensure correct run association if needed
            # For now, assume order is preserved
            run_label = f"{model}_run{i+1}" # Simple label, replace with actual run if needed
            # Try getting actual run filename
            try:
                 actual_fname = model_run_filenames[model][i]
                 run_label = Path(actual_fname).stem # Use filename stem as precise label
            except IndexError:
                 print(f"Warning: Could not find filename for {model} run index {i}")

            doc = " ".join(method_list) # Create document string
            all_docs.append(doc)
            all_doc_labels.append(run_label)
            all_model_labels.append(model)

    if not all_docs:
        print("Error: No documents created (no methods found?).")
        return None, [], []

    print(f"Vectorizing {len(all_docs)} documents using TF-IDF...")
    vectorizer = TfidfVectorizer(lowercase=False) # Keep case sensitivity for method names
    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        print(f"  TF-IDF Matrix shape: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"  [ERROR] TF-IDF Vectorization failed: {e}")
        return None, [], []


    print("Calculating cosine similarity matrix...")
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    return cosine_sim_matrix, all_doc_labels, all_model_labels

def analyze_similarity(sim_matrix: np.ndarray, doc_labels: list[str], model_labels: list[str], output_dir: Path):
    """Calculates and saves intra-LLM and inter-LLM similarity."""
    if sim_matrix is None or not doc_labels or not model_labels:
        print("Skipping similarity analysis due to missing data.")
        return

    n_docs = sim_matrix.shape[0]
    models = sorted(list(set(model_labels)))
    print(f"\nAnalyzing similarity for {len(models)} models...")

    # --- Intra-LLM Similarity ---
    intra_scores = {}
    processed_indices = 0
    # Group indices by model
    model_indices = defaultdict(list)
    for i, model in enumerate(model_labels):
        model_indices[model].append(i)

    for model in models:
        indices = model_indices[model]
        if len(indices) < 2:
            intra_scores[model] = np.nan # Cannot calculate similarity for single run
            continue
        # Extract submatrix for this model
        model_sim_submatrix = sim_matrix[np.ix_(indices, indices)]
        # Get upper triangle indices (excluding diagonal k=1)
        iu = np.triu_indices_from(model_sim_submatrix, k=1)
        if iu[0].size > 0:
            avg_sim = np.mean(model_sim_submatrix[iu])
            intra_scores[model] = avg_sim
        else: # Should not happen if len(indices) >= 2
            intra_scores[model] = np.nan

    intra_df = pd.DataFrame(list(intra_scores.items()), columns=['Model', 'Avg_Intra_LLM_Similarity'])
    intra_df.sort_values(by='Avg_Intra_LLM_Similarity', ascending=False, inplace=True)
    print("\nAverage Intra-LLM Cosine Similarity (Method Names TF-IDF):")
    print(intra_df.round(3).to_string(index=False))
    intra_path = output_dir / "similarity_intra_llm.csv"
    intra_df.round(3).to_csv(intra_path, index=False)
    print(f"Saved: {intra_path}")

    # --- Inter-LLM Similarity ---
    inter_scores = pd.DataFrame(np.nan, index=models, columns=models)
    for model_a in models:
        for model_b in models:
            if model_a == model_b: continue # Skip self-comparison
            indices_a = model_indices[model_a]
            indices_b = model_indices[model_b]
            if not indices_a or not indices_b: continue # Skip if one model has no runs

            # Extract the cross-similarity block
            cross_sim_submatrix = sim_matrix[np.ix_(indices_a, indices_b)]
            avg_sim = np.mean(cross_sim_submatrix) # Average all pairwise similarities
            inter_scores.loc[model_a, model_b] = avg_sim

    print("\nAverage Inter-LLM Cosine Similarity Matrix (Method Names TF-IDF):")
    print(inter_scores.round(3))
    inter_path = output_dir / "similarity_inter_llm_matrix.csv"
    inter_scores.round(3).to_csv(inter_path)
    print(f"Saved: {inter_path}")

    # --- Heatmap ---
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(inter_scores.astype(float), annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title('Average Inter-LLM Cosine Similarity (Method Names TF-IDF)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_path = output_dir / "similarity_inter_llm_heatmap.png"
        plt.savefig(heatmap_path)
        print(f"Saved: {heatmap_path}")
        plt.close()
    except Exception as e:
        print(f"  [Warning] Could not generate heatmap: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Cosine Similarity Analysis ---")

    # Assuming baseline JSON is named like this and present in JSON_INPUT_DIR
    baseline_json_fname = "methodless.json"

    # 1. Load data and extract method names
    model_methods, model_filenames = load_and_extract_methods(JSON_INPUT_DIR, baseline_json_fname)

    # 2. Calculate Similarity
    if model_methods:
        cosine_matrix, doc_labels, model_labels = calculate_cosine_similarity(model_methods, model_filenames)

        # 3. Analyze and Save Results
        analyze_similarity(cosine_matrix, doc_labels, model_labels, OUTPUT_DIR)
    else:
        print("Aborting analysis as no method data was loaded.")

    print("\n--- Cosine Similarity Analysis Finished ---")