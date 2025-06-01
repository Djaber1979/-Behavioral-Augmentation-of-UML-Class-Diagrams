#!/usr/bin/env python3
"""
RQ_Visualize_Existing_Jaccard_Matrix.py

Generates:
1.  A heatmap visualizing an existing Jaccard similarity matrix.
2.  A dendrogram showing hierarchical clustering of LLMs based on the existing similarity matrix.

Assumes the input Jaccard matrix represents similarity (higher is more similar).
For the dendrogram, similarity will be converted to distance (1 - similarity).

Reads from: reports/JaccardMatrix_Global.csv (or user-specified)
Saves outputs to: reports/article/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform # For converting square distance matrix to condensed form

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "JaccardMatrix_Global.csv" # The pre-calculated Jaccard matrix
ARTICLE_DIR = REPORTS_DIR / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

def load_jaccard_matrix(filepath: Path) -> pd.DataFrame | None:
    """Loads the pre-calculated Jaccard similarity matrix."""
    if not filepath.is_file():
        print(f"Error: Input Jaccard matrix CSV file not found at {filepath}")
        return None
    try:
        # The first column in your example CSV is the index
        df = pd.read_csv(filepath, index_col=0)
        print(f"Successfully loaded Jaccard matrix from {filepath}")
        if df.empty or not (df.index == df.columns).all():
            print("Warning: Loaded matrix is empty or index and columns do not match.")
            # return None # Allow processing even if not perfectly square for some checks
        return df
    except Exception as e:
        print(f"Error loading Jaccard matrix CSV from {filepath}: {e}")
        return None

def generate_heatmap_plot(df_matrix: pd.DataFrame, output_path: Path, title: str):
    """Generates and saves a heatmap plot from a similarity matrix."""
    if df_matrix.empty:
        print(f"Warning: Matrix for heatmap '{title}' is empty. Skipping plot.")
        return

    plt.figure(figsize=(max(8, len(df_matrix.columns) * 0.9), max(6, len(df_matrix.index) * 0.7)))
    sns.heatmap(df_matrix, annot=True, fmt=".2f", cmap="viridis_r", linewidths=.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Jaccard Similarity (Raw Names)'}) # Updated label
    plt.title(title, fontsize=14)
    plt.xlabel("LLM Model", fontsize=12)
    plt.ylabel("LLM Model", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    plt.close()

def generate_dendrogram_plot(similarity_matrix: pd.DataFrame, output_path: Path, title: str, linkage_method='average'):
    """
    Generates and saves a dendrogram plot from a Jaccard similarity matrix.
    """
    if similarity_matrix.empty:
        print(f"Warning: Similarity matrix for dendrogram '{title}' is empty. Skipping plot.")
        return
    if len(similarity_matrix) < 2:
        print(f"Warning: Need at least 2 items for clustering. Similarity matrix has {len(similarity_matrix)} items. Skipping dendrogram.")
        return

    print(f"Generating dendrogram using '{linkage_method}' linkage...")
    
    # Convert similarity matrix to distance matrix: distance = 1 - similarity
    distance_matrix_df = 1 - similarity_matrix
    
    # Ensure the distance matrix is symmetric and has zeros on the diagonal
    # (it should if the similarity matrix was correctly representing Jaccard)
    np.fill_diagonal(distance_matrix_df.values, 0) 
    # Ensure symmetry, though Jaccard should be symmetric
    distance_matrix_df = (distance_matrix_df + distance_matrix_df.T) / 2

    # Check for NaNs or Infs that might cause issues
    if distance_matrix_df.isnull().values.any() or np.isinf(distance_matrix_df.values).any():
        print("Warning: Distance matrix contains NaNs or Infs. Attempting to fill with a large distance (2.0).")
        # A Jaccard distance is between 0 and 1. A value of 2.0 is distinctly large.
        distance_matrix_df = distance_matrix_df.fillna(1.0).replace([np.inf, -np.inf], 1.0) # Fill with max possible Jaccard distance
        np.fill_diagonal(distance_matrix_df.values, 0)


    # Convert the square distance matrix to a condensed distance matrix (1D array)
    # which is what `linkage` function expects.
    try:
        condensed_distance_matrix = squareform(distance_matrix_df, force='tovector', checks=True)
    except ValueError as e:
        print(f"Error converting to condensed distance matrix: {e}")
        print("Distance matrix values being passed to squareform (first 5x5):")
        print(distance_matrix_df.iloc[:5, :5])
        # Attempt to repair if slightly off from perfect symmetric due to float issues
        try:
            print("Attempting to force symmetry for squareform...")
            condensed_distance_matrix = squareform(distance_matrix_df.values, force='tovector', checks=False) # Disable checks if issues persist
        except Exception as e2:
            print(f"Could not convert to condensed distance matrix even with checks=False: {e2}")
            return


    # Perform hierarchical/agglomerative clustering
    try:
        linked = linkage(condensed_distance_matrix, method=linkage_method)
    except Exception as e:
        print(f"Error during linkage calculation: {e}")
        print(f"Condensed distance matrix (first 20 elements): {condensed_distance_matrix[:20]}")
        return
    
    plt.figure(figsize=(max(10, len(similarity_matrix.index) * 0.6), 
                        max(7, len(similarity_matrix.index) * 0.35)))
    dendrogram(
        linked,
        orientation='top', 
        labels=similarity_matrix.index.tolist(),
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=45, 
        leaf_font_size=10
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel("LLM Model", fontsize=12)
    plt.ylabel(f"Distance (1 - Jaccard Similarity of Raw Names, '{linkage_method}' linkage)", fontsize=10) # Updated label
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Saved dendrogram to {output_path}")
    except Exception as e:
        print(f"Error saving dendrogram: {e}")
    plt.close()

def main():
    """Main function to visualize the existing Jaccard matrix."""
    print("Visualizing existing Jaccard Matrix for Method Name Overlap (Raw Names)...")
    input_filepath = REPORTS_DIR / INPUT_CSV_NAME
    
    jaccard_matrix_df = load_jaccard_matrix(input_filepath)
    
    if jaccard_matrix_df is None or jaccard_matrix_df.empty:
        print("Failed to load or empty Jaccard matrix. Exiting.")
        return
    
    # Generate Heatmap Plot
    heatmap_title = "Pairwise Jaccard Similarity (Exact Raw Method Names)"
    heatmap_output_path = ARTICLE_DIR / "MC_RawNames_Jaccard_Heatmap.png" # Changed filename
    generate_heatmap_plot(jaccard_matrix_df, heatmap_output_path, heatmap_title)
    
    # Generate Dendrogram Plot
    dendrogram_title = "Hierarchical Clustering by Exact Raw Method Name Similarity"
    dendrogram_output_path = ARTICLE_DIR / "MC_RawNames_Dendrogram_Similarity.png" # Changed filename
    generate_dendrogram_plot(jaccard_matrix_df, dendrogram_output_path,
                             dendrogram_title, linkage_method='average')
        
    print("\nVisualization of existing Jaccard matrix finished.")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("Warning: Seaborn style 'seaborn-v0_8-whitegrid' not found, using Matplotlib's default style.")
    main()