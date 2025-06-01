#!/usr/bin/env python3
"""
RQ1-Method_Quantity_MQ.py

Generates:
1.  A CSV table summarizing method quantity per LLM (Total, Min, Max).
2.  A bar chart with error bars for mean methods per LLM across runs.
3.  A heatmap of mean methods per class by LLM.

Reads from: reports/Combined_Struct_Counts_Metrics.csv
Saves outputs to: reports/article/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_NAME = "Combined_Struct_Counts_Metrics.csv" # From main3.py
ARTICLE_DIR = REPORTS_DIR / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True) # Ensure subdirectory exists

# Define class names that were columns in Combined_Struct_Counts_Metrics.csv
# This list should match the classes for which counts were recorded.
# If CLASS_NAMES was empty in main3.py, we need to discover these from the CSV columns.
# For now, let's assume we can infer them or have a predefined list if necessary.

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads the main CSV data."""
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def generate_method_quantity_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Generates a table: LLM, Total Methods (all runs), Min (per run), Max (per run).
    """
    if 'Model' not in df.columns or 'Target_Class_Methods' not in df.columns:
        print("Error: Required columns ('Model', 'Target_Class_Methods') not in DataFrame.")
        return None

    # Group by Model
    grouped = df.groupby('Model')['Target_Class_Methods']

    # Calculate metrics
    total_methods = grouped.sum().rename("Total_Generated_Methods_All_Runs")
    min_methods_per_run = grouped.min().rename("Min_Methods_Per_Run")
    max_methods_per_run = grouped.max().rename("Max_Methods_Per_Run")
    # mean_methods_per_run = grouped.mean().rename("Mean_Methods_Per_Run") # For bar chart

    summary_df = pd.concat([total_methods, min_methods_per_run, max_methods_per_run], axis=1)
    summary_df = summary_df.reset_index() # Make 'Model' a column

    # Add Grand Total row (only for the 'Total_Generated_Methods_All_Runs' column)
    grand_total_sum = summary_df['Total_Generated_Methods_All_Runs'].sum()
    grand_total_row = pd.DataFrame({
        'Model': ['Grand Total'],
        'Total_Generated_Methods_All_Runs': [grand_total_sum],
        'Min_Methods_Per_Run': [np.nan], # Min/Max don't make sense for grand total
        'Max_Methods_Per_Run': [np.nan]
    })
    summary_df = pd.concat([summary_df, grand_total_row], ignore_index=True)

    return summary_df

def generate_barchart_mean_methods(df: pd.DataFrame, output_path: Path):
    """
    Generates a bar chart with error bars for mean methods per LLM across runs.
    """
    if 'Model' not in df.columns or 'Target_Class_Methods' not in df.columns:
        print("Error: Required columns for bar chart not in DataFrame. Skipping bar chart.")
        return

    # Calculate mean and standard error for error bars
    model_stats = df.groupby('Model')['Target_Class_Methods'].agg(['mean', 'std', 'count'])
    model_stats['sem'] = model_stats['std'] / np.sqrt(model_stats['count']) # Standard Error of the Mean

    # Sort by mean for better visualization
    model_stats = model_stats.sort_values(by='mean', ascending=False)

    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_stats.index, model_stats['mean'], yerr=model_stats['sem'], capsize=5, color=sns.color_palette("viridis", len(model_stats)))
    
    plt.xlabel("LLM Model", fontsize=12)
    plt.ylabel("Mean Number of Methods Generated per Run", fontsize=12)
    plt.title("Mean Method Quantity per LLM (with 95% CI approximation via SEM)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Saved bar chart to {output_path}")
    except Exception as e:
        print(f"Error saving bar chart: {e}")
    plt.close()


def discover_class_columns(df_columns: pd.Index, model_col='Model', run_col='Run', file_col='File') -> list:
    """
    Discovers class name columns from the DataFrame columns.
    Excludes known non-class columns.
    """
    # Columns that are definitely not class names with counts
    # Add any other metadata columns from Combined_Struct_Counts_Metrics.csv
    known_meta_cols = [
        model_col, run_col, file_col,
        'Target_Class_Methods', 'Extra_Methods_Count', 'Unique_Method_Names',
        'Redundancy', 'ParamRichness', 'ReturnTypeCompleteness',
        'Percentage_Methods_With_UC', 'Percentage_Methods_With_Action',
        'Percentage_Methods_With_Both', 'Percentage_Methods_Without',
        'TotalMethodsInScope' # Added in main3.py
    ]
    # Also exclude structural metrics if they are present
    structural_cols_prefixes = (
        'packages_', 'enums_', 'enum_values_', 'classes_', 'attributes_', 'relationships_',
        'Overall_Preservation_%', 'Total_Baseline_Elements', 'Total_Preserved_Elements', 'Total_Added_Elements'
    )

    class_cols = []
    for col in df_columns:
        if col not in known_meta_cols and not col.startswith(structural_cols_prefixes):
            class_cols.append(col)
    
    if not class_cols:
        print("Warning: No class columns discovered for heatmap. Heatmap might be empty or incorrect.")
        print("Please check the columns in Combined_Struct_Counts_Metrics.csv.")
    return class_cols


def generate_heatmap_mean_methods_per_class(df: pd.DataFrame, class_columns: list, output_path: Path):
    """
    Generates a heatmap of mean methods per class by LLM.
    """
    if 'Model' not in df.columns or not class_columns:
        print("Error: 'Model' column or class columns missing. Skipping heatmap.")
        return

    # Ensure class columns exist in df
    valid_class_columns = [col for col in class_columns if col in df.columns]
    if not valid_class_columns:
        print("Error: None of the provided class columns exist in DataFrame. Skipping heatmap.")
        return

    # Calculate mean methods per class for each model
    # We need to melt the DataFrame first if classes are columns, or pivot if it's long
    # Assuming df has one row per run, and classes are columns with counts for that run
    
    heatmap_data_list = []
    for model, model_group in df.groupby('Model'):
        for cls_col in valid_class_columns:
            mean_val = model_group[cls_col].mean()
            heatmap_data_list.append({'Model': model, 'Class': cls_col, 'Mean_Methods': mean_val})
    
    if not heatmap_data_list:
        print("No data to plot for heatmap. Skipping.")
        return

    heatmap_df = pd.DataFrame(heatmap_data_list)
    
    # Pivot table for heatmap
    try:
        pivot_table = heatmap_df.pivot(index="Model", columns="Class", values="Mean_Methods")
    except Exception as e:
        print(f"Error creating pivot table for heatmap: {e}")
        print("Data for heatmap pivot:", heatmap_df.head())
        return

    if pivot_table.empty:
        print("Pivot table for heatmap is empty. Skipping heatmap generation.")
        return

    plt.figure(figsize=(16, 10)) # Adjust size as needed
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Mean Methods per Class'})
    plt.title("Heatmap of Mean Methods per Class by LLM", fontsize=14)
    plt.xlabel("Class Name", fontsize=12)
    plt.ylabel("LLM Model", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    plt.close()


def main():
    """Main function to generate reports."""
    print("Starting RQ1 - Method Quantity (MQ) analysis...")
    input_filepath = REPORTS_DIR / INPUT_CSV_NAME
    
    df_combined = load_data(input_filepath)
    
    if df_combined is None:
        print("Failed to load data. Exiting.")
        return

    # 1. Generate Method Quantity Table
    print("\nGenerating method quantity table...")
    mq_table_df = generate_method_quantity_table(df_combined)
    if mq_table_df is not None:
        table_output_path = ARTICLE_DIR / "MQ_LLM_Method_Quantities.csv"
        try:
            mq_table_df.to_csv(table_output_path, index=False)
            print(f"Saved method quantity table to {table_output_path}")
        except Exception as e:
            print(f"Error saving method quantity table: {e}")
    else:
        print("Skipped generating method quantity table due to errors.")

    # 2. Generate Bar Chart
    print("\nGenerating bar chart for mean methods per LLM...")
    barchart_output_path = ARTICLE_DIR / "MQ_BarChart_MeanMethodsPerLLM.png"
    generate_barchart_mean_methods(df_combined, barchart_output_path)

    # 3. Generate Heatmap
    print("\nGenerating heatmap for mean methods per class by LLM...")
    # Discover class columns from the input DataFrame
    # These are the columns in Combined_Struct_Counts_Metrics.csv that represent individual class method counts
    class_columns_for_heatmap = discover_class_columns(df_combined.columns)
    if class_columns_for_heatmap:
        print(f"Discovered {len(class_columns_for_heatmap)} class columns for heatmap: {class_columns_for_heatmap[:5]}...") # Print first 5
        heatmap_output_path = ARTICLE_DIR / "MQ_Heatmap_MeanMethodsPerClass.png"
        generate_heatmap_mean_methods_per_class(df_combined, class_columns_for_heatmap, heatmap_output_path)
    else:
        print("No class columns found for heatmap. Skipping heatmap generation.")
        
    print("\nRQ1 - Method Quantity (MQ) analysis finished.")

if __name__ == "__main__":
    # Set a default style for plots for better appearance
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
    main()