#!/usr/bin/env python3
"""
RQ1-Annotation_Completeness_AC.py

Analyzes annotation completeness for generated methods.
Reads from: reports/Annotation_and_Mapping_Combined.csv
Generates:
1.  A CSV table summarizing annotation categories per LLM.
2.  A stacked bar chart showing annotation completeness by category per model.
3.  A grouped bar chart comparing full vs. no annotation per model.
Saves outputs to: reports/article/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
REPORTS_DIR = Path("reports")
INPUT_CSV_ANNOTATION = REPORTS_DIR / "Annotation_and_Mapping_Combined.csv"
ARTICLE_DIR = REPORTS_DIR / "article"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True) # Ensure subdirectory exists

def load_annotation_data(filepath: Path) -> pd.DataFrame | None:
    """Loads the annotation and mapping CSV data."""
    if not filepath.is_file():
        print(f"Error: Input CSV file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        # Ensure 'UC_References' and 'UC_Action' are treated as strings, handle NaNs as empty
        df['UC_References'] = df['UC_References'].fillna('').astype(str)
        df['UC_Action'] = df['UC_Action'].fillna('').astype(str)
        print(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def categorize_annotation(row) -> str:
    """Categorizes a method's annotation completeness."""
    has_uc = bool(row['UC_References'].strip())
    has_action = bool(row['UC_Action'].strip())

    if has_uc and has_action:
        return "Full"
    elif has_uc and not has_action:
        return "UC-only"
    elif not has_uc and has_action:
        return "Action-only"
    else: # not has_uc and not has_action
        return "None"

def generate_annotation_completeness_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Generates a table: Model, Full, UC-only, Action-only, None.
    Counts are total number of methods in each category across all runs for the model.
    """
    if df.empty or 'Model' not in df.columns:
        print("Error: Input DataFrame is empty or missing 'Model' column.")
        return None

    df['AnnotationCategory'] = df.apply(categorize_annotation, axis=1)

    # Count methods in each category per model
    # This will count each method instance from each run
    category_counts = df.groupby(['Model', 'AnnotationCategory'], observed=True).size().unstack(fill_value=0)

    # Ensure all categories are present as columns, even if some have zero counts
    all_categories = ["Full", "UC-only", "Action-only", "None"]
    for cat in all_categories:
        if cat not in category_counts.columns:
            category_counts[cat] = 0
    
    # Reorder columns as typically presented
    summary_df = category_counts[all_categories].reset_index()
    
    # Add a total column (total methods analyzed per model)
    summary_df['Total_Methods_Analyzed'] = summary_df[all_categories].sum(axis=1)

    # Add a Grand Total row
    grand_totals = summary_df[all_categories + ['Total_Methods_Analyzed']].sum()
    grand_total_row = pd.DataFrame([grand_totals], columns=grand_totals.index)
    grand_total_row['Model'] = "Grand Total"
    
    summary_df = pd.concat([summary_df, grand_total_row], ignore_index=True)
    
    return summary_df

def plot_stacked_bar_annotation_completeness(summary_df: pd.DataFrame, output_path: Path):
    """Plots a stacked bar chart of annotation completeness categories per model, with Y-axis as percentages."""
    if summary_df.empty:
        print("No data for stacked bar plot. Skipping.")
        return

    plot_df = summary_df[summary_df['Model'] != "Grand Total"].copy()
    if plot_df.empty:
        print("No model data (excluding Grand Total) for stacked bar plot. Skipping.")
        return

    plot_df.set_index('Model', inplace=True)
    
    # Define the desired order and ensure these columns exist
    # If 'Action-only' has no data, it won't contribute to the stack or labels if not present.
    categories_in_desired_order = ["Full", "UC-only", "Action-only", "None"]
    
    # Filter to categories actually present in the data and that have some values
    categories_to_plot = [
        cat for cat in categories_in_desired_order 
        if cat in plot_df.columns and plot_df[cat].sum() > 0
    ]

    if not categories_to_plot:
        print("No categories with data to plot in stacked bar chart. Skipping.")
        return

    # Use only the categories that will be plotted for total calculation and percentages
    if 'Total_Methods_Analyzed' not in plot_df.columns:
        print("Warning: 'Total_Methods_Analyzed' column missing. Calculating it for percentages based on plottable categories.")
        plot_df['Total_Methods_Analyzed'] = plot_df[categories_to_plot].sum(axis=1)

    percent_df = plot_df[categories_to_plot].copy() # Ensure we only work with plottable categories
    for cat in categories_to_plot:
        percent_df[cat] = np.where(
            plot_df['Total_Methods_Analyzed'] > 0,
            (plot_df[cat] / plot_df['Total_Methods_Analyzed']) * 100,
            0 
        )
    
    if "Full" in percent_df.columns: # Sort by 'Full' if it's being plotted
        percent_df = percent_df.sort_values(by="Full", ascending=False)
    # else: # Optional: Fallback sort if 'Full' is not plotted
    #     percent_df = percent_df.iloc[percent_df.sum(axis=1).sort_values(ascending=False).index]

    # Get the colormap that will be used by pandas plot
    colormap_name = "viridis" # The colormap being used
    cmap = plt.get_cmap(colormap_name)
    # Assign colors to the categories that will be plotted
    category_colors = {
        category: cmap(i / (len(categories_to_plot) -1 if len(categories_to_plot) > 1 else 1) ) 
        for i, category in enumerate(categories_to_plot)
    }

    ax = percent_df.plot(kind='bar', stacked=True, figsize=(14, 8), 
                         color=[category_colors[cat] for cat in percent_df.columns]) # Use consistent colors
    
    plt.title('Annotation Completeness by LLM Model (Percentages)', fontsize=16)
    plt.xlabel('LLM Model', fontsize=12)
    plt.ylabel('Percentage of Methods (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(0, 101, 10), fontsize=10)
    ax.set_ylim(0, 100.5)

    # --- Text Labels on Stacked Bars ---
    for bar_idx, model_name in enumerate(percent_df.index): # Iterate through models (bars on x-axis)
        cumulative_height = 0
        for category_name in percent_df.columns: # Iterate through categories in the order they are stacked
            value = percent_df.loc[model_name, category_name]
            
            if value > 0.1: # Only add label if the segment has a noticeable height
                label_y_position = cumulative_height + (value / 2)
                
                text_color = 'black' # Default text color
                segment_color_rgb = category_colors.get(category_name, (0,0,0,0))[:3] # Get RGB, ignore alpha

                # Simple heuristic for text color based on luminance (approximation)
                # Luminance = 0.299*R + 0.587*G + 0.114*B
                luminance = 0.299 * segment_color_rgb[0] + 0.587 * segment_color_rgb[1] + 0.114 * segment_color_rgb[2]
                
                # Your specific color requests:
                # "white for the blue (Action-only) and purple (full), use black for yellow"
                # Assuming "viridis": "Full" is often dark purple, "UC-only" is blue-green,
                # "Action-only" might be green-yellow, "None" might be bright yellow.
                # This mapping depends on the number of categories and their order.
                
                if category_name == "Full": # Typically dark purple/blue in viridis start
                    text_color = 'white'
                elif category_name == "UC-only": # Often a mid-range color
                     text_color = 'white' if luminance < 0.4 else 'black' # Example threshold
                elif category_name == "Action-only": # Might be greenish or yellowish
                    text_color = 'white' if luminance < 0.4 else 'black' # You said white for blue (Action-only if it is blue)
                                                                        # if it's yellow, this will pick black.
                elif category_name == "None": # Typically bright yellow at the end of viridis
                    text_color = 'black'
                
                # Override based on your specific request if the category matches:
                if category_name == "Action-only": # Your "blue" (if its color is dark enough)
                     text_color = 'white' if luminance < 0.5 else 'black' # Adjust luminance threshold
                # If "Action-only" is yellow, you'd want black, which the luminance check might do.
                # Let's be more explicit for yellow zones:
                # If a segment is "yellowish" (high green, high red, low blue), use black.
                # A simple check: if G and R are high and B is low.
                # (This is a rough heuristic, proper color science is more complex)
                if segment_color_rgb[0] > 0.6 and segment_color_rgb[1] > 0.6 and segment_color_rgb[2] < 0.4: # Heuristic for yellow
                    text_color = 'black'


                # General override for very dark or very light segments if not covered above
                if text_color == 'black' and luminance < 0.3: # Very dark segment, black text won't show
                    text_color = 'white'
                elif text_color == 'white' and luminance > 0.7: # Very light segment, white text won't show
                    text_color = 'black'
                

                if value >= 2.5: # Only label segments >= 2.5%
                     ax.text(
                        bar_idx, # x-coordinate (center of the bar for this model)
                        label_y_position,
                        f"{value:.1f}%", 
                        ha='center', 
                        va='center', 
                        fontsize=7, 
                        color=text_color,
                        fontweight='bold' if text_color == 'white' else 'normal' # Bold white text for dark BG
                    )
            cumulative_height += value
    # --- End of Text Labels ---

    plt.legend(title='Annotation Category', labels=[cat.replace("_", " ") for cat in percent_df.columns], bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    try:
        plt.savefig(output_path)
        print(f"Saved stacked bar chart (percentages) to {output_path}")
    except Exception as e:
        print(f"Error saving stacked bar chart (percentages): {e}")
    plt.close()

def plot_grouped_bar_full_vs_none(summary_df: pd.DataFrame, output_path: Path):
    """Plots a grouped bar chart comparing Full vs. None annotations per model."""
    if summary_df.empty:
        print("No data for grouped bar plot. Skipping.")
        return

    plot_df = summary_df[summary_df['Model'] != "Grand Total"].copy()
    if plot_df.empty:
        print("No model data (excluding Grand Total) for grouped bar plot. Skipping.")
        return

    if 'Full' not in plot_df.columns or 'None' not in plot_df.columns:
        print("Missing 'Full' or 'None' categories in data for grouped bar plot. Skipping.")
        return

    plot_df.set_index('Model', inplace=True)
    
    # Sort models by 'Full' annotations
    plot_df = plot_df.sort_values(by="Full", ascending=False)

    # Data for grouped bar chart
    categories_to_group = ["Full", "None"]
    plot_df[categories_to_group].plot(kind='bar', figsize=(14, 8), colormap="coolwarm")
    
    plt.title('Full Annotation vs. No Annotation by LLM Model', fontsize=16)
    plt.xlabel('LLM Model', fontsize=12)
    plt.ylabel('Number of Methods', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Annotation Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    try:
        plt.savefig(output_path)
        print(f"Saved grouped bar chart to {output_path}")
    except Exception as e:
        print(f"Error saving grouped bar chart: {e}")
    plt.close()


def main():
    """Main function to generate annotation completeness reports."""
    print("Starting RQ1 - Annotation Completeness (AC) analysis...")
    
    df_annotations = load_annotation_data(INPUT_CSV_ANNOTATION)
    
    if df_annotations is None or df_annotations.empty:
        print("Failed to load or data is empty. Exiting AC analysis.")
        return

    # 1. Generate Annotation Completeness Table
    print("\nGenerating annotation completeness table...")
    ac_table_df = generate_annotation_completeness_table(df_annotations.copy()) # Pass a copy
    if ac_table_df is not None and not ac_table_df.empty:
        table_output_path = ARTICLE_DIR / "AC_Annotation_Completeness_Summary.csv"
        try:
            ac_table_df.to_csv(table_output_path, index=False, float_format='%.0f') # Counts are integers
            print(f"Saved annotation completeness table to {table_output_path}")
        except Exception as e:
            print(f"Error saving annotation completeness table: {e}")
    else:
        print("Skipped generating annotation completeness table due to errors or no data.")

    # 2. Generate Stacked Bar Chart
    print("\nGenerating stacked bar chart for annotation completeness...")
    if ac_table_df is not None and not ac_table_df.empty:
        stacked_bar_output_path = ARTICLE_DIR / "AC_StackedBar_AnnotationCompleteness.png"
        plot_stacked_bar_annotation_completeness(ac_table_df, stacked_bar_output_path)
    else:
        print("Skipping stacked bar chart as summary table is not available.")

    # 3. Generate Grouped Bar Chart (Full vs. None)
    print("\nGenerating grouped bar chart for Full vs. None annotations...")
    if ac_table_df is not None and not ac_table_df.empty:
        grouped_bar_output_path = ARTICLE_DIR / "AC_GroupedBar_Full_vs_None.png"
        plot_grouped_bar_full_vs_none(ac_table_df, grouped_bar_output_path)
    else:
        print("Skipping grouped bar chart as summary table is not available.")
        
    print("\nRQ1 - Annotation Completeness (AC) analysis finished.")

if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        print("Seaborn style 'seaborn-v0_8-whitegrid' not found. Using default Matplotlib style.")
    main()