# LLM-Generated UML Class Diagram Evaluation Artifacts

This repository contains all artifacts related to the empirical study evaluating nine large language models (LLMs) on their ability to enrich a UML class diagram with methods derived from structured use cases. These materials support reproducibility and further research based strictly on the methodology and results described in the associated article.

## Contents

- **Prompts**: The three-part standardized prompts used to guide LLM method generation are provided in the `/prompt` directory. These include:
  - Primary instruction prompt with enrichment and annotation requirements.
  - Baseline UML class diagram prompt in PlantUML format (methodless).
  - Use-case corpus prompt containing all 21 structured use cases.

- **Baseline Diagram**: The original methodless UML class diagram (21 classes, 17 relationships) is available as a PlantUML `.puml` file in the `/puml` directory.

- **Augmented Diagrams**: Ninety PlantUML files (`ModelName_RunX.puml`) enriched with generated methods and annotations are stored in `/puml`.

- **Graphical Renderings**: Corresponding PNG images of all augmented diagrams, named identically to `.puml` files (e.g., `Claude3_run7.png`), are located in the `/Diagrams` directory.

- **Parsed JSON Files**: Machine-readable JSON representations extracted from the augmented PlantUML diagrams (`ModelName_RunX.json`) are included for metric computations.

- **Metric CSVs and Visualizations**: Summaries of quantitative metrics, consensus measures, and structural fidelity analyses are provided as CSV files, with visualizations (bar charts, boxplots, heatmaps) used in the article available in the `/reports/article` subdirectories.

## Repository Structure
├── prompt/ # LLM prompt texts guiding method generation
│ ├── 01_primary_instruction.txt
│ ├── 02_baseline_diagram.puml # Baseline diagram text prompt
│ └── 03_use_cases.txt
├── puml/ # PlantUML source files with enriched diagrams
│ ├── methodless.puml # Baseline methodless diagram
│ ├── Claude3_run1.puml
│ ├── Claude3_run2.puml
│ └── ... # Other model-run combinations
├── Diagrams/ # PNG renderings of all PlantUML diagrams
│ ├── Claude3_run1.png
│ ├── Claude3_run2.png
│ └── ...
├── JSON/ # Parsed JSON representations for analysis
│ ├── Claude3_run1.json
│ ├── Claude3_run2.json
│ └── ...
├── reports/ # Metric CSV files and visualizations
│ ├── Combined_Struct_Counts_Metrics.csv
│ ├── Annotation_and_Mapping_Combined.csv
│ ├── CoreMethods_TopN.csv
│ ├── JaccardMatrix_Global.csv
│ └── ...
│ ├── stats_mq/
│ ├── stats_sr/
│ ├── stats_ac/
│ ├── stats_sc/
│ ├── stats_tmc/
│ ├── stats_cc/
│ └── stats_spc/
└── README.md # This file


## Usage

This dataset and artifact collection enable direct reproduction of the quantitative analyses reported in the article, focusing on:

- The generate–measure–compare evaluation pipeline from methodless UML diagrams to enriched, annotated PlantUML diagrams.
- Multi-run, multi-model quantitative metrics: method quantity, signature richness, annotation completeness, structural fidelity, syntactic correctness, and consensus measures.
- Visualization of class-level method distribution and naming consensus.

No additional code or tooling is required to inspect the artifacts. Users interested in metric recalculation or further statistical analysis can leverage the parsed JSON and CSV data files.

## Citation

If you use these artifacts in your research, please cite the corresponding article:

> Djaber ROUABHIA and Ismail HADJADJ, "Behavioral Augmentation of UML Class Diagrams: An Empirical Study of Large Language Models for Method Generation" Journal/Conference Name, 2025.  
> DOI: [Insert DOI]

## Contact

For questions or collaborations, please contact:

Djaber ROUABHIA — djaber.rouabhia@univ-tebessa.dz  
Project link: https://github.com/yourusername/your-repo-name

