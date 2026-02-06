# Cascading Disaster Prediction

How can ML modeling capture cascade dynamics in disaster sequences?

## Project Overview

This project analyzes NOAA Storm Events data to predict cascading disaster patterns - understanding how one weather event can trigger secondary events (e.g., hurricanes → flash floods → debris flows).

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn networkx

# Run the EDA notebook
jupyter notebook notebooks/01_cascade_eda.ipynb
```

## Project Structure

```
cascading-disaster-prediction/
├── data/
│   ├── processed/          # Processed datasets
│   │   ├── events_labeled.csv
│   │   └── events_labeled_full.csv
│   └── raw/                # Raw data files
├── docs/
│   ├── cascade_definition.md      # Cascade pattern definitions
│   └── cascade_eda_notebook.md    # Detailed notebook documentation
├── notebooks/
│   └── 01_cascade_eda.ipynb       # Main EDA notebook
├── results/                # Generated visualizations and tables
├── src/
│   ├── cascade_definition.py     # Causal cascade patterns
│   └── cascade_identification.py # Cascade detection logic
└── noaa_data/             # Raw NOAA Storm Events data
```

## Documentation

- **[Cascade Definition](docs/cascade_definition.md)** - Defines the causal relationships used for cascade identification
- **[EDA Notebook Guide](docs/cascade_eda_notebook.md)** - Comprehensive guide to the analysis notebook

## Key Outputs

The EDA notebook generates:

| Output Type | Location | Description |
|-------------|----------|-------------|
| Visualizations | `results/*.png` | 12 figures analyzing cascade patterns |
| Summary Tables | `results/*.csv` | 4 CSVs with key statistics |
| Labeled Dataset | `data/processed/events_labeled_full.csv` | Prediction-ready data |

## Data Source

NOAA Storm Events Database: https://www.ncei.noaa.gov/stormevents/ftp.jsp

## Collaboration

Project doc: https://docs.google.com/document/d/1Es5w9ciBOnjKLxs32QlmwyXypAC8rs1m9SwG4Km8HAE/edit?usp=sharing