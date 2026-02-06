# Cascade EDA Notebook Documentation

This document provides a comprehensive guide to the `01_cascade_eda.ipynb` exploratory data analysis notebook for understanding cascading disaster patterns.

## Overview

The notebook analyzes NOAA Storm Events data to:
1. Identify relationships between weather events that trigger secondary events (cascades)
2. Visualize cascade patterns across different dimensions
3. Prepare labeled data for cascade prediction models

## Prerequisites

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn networkx
```

### Data
- NOAA Storm Events data (processed) in `data/processed/`
- Raw NOAA data should be in `noaa_data/` if re-processing is needed

---

## Key Concepts

### What is a Cascade?
A **cascade** occurs when one weather event (primary) triggers a secondary event through a documented causal mechanism.

**Examples:**
- Hurricane → Flash Flood (heavy rainfall causes rapid flooding)
- Wildfire → Debris Flow (burned soil becomes hydrophobic, causing erosion)
- Thunderstorm Wind → Tornado (severe storm evolution)

### Data Format

The `target` column is a **Python list** of secondary event types:

| Value | Example | Meaning |
|-------|---------|---------|
| Empty list | `[]` | No cascade triggered (isolated event) |
| Single item | `['Flash Flood']` | Triggers one secondary type |
| Multiple items | `['Flash Flood', 'Tornado']` | Triggers multiple secondary types |

> **Note:** When saved to CSV, lists are serialized as strings (e.g., `"['Flash Flood']"`). Use `ast.literal_eval()` to parse them back to lists.

### Key Columns

| Column | Description |
|--------|-------------|
| `target` | Secondary event type(s) this event triggers |
| `is_cascade_result` | True if this event was triggered by another event |
| `EVENT_TYPE` | NOAA event type classification |
| `TOTAL_DAMAGE_USD` | Total economic damage in USD |
| `FATALITY_COUNT` | Number of fatalities |

---

## Notebook Structure

### Part 1: Data Loading & Overview
Loads the NOAA storm events data and displays:
- Dataset shape and memory usage
- Column data types and missing value percentages
- Sample records

**Key Variables Created:**
- `df`: The main dataframe with all storm events

### Part 2: Key Columns for Cascade Analysis
Defines the essential columns used throughout the analysis:

| Category | Columns |
|----------|---------|
| Identifiers | EVENT_ID, EPISODE_ID, EVENT_TYPE |
| Location | STATE, CZ_NAME, BEGIN_LAT/LON, STATE_FIPS |
| Temporal | BEGIN_DATETIME, END_DATETIME |
| Severity | MAGNITUDE, TOR_F_SCALE, FLOOD_CAUSE |
| Impact | INJURIES_*, DEATHS_*, DAMAGE_*_USD |
| Narrative | EVENT_NARRATIVE, EPISODE_NARRATIVE |

### Part 3: Cascade Identification & Labeling
Applies domain-constrained cascade identification:
- Uses causal patterns from `src/cascade_definition.py`
- Labels events with their potential secondary event types

**Key Variables Created:**
- `df_labeled`: Events with cascade labels (`target` column)
- `cascade_events`: Subset of events that trigger cascades

### Part 4: What to Visualize Next?
Lists visualization ideas (implemented in Parts 5-9).

---

## Part 5: Pattern Analysis

### Co-occurrence Heatmap
**Purpose:** Shows which primary → secondary event pairs occur most frequently.

**How it works:**
1. Extracts all cascade pairs from the `target` column
2. Handles multi-label targets (pipe-separated or list format)
3. Creates a matrix counting pair frequencies
4. Filters to top 15 primaries and secondaries for readability

**Key function:**
```python
def explode_targets(row):
    target = row['target']
    # Split by | or parse list format
    targets = str(target).split('|')
    return [(row['EVENT_TYPE'], t.strip()) for t in targets if t.strip()]
```

**Output:** `results/cooccurrence_heatmap.png`

### Network Graph
**Purpose:** Visualizes cascade relationships as a directed graph.

**How it works:**
1. Creates a NetworkX directed graph from cascade pairs
2. Node size = total cascade involvement (in-degree + out-degree)
3. Edge width = frequency of that specific pair
4. Uses spring layout for visual clarity

**Output:** `results/cascade_network.png`

---

## Part 6: Temporal Analysis

### Event Duration by Type
**Purpose:** Shows how long cascade-triggering events typically last.

**How it works:**
1. Calculates duration = END_DATETIME - BEGIN_DATETIME
2. Filters to reasonable durations (0-72 hours)
3. Shows box plots by event type for cascade-triggering events

**Output:** `results/event_duration_by_type.png`

### Seasonal Patterns
**Purpose:** Identifies which months have the most cascades by type.

**How it works:**
1. Extracts month from BEGIN_DATETIME
2. Creates a pivot table: month × event type
3. Displays as a heatmap showing seasonal concentration

**Output:** `results/seasonal_cascade_patterns.png`

---

## Part 7: Geographic Analysis

### States with Highest Cascade Rates
**Purpose:** Identifies states with the most cascade activity.

**Shows two views:**
1. **Absolute counts:** Total number of cascade-triggering events
2. **Cascade rate:** Percentage of events that trigger cascades

**Calculation:**
```python
cascade_rate = (cascade_events_per_state / total_events_per_state) * 100
```

**Output:** `results/state_cascade_rates.png`

### Regional Cascade Patterns
**Purpose:** Groups states by region to show regional differences.

**Regions defined:**
- West Coast (CA, OR, WA)
- Mountain (MT, ID, WY, CO, UT, NV, AZ, NM)
- Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH)
- South Central (TX, OK, AR, LA)
- Southeast (KY, TN, MS, AL, GA, FL, SC, NC, VA, WV)
- Northeast (ME, NH, VT, MA, RI, CT, NY, NJ, PA, DE, MD)

**Output:** `results/regional_cascade_patterns.png`

---

## Part 8: Impact Analysis

### Cascade vs Isolated Damage Comparison
**Purpose:** Determines if cascade events cause more damage than isolated events.

**Classification logic:**
```python
def has_valid_target(target):
    # Returns True only for non-empty, non-null targets
    if target in ['', '[]', 'nan', None]:
        return False
    return True

is_cascade = has_valid_target(target) | (is_cascade_result == True)
```

**Metrics compared:**
- Mean damage
- Median damage
- Total damage
- Event counts

**Output:** `results/cascade_vs_isolated_damage.png`

### Fatalities in Cascade Chains
**Purpose:** Analyzes fatality patterns in cascade events.

**Shows:**
1. Total fatalities by event category
2. Fatality rate (per 1000 events)

**Output:** `results/cascade_fatalities.png`

### Economic Impact by Cascade Type
**Purpose:** Identifies which cascade types cause the most economic damage.

**Shows:**
1. Total damage by primary event type
2. Average damage per event

**Output:** `results/economic_impact_by_type.png`

---

## Part 9: Chain Analysis

### Multiple Secondaries
**Purpose:** Shows how many secondary types each primary event triggers.

**Key function:**
```python
def count_secondaries(target):
    # Handle list format: "['Flash Flood', 'Tornado']"
    if target.startswith('['):
        items = ast.literal_eval(target)
        return len(items)
    # Handle pipe format: "Flash Flood|Tornado"
    return len(target.split('|'))
```

**Key insight:** Most cascade events trigger 1 secondary type, but some trigger multiple (e.g., hurricanes → floods, tornadoes, storm surge).

**Output:** `results/multi_secondary_distribution.png`

### Hub Events
**Purpose:** Identifies "hub" event types that initiate the most cascades.

**Analyzes:**
1. Count of cascades triggered by each event type
2. Diversity (unique secondary types triggered)
3. Scatter plot: frequency vs diversity

**Output:** `results/hub_events.png`

### Multi-step Chains (A→B→C)
**Purpose:** Finds events that are both triggered by AND trigger other events.

**Filter logic:**
```python
middle_events = df[
    (df['is_cascade_result'] == True) &  # Was triggered
    (df['target'].apply(has_valid_target))  # Triggers another
]
```

**Example:** Hurricane → Flash Flood → Debris Flow

**Output:** `results/multi_step_chains.png`

---

## Part 10: Save Enhanced Dataset

### Summary Tables Export
Saves key analysis results as CSV files:
- `top_cascade_pairs.csv`: Most frequent primary→secondary pairs
- `hub_events_analysis.csv`: Hub event statistics
- `state_cascade_counts.csv`: Cascades by state
- `damage_by_cascade_type.csv`: Economic impact by type

### Enhanced Dataset Export
Saves `events_labeled_full.csv` with all columns needed for prediction:

| Category | Columns Included |
|----------|-----------------|
| Identifiers | EVENT_ID, EPISODE_ID, EVENT_TYPE |
| Location | STATE, CZ_NAME, FIPS codes, LAT/LON, WFO |
| Temporal | BEGIN_DATETIME, END_DATETIME |
| Severity | MAGNITUDE, TOR_* (tornado), FLOOD_CAUSE |
| Impact | Injuries, Deaths, Damage (property, crops, total) |
| Narrative | EVENT_NARRATIVE, EPISODE_NARRATIVE |
| Labels | target, is_cascade_result |

---

## Output Files

### Results Directory (`results/`)

| File | Description |
|------|-------------|
| `cooccurrence_heatmap.png` | Primary→Secondary pair frequency |
| `cascade_network.png` | Directed graph of cascade relationships |
| `event_duration_by_type.png` | Duration distributions |
| `seasonal_cascade_patterns.png` | Monthly patterns by type |
| `state_cascade_rates.png` | State-level cascade statistics |
| `regional_cascade_patterns.png` | Regional breakdown |
| `cascade_vs_isolated_damage.png` | Damage comparison |
| `cascade_fatalities.png` | Fatality analysis |
| `economic_impact_by_type.png` | Damage by cascade type |
| `multi_secondary_distribution.png` | Multi-secondary statistics |
| `hub_events.png` | Hub event analysis |
| `multi_step_chains.png` | A→B→C chain analysis |
| `top_cascade_pairs.csv` | Top 20 cascade pairs |
| `hub_events_analysis.csv` | Hub statistics |
| `state_cascade_counts.csv` | State counts |
| `damage_by_cascade_type.csv` | Damage by type |

### Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `events_labeled.csv` | Basic labeled dataset |
| `events_labeled_full.csv` | Complete dataset with all prediction columns |

---

## Running the Notebook

1. **Activate your environment:**
   ```bash
   conda activate your_env  # or your venv
   ```

2. **Ensure data is in place:**
   ```bash
   ls data/processed/  # Should contain events_labeled.csv
   ```

3. **Run cells sequentially:**
   - Parts 1-4: Load data and create labels
   - Part 5 setup cell: Creates `results/` directory
   - Parts 5-9: Generate visualizations (auto-saved)
   - Part 10: Save final outputs

4. **Check outputs:**
   ```bash
   ls results/  # Should contain 12 PNG + 4 CSV files
   ```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'networkx'"
```bash
pip install networkx
```

### "ValueError: The truth value of an array..."
The `target` column may contain array values. Use explicit checks:
```python
# Wrong
if pd.isna(target):

# Right
if target is None or (isinstance(target, float) and pd.isna(target)):
```

### All events classified as cascades
Check the `target` column format. If targets are stored as `['Flash Flood']` (list-as-string), use:
```python
import ast
if target.startswith('['):
    items = ast.literal_eval(target)
```

### Missing columns in saved dataset
Some columns may be missing if the original data didn't include them. The notebook merges only available columns.

### "posx and posy should be finite values"
This matplotlib warning occurs when plotting NaN values. Filter data before plotting:
```python
data = data.dropna()
data = data[data > 0]  # For log scale plots
```

---

## Code Patterns Reference

### Parsing Target from CSV (List-as-String)
When loading from CSV, the target column is serialized as a string. Use this to parse:

```python
import ast

def parse_target(target):
    """Parse target column from CSV (where lists become strings)."""
    if target is None or (isinstance(target, float) and pd.isna(target)):
        return []
    
    target_str = str(target).strip()
    if target_str in ['', '[]', 'nan', 'None']:
        return []
    
    # Parse list format: "['Flash Flood', 'Tornado']"
    if target_str.startswith('['):
        try:
            return ast.literal_eval(target_str)
        except:
            return []
    
    return [target_str]  # Single value
```

### Checking Valid Targets
```python
def has_valid_target(target):
    """Return True if target is non-empty."""
    if target is None:
        return False
    if isinstance(target, float) and pd.isna(target):
        return False
    target_str = str(target).strip()
    return target_str not in ['', '[]', 'nan', 'None', 'NaN']
```
