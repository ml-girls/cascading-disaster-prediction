# Feature Engineering Documentation
## Multilabel Cascade Prediction

---

## Overview

The feature engineering pipeline transforms raw NOAA storm event data into appropriate features for predicting cascade disasters. Features are divided into:

- **Base Features** (~40 features): Created from single events. Now includes detailed characteristics like severe wind/hail flags and path metrics.
- **Aggregate Features** (~130 features): Learned from training data (Location, State, and Event Type historical statistics).
- **Historical Features** (Optional, window-based): Temporal lookback for cumulative activity.

---

## Raw Features (From NOAA Data)

### Event Identification
- `EVENT_ID` - Unique event identifier (excluded from modeling)
- `EPISODE_ID` - Episode grouping (excluded from modeling)
- `EVENT_TYPE` - Type of weather event (encoded for modeling)

### Temporal Information
- `BEGIN_DATETIME` - Event start timestamp
- `END_DATETIME` - Event end timestamp
- `YEAR`, `MONTH_NAME` - Calendar information

### Geographic Information
- `STATE` - State name
- `CZ_TYPE` - County zone type (C=County, Z=Zone, M=Marine)
- `CZ_FIPS` - County zone FIPS code
- `BEGIN_LAT`, `BEGIN_LON` - Start coordinates
- `END_LAT`, `END_LON` - End coordinates
- `LOCATION_KEY` - Composite location identifier (STATE_FIPS + CZ_FIPS)

### Impact Metrics
- `DAMAGE_PROPERTY` - Property damage (raw string, e.g., '10.00K')
- `DAMAGE_CROPS` - Crop damage (raw string)
- `total_damage_usd` - Parsed numeric total USD (combined property + crops)

### Event-Specific Details
- `MAGNITUDE` - Event magnitude (wind knots or hail inches)
- `MAGNITUDE_TYPE` - Type of magnitude measurement
- `TOR_F_SCALE` - Tornado F/EF scale
- `TOR_LENGTH` - Tornado path length (miles)
- `TOR_WIDTH` - Tornado path width (yards)
- `FLOOD_CAUSE` - Cause of flooding (Ice Jam, Rain, Snowmelt, etc.)

---

## Engineered Features

### 1. Temporal Features
**Purpose:** Capture seasonal, daily, and hourly patterns.

| Feature | Description |
| :--- | :--- |
| `event_duration_hours` | Duration of event in hours. |
| `month`, `hour`, `day_of_week`, `day_of_year` | Raw time components for tree models. |
| `month_sin/cos`, `hour_sin/cos` | Cyclical encoding for linear models. |

---

### 2. Impact/Severity Features
**Purpose:** Quantify event magnitude and human/financial disruption.

| Feature | Description |
| :--- | :--- |
| `damage_property_usd`, `damage_crops_usd` | Numeric USD values parsed from NOAA strings. |
| `total_damage_usd` | Sum of property and crop damage. |
| `log_damage` | $log(1 + \text{total\_damage\_usd})$ to handle dispersion. |
| `total_injuries`, `total_deaths` | Sum of direct and indirect human impact. |
| `has_fatalities` | Binary flag for any death. |
| `severity_score` | Composite: $(\text{damage\_usd} / 1M) + (\text{deaths} \times 100) + (\text{injuries} \times 10)$. |

---

### 3. Event Detail & Path Features
**Purpose:** New characteristics characterizing small-scale intensities and event movement.

| Feature | Description |
| :--- | :--- |
| `magnitude` | Numeric magnitude (wind speed or hail size). |
| `is_severe_wind` | Flag for wind $\ge 50$ knots. |
| `is_significant_hail` | Flag for hail $\ge 1.0$ inch. |
| `is_marine_event`, `is_zone_event` | Flags based on `CZ_TYPE`. |
| `is_ice_jam_flood`, `is_rain_flood`, `is_snowmelt_flood` | Parsed from `FLOOD_CAUSE` text. |
| `event_path_length_miles` | Haversine distance calculated between BEGIN and END coordinates. |
| `tor_path_area` | Calculated tornado footprint (length $\times$ width in miles). |

---

### 4. Historical Features (Windowed)
**Purpose:** Recent activity at a location predicts future cascades. Calculates 7-day and 30-day windows for:
- **Event Frequency**: Count of events and cascades.
- **Cumulative Damage**: Total USD damage in window.
- **Max Severity**: Peak human/financial impact in window.

---

### 5. Aggregate Statistics (Learned from Training)
**Purpose:** Captures long-term historical tendencies per location, state, and event type.

#### General Statistics
| Category | Features |
| :--- | :--- |
| **Location** | historical event count, avg damage, avg severity, cascade rate. |
| **State** | historical avg damage, avg severity, cascade rate. |
| **Event Type** | historical avg damage, avg severity. |

#### Conditional Cascade Probabilities
Calculates $P(\text{Secondary} | \text{Primary})$ for each of the secondary event types (e.g., Flash Flood, High Wind). 
- **Example**: If historically 15% of Tornadoes trigger a Flash Flood, the feature `p_Flash_Flood_given_primary` will be 0.15 for all Tornado events.

---