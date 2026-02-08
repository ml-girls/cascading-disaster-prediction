# Feature Engineering Documentation
## Multilabel Cascade Prediction

---

## Overview

The feature engineering pipeline transforms raw NOAA storm event data into appropriate features for predicting cascade disasters. Features are divided into:

- **Base Features** (~28 features): Created from single events with no temporal dependencies
- **Aggregate Features** (~122 features): Learned from training data, prevent leakage
- **Historical Features** (optional, ~20 features): Temporal lookback

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
- `STATE_FIPS` - State FIPS code
- `CZ_TYPE` - County zone type
- `CZ_FIPS` - County zone FIPS code
- `BEGIN_LAT`, `BEGIN_LON` - Start coordinates
- `END_LAT`, `END_LON` - End coordinates
- `LOCATION_KEY` - Composite location identifier (STATE_FIPS + CZ_FIPS)

### Impact Metrics
- `DEATHS_DIRECT` - Direct fatalities
- `DEATHS_INDIRECT` - Indirect fatalities
- `INJURIES_DIRECT` - Direct injuries
- `INJURIES_INDIRECT` - Indirect injuries
- `DAMAGE_PROPERTY` - Property damage (raw string)
- `DAMAGE_CROPS` - Crop damage (raw string)
- `TOTAL_DAMAGE_USD` - Total damage in USD (parsed numeric)

### Event-Specific Details
- `MAGNITUDE` - Event magnitude (type-dependent)
- `MAGNITUDE_TYPE` - Type of magnitude measurement
- `TOR_F_SCALE` - Tornado F/EF scale (F0-F5, EF0-EF5)
- `TOR_LENGTH` - Tornado path length (miles)
- `TOR_WIDTH` - Tornado path width (yards)
- `FLOOD_CAUSE` - Cause of flooding
- `CATEGORY` - Hurricane/storm category

### Narrative
- `EVENT_NARRATIVE` - Detailed event description (excluded from modeling)
- `EPISODE_NARRATIVE` - Episode-level description (excluded from modeling)

### Target Variables
- `is_cascade_result` - Binary: whether this event is a cascade result
- `target` - List of secondary event types triggered (multilabel target)

---

## Engineered Features

### 1. Temporal Features (11 features)

**Purpose:** Capture seasonal, daily, and hourly patterns in cascade likelihood.

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `event_duration_hours` | Continuous | 0-∞ | Duration of event in hours |
| `month` | Integer | 1-12 | Month of year (raw, good for tree models) |
| `hour` | Integer | 0-23 | Hour of day (raw) |
| `day_of_week` | Integer | 0-6 | Day of week (0=Monday, 6=Sunday) |
| `day_of_year` | Integer | 1-365 | Day of year |
| `month_sin` | Float | -1 to 1 | Sine of month (cyclical encoding) |
| `month_cos` | Float | -1 to 1 | Cosine of month (cyclical encoding) |
| `hour_sin` | Float | -1 to 1 | Sine of hour (cyclical encoding) |
| `hour_cos` | Float | -1 to 1 | Cosine of hour (cyclical encoding) |

**Rationale:**
- Cascades may be more likely in certain seasons (e.g., hurricanes in late summer)
- Time of day matters (e.g., flooding at night is more dangerous)
- Cyclical encoding helps linear models understand circular nature of time

---

### 2. Impact/Severity Features (5 features)

**Purpose:** Quantify event severity - stronger events more likely to cascade.

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `total_injuries` | Integer | 0-∞ | Direct + indirect injuries |
| `total_deaths` | Integer | 0-∞ | Direct + indirect fatalities |
| `has_fatalities` | Binary | 0-1 | Whether event caused any deaths |
| `log_damage` | Float | 0-∞ | log(1 + total_damage_usd) |
| `severity_score` | Float | 0-∞ | Composite: damage/1M + deaths×100 + injuries×10 |

**Severity Score Formula:**
```python
severity_score = (TOTAL_DAMAGE_USD / 1,000,000) + 
                 (DEATHS_DIRECT × 100) + 
                 (DEATHS_INDIRECT × 50) + 
                 (INJURIES_DIRECT × 10) + 
                 (INJURIES_INDIRECT × 5)
```

**Rationale:**
- Higher severity events disrupt more infrastructure → more cascades
- Log transformation handles extreme outliers in damage
- Composite score balances economic and human impact

---

### 3. Event Type Features (6 features)

**Purpose:** Different event types have different cascade patterns.

| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| `event_type_encoded` | Integer | 0-N | Numeric encoding of EVENT_TYPE |
| `is_hurricane_type` | Binary | 0-1 | Hurricane/tropical storm/storm surge |
| `is_flood_type` | Binary | 0-1 | Flash flood/flood/coastal flood |
| `is_winter_type` | Binary | 0-1 | Winter storm/blizzard/ice storm |
| `is_convective_type` | Binary | 0-1 | Tornado/thunderstorm/hail/lightning |
| `event_rarity_score` | Float | 0-1 | Frequency of this event type |

**Event Type Categories:**

**Hurricane Types:**
- Hurricane, Hurricane (Typhoon), Tropical Storm, Tropical Depression, Storm Surge/Tide

**Flood Types:**
- Flash Flood, Flood, Coastal Flood, Lakeshore Flood

**Winter Types:**
- Winter Storm, Blizzard, Heavy Snow, Ice Storm, Cold/Wind Chill, Extreme Cold/Wind Chill, Frost/Freeze, Winter Weather

**Convective Types:**
- Tornado, Thunderstorm Wind, Hail, Lightning

**Rationale:**
- Tree models benefit from numeric encoding
- Category flags capture domain knowledge (e.g., hurricanes often trigger floods)
- Rarity score: rare events may have different cascade patterns

---

### 4. Spatial Features (5 features)

**Purpose:** Geographic location affects cascade risk.

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `has_coordinates` | Binary | 0-1 | Whether lat/lon available |
| `latitude` | Float | ~25-50 | North latitude (continental US) |
| `longitude` | Float | ~-125 to -65 | West longitude (continental US) |
| `abs_latitude` | Float | 0-90 | Distance from equator |
| `is_coastal_state` | Binary | 0-1 | Whether state has coastline |

**Coastal States** (23 total):
- AL, AK, CA, CT, DE, FL, GA, HI, LA, ME, MD, MA, MS, NH, NJ, NY, NC, OR, RI, SC, TX, VA, WA

**Rationale:**
- Coastal areas: higher hurricane/storm surge risk → flooding cascades
- Latitude: affects seasonal patterns, storm types
- Missing coordinates: signals data quality issues

---

### 5. Tornado-Specific Features (1 feature)

**Purpose:** Tornado intensity affects damage and cascade likelihood.

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `tornado_intensity` | Integer | -1 to 5 | F/EF scale (0-5), -1 if not tornado |

**F/EF Scale Mapping:**
- F0/EF0 → 0 (65-85 mph)
- F1/EF1 → 1 (86-110 mph)
- F2/EF2 → 2 (111-135 mph)
- F3/EF3 → 3 (136-165 mph)
- F4/EF4 → 4 (166-200 mph)
- F5/EF5 → 5 (>200 mph)
- Non-tornado → -1

**Rationale:**
- Stronger tornadoes (EF3+) more likely to damage infrastructure
- Numeric scale captures intensity gradient

---

### 6. Historical Features (Optional, ~20 features)

**Purpose:** Recent activity at a location predicts future cascades.

#### Days Since Last Event (3 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `days_since_last_event` | Float | 0-999 | Days since any event at this location |
| `days_since_last_major` | Float | 0-999 | Days since high-severity event (>90th percentile) |
| `days_since_last_cascade` | Float | 0-999 | Days since last cascade at this location |

#### Windowed Counts (per window: 7d, 30d = 8 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `events_last_7d` | Integer | 0-∞ | Count of events in past 7 days |
| `events_last_30d` | Integer | 0-∞ | Count of events in past 30 days |
| `cascades_last_7d` | Integer | 0-∞ | Count of cascades in past 7 days |
| `cascades_last_30d` | Integer | 0-∞ | Count of cascades in past 30 days |

#### Windowed Aggregates (per window: 7d, 30d = 8 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `damage_last_7d` | Float | 0-∞ | Total damage (USD) in past 7 days |
| `damage_last_30d` | Float | 0-∞ | Total damage (USD) in past 30 days |
| `max_severity_last_7d` | Float | 0-∞ | Max severity score in past 7 days |
| `max_severity_last_30d` | Float | 0-∞ | Max severity score in past 30 days |

#### Event Density (1 feature)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `recent_event_density` | Float | 0-∞ | Events per day in 30-day window |

**Rationale:**
- Cascades often follow clusters of events
- Recent damage indicates vulnerable infrastructure
- Temporal proximity matters (events days apart vs. months apart)

---

### 7. Aggregate Features (Learned from Training Data, ~122 features)

These features are learned from training data only and applied to both train and test sets to prevent data leakage.

#### Location Statistics (4 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `location_event_count` | Integer | 0-∞ | Total historical events at this location |
| `location_avg_damage` | Float | 0-∞ | Average damage at this location |
| `location_avg_severity` | Float | 0-∞ | Average severity score at this location |
| `location_cascade_rate` | Float | 0-1 | Fraction of events that cascaded historically |

#### State Statistics (2 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `state_avg_damage` | Float | 0-∞ | State-level average damage |
| `state_cascade_rate` | Float | 0-1 | State-level cascade rate |

#### Cascade Pattern Features (~116 features)

**Purpose:** Learn cascade patterns and spatiotemporal characteristics from historical data.

For each of the 29 secondary event types, create **4 features**:

**1. Probability Features (29 features)**
- `p_{EventType}_given_primary` - P(trigger this secondary | primary event type)
- Example: `p_Flood_given_primary` = 0.15 (15% of Tornadoes trigger Floods)

**2. Time Gap Features (29 features)** 
- `avg_time_gap_{EventType}` - Average hours between primary and secondary event
- Example: `avg_time_gap_Flood` = 12.5 (Floods occur ~12.5 hours after primary)

**3. Distance Features (29 features)** 
- `avg_dist_{EventType}` - Average distance (km) between primary and secondary locations
- Example: `avg_dist_Flood` = 25.3 (Floods occur ~25km from primary epicenter)

**4. Same County Probability Features (29 features)** 
- `prob_same_county_{EventType}` - Probability cascade occurs in same county
- Example: `prob_same_county_Flood` = 0.85 (85% of floods are in-county)

**Total: 29 event types × 4 features = 116 features**

**Plus Aggregate:**
- `total_cascade_probability` - Sum of all probability features

**Example Values for a Tornado Primary Event:**
```python
# Flood cascade
p_Flood_given_primary = 0.15           # 15% probability
avg_time_gap_Flood = 12.5              # 12.5 hours later
avg_dist_Flood = 25.3                  # 25.3 km away
prob_same_county_Flood = 0.85          # 85% in same county

# Hail cascade (co-occurring)
p_Hail_given_primary = 0.45            # 45% probability
avg_time_gap_Hail = 0.5                # 30 minutes later
avg_dist_Hail = 5.2                    # 5.2 km away
prob_same_county_Hail = 0.95           # 95% in same county
```

**Rationale:**
- **Probability features**: Different event types have VERY different cascade tendencies
- **Time gap features**: Cascades have distinct temporal patterns (immediate vs delayed)
- **Distance features**: Cascades spread differently (localized vs widespread)
- **Same county features**: Administrative boundaries matter for emergency response
- All learned from historical cascade_pairs.csv (training data only)

---