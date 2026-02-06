# Cascade Definition

## Overview

A **cascade** is when one weather event (primary) triggers or causes a secondary event. This project uses domain-constrained cascade identification based on documented causal relationships from peer-reviewed research and government agencies.

## Problem Framing

**Given event X, predict what types of secondary events will occur (multilabel classification).**

- **Input**: A primary weather event with its attributes (type, location, time, magnitude, etc.)
- **Output**: Zero or more secondary event types that this event may trigger

---

## Cascade Configuration

The `CascadeConfig` class in `src/cascade_definition.py` controls how cascades are identified:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temporal_window_days` | 7 | Max days between primary end and secondary start |
| `spatial_proximity` | SAME_COUNTY | Spatial matching (county, state, or distance) |
| `distance_threshold_km` | 50.0 | Radius if using distance-based matching |
| `require_same_episode` | True | Events must share NOAA episode ID |
| `require_different_event_types` | True | Primary ≠ secondary type |
| `use_domain_patterns` | True | Only allow documented causal relationships |

### Usage

```python
from src.cascade_definition import get_conservative_config, CascadeConfig

# Conservative config (recommended)
config = get_conservative_config()
# temporal_window_days=7, require_same_episode=True, use_domain_patterns=True

# Custom config
config = CascadeConfig(
    temporal_window_days=14,
    spatial_proximity=SpatialProximity.SAME_STATE,
    use_domain_patterns=True
)
```

---

## Documented Causal Patterns

All patterns are derived from peer-reviewed literature and government sources. The system recognizes **100+ documented causal pairs** across **35+ primary event types**.

### Tropical Systems
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Hurricane (Typhoon) | Flash Flood, Flood, Storm Surge/Tide, Coastal Flood, High Wind, Tornado, Rip Current, High Surf, Heavy Rain, Debris Flow | NOAA NWS, FEMA |
| Tropical Storm | Flash Flood, Flood, Coastal Flood, High Wind, Tornado, Heavy Rain, Rip Current, High Surf, Debris Flow | NOAA NWS |
| Tropical Depression | Flash Flood, Flood, Heavy Rain, High Surf | NOAA NWS |

### Severe Thunderstorms  
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Thunderstorm Wind | Tornado, Flash Flood, Hail, Lightning, Debris Flow, Dust Storm | NOAA NSSL |
| Tornado | Flash Flood, Hail, Debris Flow | NOAA NSSL |
| Hail | Flash Flood, Lightning, Thunderstorm Wind | NOAA NSSL |
| Lightning | Wildfire, Flash Flood | NOAA NSSL |
| Funnel Cloud | Tornado, Hail, Thunderstorm Wind | NOAA NSSL |

### Wildfire Cascades
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Wildfire | Debris Flow, Flash Flood, Flood, Dust Storm, Dense Smoke | USGS, NOAA PSL |

> **Key insight**: Wildfires alter soil hydrophobicity, causing debris flows and flash floods for up to 10 years after the fire (Cannon et al., 2008).

### Flood-Induced Mass Movements
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Flash Flood | Debris Flow, Landslide, Flood | USGS |
| Flood | Debris Flow, Landslide, Lakeshore Flood | USGS |
| Heavy Rain | Flash Flood, Flood, Debris Flow, Landslide | NOAA NWS |

### Winter Weather
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Winter Storm | Blizzard, Heavy Snow, Ice Storm, Avalanche, Frost/Freeze, Cold/Wind Chill, Flash Flood | NOAA NWS |
| Blizzard | Avalanche, Frost/Freeze, Extreme Cold/Wind Chill, Heavy Snow | NOAA NWS, CAIC |
| Heavy Snow | Avalanche, Blizzard, Flash Flood, Flood, Debris Flow | NOAA NWS |
| Ice Storm | Winter Weather, Heavy Snow, Frost/Freeze, Flash Flood | NOAA NWS |
| Avalanche | Flash Flood, Debris Flow, Flood | Colorado Avalanche Center |

### Coastal & Marine
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Storm Surge/Tide | Coastal Flood, Flash Flood, High Surf, Rip Current | NOAA NHC |
| High Surf | Coastal Flood, Rip Current | NOAA NWS |
| Tsunami | Coastal Flood, Flash Flood, Debris Flow | USGS |

### Heat & Drought
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| Drought | Wildfire, Heat, Excessive Heat, Dust Storm | NOAA Climate.gov |
| Excessive Heat | Heat, Wildfire, Drought | NOAA NWS |

### Wind Events
| Primary | Secondary Events | Source |
|---------|-----------------|--------|
| High Wind | Dust Storm, Wildfire, Debris Flow | NOAA NWS |
| Strong Wind | Dust Storm, Dust Devil | NOAA NWS |

---

## References

| Code | Source | URL |
|------|--------|-----|
| [1] | USGS Landslide Hazards | https://www.usgs.gov/programs/landslide-hazards |
| [2] | NOAA PSL Wildfire Impacts | https://psl.noaa.gov/special_analyses/wildfires/ |
| [3] | NOAA Compound Flooding | https://www.noaa.gov/compound-flooding |
| [4] | FEMA National Risk Index | https://hazards.fema.gov/nri/ |
| [5] | NOAA NSSL Severe Storms | https://www.nssl.noaa.gov/ |
| [6] | Cannon et al. (2008) | Geomorphology 96(3-4): 171-188 |
| [7] | NOAA NWS Tropical Cyclone Hazards | https://www.weather.gov/safety/hurricane-hazards |
| [8] | USGS Earthquake-Induced Landslides | https://www.usgs.gov/programs/earthquake-hazards |
| [9] | NOAA NWS Winter Weather Hazards | https://www.weather.gov/safety/winter |
| [10] | Colorado Avalanche Center | https://avalanche.state.co.us/ |
| [11] | NOAA Climate.gov | https://www.climate.gov/ |
| [12] | NOAA NWS Storm Surge | https://www.nhc.noaa.gov/surge/ |
| [13] | NOAA Rip Current Science | https://www.weather.gov/safety/ripcurrent-science |
| [14] | Gill & Malamud (2014) | Reviews of Geophysics 52(4): 680-722 |
| [15] | NOAA NSSL Tornado Research | https://www.nssl.noaa.gov/research/tornadoes/ |
| [16] | NWS Instruction 10-1605 | https://www.nws.noaa.gov/directives/ |
| [17] | IPCC AR6 Chapter 11 | Climate Extreme Events (2021) |
| [18] | AghaKouchak et al. (2020) | Annual Rev. Earth Planet. Sci. 48: 519-548 |

---

## NOAA Event Types

The system supports all 48 NOAA Storm Event types (from NWS Instruction 10-1605):

```
Astronomical Low Tide, Avalanche, Blizzard, Coastal Flood, Cold/Wind Chill,
Debris Flow, Dense Fog, Dense Smoke, Drought, Dust Devil, Dust Storm,
Excessive Heat, Extreme Cold/Wind Chill, Flash Flood, Flood, Freezing Fog,
Frost/Freeze, Funnel Cloud, Hail, Heat, Heavy Rain, Heavy Snow, High Surf,
High Wind, Hurricane (Typhoon), Ice Storm, Lake-Effect Snow, Lakeshore Flood,
Lightning, Marine Hail, Marine High Wind, Marine Strong Wind,
Marine Thunderstorm Wind, Rip Current, Seiche, Sleet, Storm Surge/Tide,
Strong Wind, Thunderstorm Wind, Tornado, Tropical Depression, Tropical Storm,
Tsunami, Volcanic Ash, Waterspout, Wildfire, Winter Storm, Winter Weather
```

---

## API Reference

### Functions

```python
from src.cascade_definition import (
    get_conservative_config,  # Returns CascadeConfig with strict defaults
    get_valid_cascade_pairs,  # Returns list of (primary, secondary) tuples
    is_valid_cascade_pair,    # Check if a pair is documented
    get_all_event_types,      # All 48 NOAA event types
)
```

### Example

```python
from src.cascade_definition import is_valid_cascade_pair, get_valid_cascade_pairs

# Check specific pairs
is_valid_cascade_pair("Hurricane (Typhoon)", "Flash Flood")  # True
is_valid_cascade_pair("Hail", "Wildfire")                    # False

# Get all valid pairs
pairs = get_valid_cascade_pairs()
print(f"Total documented cascade pairs: {len(pairs)}")
# Output: Total documented cascade pairs: 100+
```
