"""
Cascade Definition Module

Defines configuration and domain-based causal patterns for cascade identification.
Causal patterns are derived from peer-reviewed research and government agency sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple


class SpatialProximity(Enum):
    """Spatial proximity constraints for cascade identification."""
    SAME_COUNTY = "same_county"  # CZ_FIPS match
    SAME_STATE = "same_state"    # STATE match only
    DISTANCE_KM = "distance_km"  # Within radius (requires lat/lon)


@dataclass
class CascadeConfig:
    """Configuration for cascade identification."""
    
    # Temporal constraint: max days between primary end and secondary start
    temporal_window_days: int = 7
    
    # Spatial constraint
    spatial_proximity: SpatialProximity = SpatialProximity.SAME_COUNTY
    distance_threshold_km: float = 50.0  # Only used if spatial_proximity = DISTANCE_KM
    
    # Episode constraint: require same NOAA episode
    require_same_episode: bool = True
    
    # Event type constraint
    require_different_event_types: bool = True
    
    # Use domain patterns to filter (recommended for causal cascades)
    use_domain_patterns: bool = True


# =============================================================================
# DOCUMENTED CAUSAL CASCADE PATTERNS
# =============================================================================
#
# These patterns are derived from peer-reviewed literature and government sources.
# Each entry is (primary_event_type, secondary_event_type) as they appear in NOAA data.
#
# References:
# [1] USGS - Landslide Hazards Program: Post-fire debris flows
#     https://www.usgs.gov/programs/landslide-hazards/science/postfire-debris-flow-hazards
# [2] NOAA PSL - Wildfire and Post-Fire Impacts
#     https://psl.noaa.gov/special_analyses/wildfires/
# [3] NOAA - Compound Flooding Research
#     https://www.noaa.gov/compound-flooding
# [4] FEMA - National Risk Index
#     https://hazards.fema.gov/nri/
# [5] NOAA NSSL - Severe Storms Laboratory
#     https://www.nssl.noaa.gov/
# [6] Cannon et al. (2008) "Wildfire-related debris flow initiation processes"
#     Geomorphology 96(3-4): 171-188
# [7] NOAA NWS - Tropical Cyclone Hazards
#     https://www.weather.gov/safety/hurricane-hazards
# [8] USGS - Earthquake-Induced Landslides
#     https://www.usgs.gov/programs/earthquake-hazards/earthquake-induced-landslides
# [9] NOAA NWS - Winter Weather Hazards
#     https://www.weather.gov/safety/winter
# [10] Colorado Avalanche Information Center
#      https://avalanche.state.co.us/
# [11] NOAA Climate.gov - Drought-Wildfire-Heat Connection
#      https://www.climate.gov/
# [12] NOAA NWS - Storm Surge Overview
#      https://www.nhc.noaa.gov/surge/
# [13] NOAA NWS - Rip Current Science
#      https://www.weather.gov/safety/ripcurrent-science
# [14] Gill & Malamud (2014) "Reviewing and visualizing the interactions of natural hazards"
#      Reviews of Geophysics 52(4): 680-722
# [15] NOAA NSSL - Supercell and Tornado Research
#      https://www.nssl.noaa.gov/research/tornadoes/
# [16] NWS Instruction 10-1605: Storm Data Preparation
#      https://www.nws.noaa.gov/directives/
# [17] Seneviratne et al. (2021) "Weather and Climate Extreme Events in a Changing Climate"
#      IPCC AR6, Chapter 11
# [18] AghaKouchak et al. (2020) "Climate Extremes and Compound Hazards in a Warming World"
#      Annual Review of Earth and Planetary Sciences 48: 519-548


# =============================================================================
# COMPLETE NOAA STORM EVENT TYPES (NWS Instruction 10-1605)
# =============================================================================
# The following 48 event types have been in use since 1996:
#
# Astronomical Low Tide, Avalanche, Blizzard, Coastal Flood, Cold/Wind Chill,
# Debris Flow, Dense Fog, Dense Smoke, Drought, Dust Devil, Dust Storm,
# Excessive Heat, Extreme Cold/Wind Chill, Flash Flood, Flood, Freezing Fog,
# Frost/Freeze, Funnel Cloud, Hail, Heat, Heavy Rain, Heavy Snow, High Surf,
# High Wind, Hurricane (Typhoon), Ice Storm, Lake-Effect Snow, Lakeshore Flood,
# Lightning, Marine Hail, Marine High Wind, Marine Strong Wind,
# Marine Thunderstorm Wind, Rip Current, Seiche, Sleet, Storm Surge/Tide,
# Strong Wind, Thunderstorm Wind, Tornado, Tropical Depression, Tropical Storm,
# Tsunami, Volcanic Ash, Waterspout, Wildfire, Winter Storm, Winter Weather
# =============================================================================


# Primary patterns with strong causal mechanisms
CAUSAL_CASCADE_PATTERNS: Dict[str, List[str]] = {
    
    # =========================================================================
    # TROPICAL SYSTEMS
    # =========================================================================
    # Hurricanes cause flooding through heavy precipitation and storm surge [4, 7, 12]
    # Storm surge is the greatest threat to life in hurricanes [12]
    # Hurricanes spawn tornadoes in outer rain bands, especially right-front quadrant [5, 7, 15]
    
    "Hurricane (Typhoon)": [
        "Flash Flood",          # Heavy rainfall causes rapid flooding [7]
        "Flood",                # Prolonged rainfall leads to river flooding [7]
        "Storm Surge/Tide",     # Wind-driven water rise, up to 30+ feet [12]
        "Coastal Flood",        # Combined surge and tides inundate coast [12]
        "High Wind",            # Eye wall and rainband winds [7]
        "Tornado",              # Spawned in spiral rainbands [5, 7]
        "Rip Current",          # High surf from distant storms [13]
        "High Surf",            # Large waves from storm approach [12]
        "Heavy Rain",           # Often 6+ inches, can persist for days [7]
        "Debris Flow",          # Post-storm rainfall on saturated slopes [1]
        "Landslide",            # Rainfall-saturated slopes fail [14]
    ],
    
    "Tropical Storm": [
        "Flash Flood",          # Intense rainfall bands [7]
        "Flood",                # River and stream flooding from precipitation [7]
        "Coastal Flood",        # Moderate storm surge [12]
        "High Wind",            # Sustained 39-73 mph winds [7]
        "Tornado",              # Weak tornadoes in rainbands [5]
        "Heavy Rain",           # Prolific rain producer [7]
        "Rip Current",          # Enhanced wave action [13]
        "High Surf",            # Elevated seas from tropical system [12]
        "Debris Flow",          # Post-rainfall mass movement [1]
    ],
    
    "Tropical Depression": [
        "Flash Flood",          # Heavy tropical rainfall [7]
        "Flood",                # Slow-moving systems produce prolonged rain [7]
        "Heavy Rain",           # Primary hazard of depressions [7]
        "High Surf",            # Elevated seas during approach [12]
    ],
    
    # =========================================================================
    # STORM SURGE AND COASTAL HAZARDS
    # =========================================================================
    # Storm surge directly causes coastal flooding [12]
    # High surf and surge combined cause extensive coastal erosion [12]
    
    "Storm Surge/Tide": [
        "Coastal Flood",        # Direct inundation from surge [12]
        "Flash Flood",          # Rapid coastal inundation [12]
        "High Surf",            # Enhanced wave heights ride on surge [12]
        "Rip Current",          # Post-surge wave conditions [13]
    ],
    
    "High Surf": [
        "Coastal Flood",        # Wave overtopping of barriers [12]
        "Rip Current",          # Breaking waves create return flow [13]
    ],
    
    "Seiche": [
        "Coastal Flood",        # Standing wave inundation in enclosed basins [16]
        "Lakeshore Flood",      # Lake-level oscillations [16]
    ],
    
    "Tsunami": [
        "Coastal Flood",        # Seismic sea wave inundation [8]
        "Flash Flood",          # Rapid inland surge [8]
        "Debris Flow",          # Water entrains coastal debris [14]
    ],
    
    # =========================================================================
    # SEVERE THUNDERSTORM SYSTEMS
    # =========================================================================
    # Supercell thunderstorms produce tornadoes, hail, and high winds [5, 15]
    # NSSL research shows TORFFs (tornado + flash flood) are common [5]
    
    "Thunderstorm Wind": [
        "Tornado",              # Supercell mesocyclone evolution [5, 15]
        "Flash Flood",          # Associated heavy rainfall [5]
        "Hail",                 # Common co-occurrence in supercells [5]
        "Lightning",            # Intrinsic to thunderstorms [5]
        "Debris Flow",          # Post-storm rainfall triggering [1]
        "Dust Storm",           # Outflow winds mobilize dust [14]
    ],
    
    "Tornado": [
        "Flash Flood",          # Parent supercell produces heavy rain [5]
        "Hail",                 # Co-occurrence in severe storms [5]
        "Debris Flow",          # Ground disturbance + rainfall [14]
    ],
    
    "Hail": [
        "Flash Flood",          # Severe storm precipitation [5]
        "Lightning",            # Severe thunderstorm activity [5]
        "Thunderstorm Wind",    # Severe storm downdrafts [5]
    ],
    
    "Lightning": [
        "Wildfire",             # Dry lightning ignition [2, 11]
        "Flash Flood",          # Thunderstorm rainfall [5]
    ],
    
    "Funnel Cloud": [
        "Tornado",              # Funnel reaching ground [15]
        "Hail",                 # Parent storm activity [5]
        "Thunderstorm Wind",    # Associated storm [5]
    ],
    
    "Waterspout": [
        "Tornado",              # Waterspout moving onshore [15]
        "Thunderstorm Wind",    # Associated storm activity [5]
        "Marine Thunderstorm Wind",  # Marine storm environment [5]
    ],
    
    # =========================================================================
    # MARINE EVENTS
    # =========================================================================
    # Marine events can transition to land-based hazards [16]
    
    "Marine Thunderstorm Wind": [
        "Waterspout",           # Rotating updraft over water [15]
        "Marine Hail",          # Marine severe thunderstorm [5]
        "Thunderstorm Wind",    # Storm moving ashore [5]
        "Flash Flood",          # Coastal heavy rain [5]
    ],
    
    "Marine High Wind": [
        "Coastal Flood",        # Wind-driven water [12]
        "High Surf",            # Wind-generated waves [12]
        "High Wind",            # Winds affecting coast [16]
    ],
    
    "Marine Strong Wind": [
        "High Surf",            # Wave generation [12]
        "Strong Wind",          # Coastal wind effects [16]
    ],
    
    "Marine Hail": [
        "Hail",                 # Storm moving ashore [5]
        "Marine Thunderstorm Wind",  # Associated storm [5]
    ],
    
    # =========================================================================
    # WILDFIRE CASCADES
    # =========================================================================
    # Wildfires alter soil hydrophobicity, causing debris flows and flash floods [1, 2, 6]
    # Most post-fire debris flows occur within 1-2 years, some up to 10 years [1]
    # Exposed soil after fire is susceptible to wind erosion [2]
    
    "Wildfire": [
        "Debris Flow",          # Post-fire hydrophobic soil + rain [1, 6]
        "Flash Flood",          # Reduced infiltration post-fire [1, 2]
        "Flood",                # Altered watershed response [1]
        "Landslide",            # Loss of root cohesion [1, 6]
        "Dust Storm",           # Exposed soil after fire [2]
        "Dense Smoke",          # Combustion products [2]
    ],
    
    # =========================================================================
    # FLOOD-INDUCED MASS MOVEMENTS
    # =========================================================================
    # Heavy precipitation saturates slopes, triggering landslides and debris flows [1, 3, 14]
    # Flash floods can escalate to sustained flooding [3]
    
    "Flash Flood": [
        "Debris Flow",          # Water-saturated slope failure [1, 14]
        "Landslide",            # Rapid saturation of slopes [14]
        "Flood",                # Flash floods can become riverine floods [3]
    ],
    
    "Flood": [
        "Debris Flow",          # Prolonged saturation of slopes [14]
        "Landslide",            # Undermining and saturation [14]
        "Lakeshore Flood",      # Lake level rise [16]
    ],
    
    "Heavy Rain": [
        "Flash Flood",          # Rapid runoff [3]
        "Flood",                # Sustained precipitation [3]
        "Debris Flow",          # Slope saturation [1, 14]
        "Landslide",            # Rainfall-induced failures [14]
    ],
    
    "Lakeshore Flood": [
        "Coastal Flood",        # Great Lakes coastal impacts [16]
        "Flash Flood",          # Rapid shoreline inundation [16]
    ],
    
    # =========================================================================
    # WINTER WEATHER CASCADES
    # =========================================================================
    # Ice accumulation causes infrastructure damage [4, 9]
    # Heavy snowfall increases avalanche risk significantly [9, 10]
    # Rapid snowmelt can cause flooding [9]
    
    "Winter Storm": [
        "Blizzard",             # Heavy snow + strong winds [9]
        "Heavy Snow",           # Primary precipitation type [9]
        "Ice Storm",            # Freezing rain component [9]
        "Avalanche",            # Snow loading on slopes [10]
        "Frost/Freeze",         # Cold air mass [9]
        "Cold/Wind Chill",      # Associated cold temperatures [9]
        "Sleet",                # Ice pellet precipitation [9]
        "Flash Flood",          # Rain-on-snow events [9]
        "Flood",                # Rapid snowmelt [9]
    ],
    
    "Blizzard": [
        "Avalanche",            # Heavy snow loading + wind slab [10]
        "Frost/Freeze",         # Associated cold [9]
        "Extreme Cold/Wind Chill",  # High winds + cold [9]
        "Heavy Snow",           # Definition component [9]
        "Lake-Effect Snow",     # Enhanced by Great Lakes [9]
    ],
    
    "Heavy Snow": [
        "Avalanche",            # New snow instability [10]
        "Blizzard",             # Combined with high winds [9]
        "Flash Flood",          # Rapid melt [9]
        "Flood",                # Seasonal melt [9]
        "Debris Flow",          # Saturated slopes during melt [14]
    ],
    
    "Ice Storm": [
        "Winter Weather",       # Mixed precipitation [9]
        "Heavy Snow",           # Transition to snow [9]
        "Frost/Freeze",         # Cold air mass [9]
        "Flash Flood",          # Ice dam breaks [9]
    ],
    
    "Sleet": [
        "Ice Storm",            # Ice accumulation [9]
        "Winter Weather",       # Mixed winter precipitation [9]
        "Freezing Fog",         # Cold, moist conditions [9]
    ],
    
    "Lake-Effect Snow": [
        "Blizzard",             # Extreme accumulation + wind [9]
        "Heavy Snow",           # Localized heavy snow [9]
        "Avalanche",            # In lake-effect snow belts with terrain [10]
    ],
    
    "Frost/Freeze": [
        "Cold/Wind Chill",      # Continued cold [9]
        "Extreme Cold/Wind Chill",  # Severe cold events [9]
    ],
    
    "Cold/Wind Chill": [
        "Frost/Freeze",         # Cold temperature effects [9]
        "Extreme Cold/Wind Chill",  # Intensifying cold [9]
    ],
    
    "Extreme Cold/Wind Chill": [
        "Frost/Freeze",         # Prolonged cold [9]
        "Avalanche",            # Cold snow weak layers [10]
    ],
    
    "Freezing Fog": [
        "Ice Storm",            # Rime ice accumulation [9]
        "Frost/Freeze",         # Cold conditions [9]
    ],
    
    "Avalanche": [
        "Flash Flood",          # Rapid meltwater from avalanche debris [10]
        "Debris Flow",          # Avalanche-triggered mass movement [10, 14]
        "Flood",                # Spring melt of avalanche deposits [10]
    ],
    
    # =========================================================================
    # WIND EVENTS
    # =========================================================================
    # High winds mobilize dust and debris, spread fires [2, 14]
    # Strong winds can cause power outages leading to indirect impacts
    
    "High Wind": [
        "Dust Storm",           # Wind-eroded particulates [14]
        "Wildfire",             # Wind spreads existing fires [2]
        "Debris Flow",          # Post-wind events with rain [14]
    ],
    
    "Strong Wind": [
        "Dust Storm",           # Moderate dust mobilization [14]
        "Dust Devil",           # Localized vortex formation [16]
    ],
    
    "Dust Storm": [
        "Dense Smoke",          # Visibility obscuration [16]
        "Dense Fog",            # Particle-seeded fog [14]
    ],
    
    "Dust Devil": [
        "Dust Storm",           # Multiple dust devils [16]
    ],
    
    # =========================================================================
    # TEMPERATURE EXTREMES
    # =========================================================================
    # Drought-heat-wildfire connection is well-documented [11, 17, 18]
    # Heat waves accelerate fuel drying, increasing fire risk [11, 17]
    # These are compound hazards that amplify each other [18]
    
    "Drought": [
        "Wildfire",             # Dry fuels ignite easily [11, 17]
        "Heat",                 # Feedback loop with drying [11]
        "Excessive Heat",       # Amplified by dry conditions [17]
        "Dust Storm",           # Dry soil mobilized by wind [14]
        "Dense Smoke",          # Wildfires during drought [2]
    ],
    
    "Heat": [
        "Drought",              # Increased evapotranspiration [11, 17]
        "Wildfire",             # Fuel desiccation [11]
        "Excessive Heat",       # Heat wave intensification [17]
    ],
    
    "Excessive Heat": [
        "Drought",              # Evaporative demand [11, 17]
        "Wildfire",             # Critical fire weather [11]
        "Heat",                 # Sustained high temperatures [17]
    ],
    
    # =========================================================================
    # SEISMIC EVENTS
    # =========================================================================
    # Earthquake-landslide connection is well-documented [8, 14]
    # Ground shaking destabilizes slopes, especially saturated ones [8]
    
    "Earthquake": [
        "Landslide",            # Shaking-induced slope failure [8]
        "Tsunami",              # Submarine earthquake displacement [8]
        "Debris Flow",          # Mobilized slope material [8, 14]
        "Flash Flood",          # Dam failures, landslide dams [8]
        "Avalanche",            # Shaking-triggered snow release [10]
    ],
    
    # =========================================================================
    # VISIBILITY HAZARDS
    # =========================================================================
    # Secondary hazards from visibility events are primarily indirect
    
    "Dense Fog": [
        "Freezing Fog",         # Fog + cold temperatures [9]
    ],
    
    "Dense Smoke": [
        "Wildfire",             # Source of smoke [2]
    ],
    
    # =========================================================================
    # VOLCANIC EVENTS
    # =========================================================================
    # Volcanic activity produces multiple cascade hazards [14]
    
    "Volcanic Ash": [
        "Flash Flood",          # Lahars from ash + precipitation [14]
        "Debris Flow",          # Mobilized volcanic material [14]
        "Dense Smoke",          # Combined with gas emissions [14]
    ],
    
    # =========================================================================
    # TIDAL EVENTS
    # =========================================================================
    
    "Astronomical Low Tide": [
        # Low tides generally don't cause cascading hazards
        # but can expose seabed, affect navigation
    ],
    
    # =========================================================================
    # MASS MOVEMENT EVENTS
    # =========================================================================
    # Debris flows and landslides can trigger secondary hazards [14]
    
    "Debris Flow": [
        "Flood",                # Debris-dammed rivers [14]
        "Flash Flood",          # Dam breach flooding [14]
    ],
    
    "Landslide": [
        "Flood",                # Landslide-dammed rivers [14]
        "Flash Flood",          # Dam breach flooding [14]
        "Debris Flow",          # Mobilized debris [14]
    ],
}


def get_valid_cascade_pairs() -> Set[Tuple[str, str]]:
    """
    Get set of all valid (primary, secondary) pairs from domain knowledge.
    
    Returns:
        Set of tuples representing valid causal cascade pairs
    """
    pairs = set()
    for primary, secondaries in CAUSAL_CASCADE_PATTERNS.items():
        for secondary in secondaries:
            pairs.add((primary, secondary))
    return pairs


def is_valid_cascade(primary_type: str, secondary_type: str) -> bool:
    """
    Check if a primary→secondary pair is a documented causal cascade.
    
    Args:
        primary_type: Primary event type (e.g., "Hurricane (Typhoon)")
        secondary_type: Secondary event type (e.g., "Flood")
        
    Returns:
        True if this is a valid causal cascade, False otherwise
    """
    secondaries = CAUSAL_CASCADE_PATTERNS.get(primary_type, [])
    return secondary_type in secondaries


def get_all_event_types() -> Set[str]:
    """
    Get all NOAA event types that appear in the cascade patterns.
    
    Returns:
        Set of all event type strings
    """
    event_types = set(CAUSAL_CASCADE_PATTERNS.keys())
    for secondaries in CAUSAL_CASCADE_PATTERNS.values():
        event_types.update(secondaries)
    return event_types


def get_primary_event_types() -> Set[str]:
    """
    Get all event types that can be primary triggers.
    
    Returns:
        Set of primary event type strings
    """
    return set(CAUSAL_CASCADE_PATTERNS.keys())


def get_secondary_events_for(primary_type: str) -> List[str]:
    """
    Get all valid secondary events for a given primary event type.
    
    Args:
        primary_type: Primary event type
        
    Returns:
        List of valid secondary event types, empty if not a valid primary
    """
    return CAUSAL_CASCADE_PATTERNS.get(primary_type, [])


def get_conservative_config() -> CascadeConfig:
    """
    Get conservative (strict) cascade configuration.
    
    Uses domain patterns to ensure only causal cascades are identified.
    """
    return CascadeConfig(
        temporal_window_days=7,
        spatial_proximity=SpatialProximity.SAME_COUNTY,
        require_same_episode=True,
        require_different_event_types=True,
        use_domain_patterns=True,
    )


def get_exploratory_config() -> CascadeConfig:
    """
    Get exploratory (relaxed) config without domain pattern filtering.
    
    Use for EDA to discover potential new cascade patterns.
    """
    return CascadeConfig(
        temporal_window_days=7,
        spatial_proximity=SpatialProximity.SAME_COUNTY,
        require_same_episode=True,
        require_different_event_types=True,
        use_domain_patterns=False,
    )


def print_cascade_patterns():
    """Print all documented cascade patterns for reference."""
    print("DOCUMENTED CAUSAL CASCADE PATTERNS")
    print("=" * 60)
    
    total_pairs = 0
    for primary, secondaries in sorted(CAUSAL_CASCADE_PATTERNS.items()):
        if secondaries:  # Only print if there are secondary events
            print(f"\n{primary} →")
            for sec in secondaries:
                print(f"    {sec}")
                total_pairs += 1
    
    primaries = len([k for k, v in CAUSAL_CASCADE_PATTERNS.items() if v])
    print(f"\n{'=' * 60}")
    print(f"Total: {primaries} primary types, {total_pairs} causal pairs")


def print_reference_summary():
    """Print a summary of references used for causal patterns."""
    print("REFERENCES FOR CAUSAL CASCADE PATTERNS")
    print("=" * 60)
    references = """
    [1] USGS Landslide Hazards Program: Post-fire debris flows
    [2] NOAA PSL: Wildfire and Post-Fire Impacts
    [3] NOAA Compound Flooding Research
    [4] FEMA National Risk Index
    [5] NOAA NSSL: Severe Storms Laboratory
    [6] Cannon et al. (2008) Geomorphology: Post-fire debris flows
    [7] NOAA NWS: Tropical Cyclone Hazards
    [8] USGS: Earthquake-Induced Landslides
    [9] NOAA NWS: Winter Weather Hazards
    [10] Colorado Avalanche Information Center
    [11] NOAA Climate.gov: Drought-Wildfire-Heat Connection
    [12] NOAA NWS: Storm Surge Overview
    [13] NOAA NWS: Rip Current Science
    [14] Gill & Malamud (2014) Reviews of Geophysics: Natural hazard interactions
    [15] NOAA NSSL: Supercell and Tornado Research
    [16] NWS Instruction 10-1605: Storm Data Preparation
    [17] IPCC AR6 Chapter 11: Extreme Events
    [18] AghaKouchak et al. (2020) Annual Review: Compound Hazards
    """
    print(references)
