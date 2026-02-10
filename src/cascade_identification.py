"""
Cascade Identification Module

Identifies cascade sequences in storm event data based on temporal,
spatial, and causal domain constraints.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import timedelta
from dataclasses import dataclass

from .cascade_definition import (
    CascadeConfig, 
    SpatialProximity,
    is_valid_cascade,
    get_valid_cascade_pairs,
    get_conservative_config
)


@dataclass
class CascadePair:
    """Represents a primary-secondary event cascade pair."""
    primary_event_id: int
    secondary_event_id: int
    primary_event_type: str
    secondary_event_type: str
    time_gap_hours: float
    same_county: bool
    same_episode: bool
    distance_km: Optional[float] = None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def check_spatial_proximity(
    event1: pd.Series, 
    event2: pd.Series,
    config: CascadeConfig
) -> Tuple[bool, Optional[float]]:
    """Check if two events satisfy spatial proximity constraints."""
    
    if config.spatial_proximity == SpatialProximity.SAME_COUNTY:
        same_county = (
            event1['STATE_FIPS'] == event2['STATE_FIPS'] and
            event1['CZ_FIPS'] == event2['CZ_FIPS']
        )
        return same_county, None
    
    elif config.spatial_proximity == SpatialProximity.SAME_STATE:
        same_state = event1['STATE_FIPS'] == event2['STATE_FIPS']
        return same_state, None
    
    elif config.spatial_proximity == SpatialProximity.DISTANCE_KM:
        if pd.isna(event1.get('BEGIN_LAT')) or pd.isna(event2.get('BEGIN_LAT')):
            # Fall back to same county if no coordinates
            same_county = (
                event1['STATE_FIPS'] == event2['STATE_FIPS'] and
                event1['CZ_FIPS'] == event2['CZ_FIPS']
            )
            return same_county, None
        
        distance = haversine_distance(
            event1['BEGIN_LAT'], event1['BEGIN_LON'],
            event2['BEGIN_LAT'], event2['BEGIN_LON']
        )
        return distance <= config.distance_threshold_km, distance
    
    return False, None


def _get_spatial_group_key(config: CascadeConfig) -> list:
    """Get the columns to group by based on spatial proximity config."""
    if config.spatial_proximity == SpatialProximity.SAME_COUNTY:
        return ['STATE_FIPS', 'CZ_FIPS']
    elif config.spatial_proximity == SpatialProximity.SAME_STATE:
        return ['STATE_FIPS']
    else:
        # For distance-based, group by county as a reasonable approximation
        return ['STATE_FIPS', 'CZ_FIPS']


def _identify_cascades_in_group(
    group_df: pd.DataFrame,
    temporal_window: timedelta,
    config: CascadeConfig,
    valid_pairs: set,
) -> List[CascadePair]:
    """
    Identify cascades within a single spatial/episode group using sorted scan.
    
    Uses numpy arrays for fast column access and breaks early when the
    temporal window is exceeded.
    """
    if len(group_df) < 2:
        return []

    group_df = group_df.sort_values('BEGIN_DATETIME')

    # Extract columns into arrays for fast indexed access
    begin_dt = group_df['BEGIN_DATETIME'].values  # numpy datetime64
    end_dt = group_df['END_DATETIME'].values
    event_types = group_df['EVENT_TYPE'].values
    event_ids = group_df['EVENT_ID'].values
    state_fips = group_df['STATE_FIPS'].values
    cz_fips = group_df['CZ_FIPS'].values
    episode_ids = group_df['EPISODE_ID'].values

    has_coords = 'BEGIN_LAT' in group_df.columns
    if has_coords:
        begin_lats = group_df['BEGIN_LAT'].values
        begin_lons = group_df['BEGIN_LON'].values

    tw_ns = np.timedelta64(int(temporal_window.total_seconds() * 1e9), 'ns')
    zero_td = np.timedelta64(0, 'ns')

    cascades = []
    n = len(group_df)

    for i in range(n):
        end_i = end_dt[i]
        etype_i = event_types[i]

        for j in range(i + 1, n):
            time_gap = begin_dt[j] - end_i

            # Since sorted by begin_datetime, once begin[j] is too far
            # past end[i], all subsequent j will also be too far
            if time_gap > tw_ns:
                break

            if time_gap < zero_td:
                continue

            etype_j = event_types[j]

            # Check different event types
            if config.require_different_event_types and etype_i == etype_j:
                continue

            # Check domain patterns via pre-built set (O(1))
            if config.use_domain_patterns:
                if (etype_i, etype_j) not in valid_pairs:
                    continue

            # Spatial proximity (for distance-based mode within a county group)
            if config.spatial_proximity == SpatialProximity.DISTANCE_KM:
                if has_coords and not (np.isnan(begin_lats[i]) or np.isnan(begin_lats[j])):
                    dist = haversine_distance(
                        begin_lats[i], begin_lons[i],
                        begin_lats[j], begin_lons[j]
                    )
                    if dist > config.distance_threshold_km:
                        continue
                    distance = dist
                else:
                    # Fall back to same county check
                    if state_fips[i] != state_fips[j] or cz_fips[i] != cz_fips[j]:
                        continue
                    distance = None
            else:
                distance = None

            gap_hours = time_gap / np.timedelta64(1, 'h')

            cascades.append(CascadePair(
                primary_event_id=int(event_ids[i]),
                secondary_event_id=int(event_ids[j]),
                primary_event_type=etype_i,
                secondary_event_type=etype_j,
                time_gap_hours=float(gap_hours),
                same_county=(state_fips[i] == state_fips[j] and cz_fips[i] == cz_fips[j]),
                same_episode=(episode_ids[i] == episode_ids[j]),
                distance_km=distance
            ))

    return cascades


def identify_cascades(
    df: pd.DataFrame,
    config: Optional[CascadeConfig] = None,
    verbose: bool = True
) -> List[CascadePair]:
    """
    Identify cascade pairs in storm event data.
    
    Args:
        df: DataFrame with storm events (must have BEGIN_DATETIME, END_DATETIME)
        config: Cascade configuration (uses conservative default if None)
        verbose: Print progress information
        
    Returns:
        List of CascadePair objects
    """
    if config is None:
        config = get_conservative_config()
    
    if verbose:
        print(f"Config: temporal={config.temporal_window_days}d, "
              f"spatial={config.spatial_proximity.value}, "
              f"same_episode={config.require_same_episode}, "
              f"use_domain_patterns={config.use_domain_patterns}")
    
    # Validate required columns
    if 'BEGIN_DATETIME' not in df.columns or 'END_DATETIME' not in df.columns:
        raise ValueError("DataFrame must have BEGIN_DATETIME and END_DATETIME columns")
    
    # Remove events with missing datetimes
    df_valid = df.dropna(subset=['BEGIN_DATETIME', 'END_DATETIME']).copy()
    
    if verbose:
        print(f"Processing {len(df_valid):,} events...")
    
    temporal_window = timedelta(days=config.temporal_window_days)

    # Pre-build valid pairs set for O(1) lookup
    if config.use_domain_patterns:
        valid_pairs = get_valid_cascade_pairs()
        if verbose:
            print(f"Using {len(valid_pairs)} domain-validated cascade patterns")
    else:
        valid_pairs = set()

    # Determine grouping strategy
    if config.require_same_episode:
        # Original behavior: group by episode
        groups = df_valid.groupby('EPISODE_ID')
        group_label = "episodes"
    else:
        # Optimized: group by spatial unit to avoid O(n²) over all events
        group_keys = _get_spatial_group_key(config)
        groups = df_valid.groupby(group_keys)
        group_label = "spatial groups"

    n_groups = len(groups)
    if verbose:
        print(f"Split into {n_groups:,} {group_label}")

    cascades = []
    processed = 0
    for group_id, group_events in groups:
        group_cascades = _identify_cascades_in_group(
            group_events, temporal_window, config, valid_pairs
        )
        cascades.extend(group_cascades)
        
        processed += 1
        if verbose and processed % 500 == 0:
            print(f"  Processed {processed:,}/{n_groups:,} {group_label}, "
                  f"found {len(cascades):,} cascades so far...")
    
    if verbose:
        print(f"Found {len(cascades):,} cascade pairs")
    
    return cascades


def cascades_to_dataframe(cascades: List[CascadePair]) -> pd.DataFrame:
    """Convert list of CascadePair objects to DataFrame."""
    if not cascades:
        return pd.DataFrame()
    
    return pd.DataFrame([
        {
            'primary_event_id': c.primary_event_id,
            'secondary_event_id': c.secondary_event_id,
            'primary_event_type': c.primary_event_type,
            'secondary_event_type': c.secondary_event_type,
            'time_gap_hours': c.time_gap_hours,
            'same_county': c.same_county,
            'same_episode': c.same_episode,
            'distance_km': c.distance_km
        }
        for c in cascades
    ])


def get_cascade_statistics(cascade_df: pd.DataFrame) -> dict:
    """Get summary statistics of identified cascades."""
    if cascade_df.empty:
        return {"total_cascades": 0}
    
    cascade_types = cascade_df.groupby(
        ['primary_event_type', 'secondary_event_type']
    ).size().sort_values(ascending=False)
    
    return {
        "total_cascades": len(cascade_df),
        "unique_primary_types": cascade_df['primary_event_type'].nunique(),
        "unique_secondary_types": cascade_df['secondary_event_type'].nunique(),
        "mean_time_gap_hours": cascade_df['time_gap_hours'].mean(),
        "median_time_gap_hours": cascade_df['time_gap_hours'].median(),
        "top_cascade_types": cascade_types.head(10).to_dict()
    }
