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
              f"use_domain_patterns={config.use_domain_patterns}")
    
    # Validate required columns
    if 'BEGIN_DATETIME' not in df.columns or 'END_DATETIME' not in df.columns:
        raise ValueError("DataFrame must have BEGIN_DATETIME and END_DATETIME columns")
    
    # Remove events with missing datetimes
    df_valid = df.dropna(subset=['BEGIN_DATETIME', 'END_DATETIME']).copy()
    
    if verbose:
        print(f"Processing {len(df_valid):,} events...")
    
    cascades = []
    temporal_window = timedelta(days=config.temporal_window_days)
    
    # Group by episode if required
    if config.require_same_episode:
        groups = df_valid.groupby('EPISODE_ID')
    else:
        groups = [('all', df_valid)]
    
    processed = 0
    for episode_id, episode_events in groups:
        if len(episode_events) < 2:
            continue
        
        # Sort by begin time
        episode_events = episode_events.sort_values('BEGIN_DATETIME')
        events_list = list(episode_events.iterrows())
        
        for i, (idx1, event1) in enumerate(events_list):
            for j, (idx2, event2) in enumerate(events_list[i+1:], start=i+1):
                # Calculate time gap
                time_gap = event2['BEGIN_DATETIME'] - event1['END_DATETIME']
                
                # Skip if outside temporal window
                if time_gap < timedelta(0) or time_gap > temporal_window:
                    continue
                
                # Check if different event types
                if config.require_different_event_types:
                    if event1['EVENT_TYPE'] == event2['EVENT_TYPE']:
                        continue
                
                # Check domain patterns (causal filtering)
                if config.use_domain_patterns:
                    if not is_valid_cascade(event1['EVENT_TYPE'], event2['EVENT_TYPE']):
                        continue
                
                # Check spatial proximity
                is_proximate, distance = check_spatial_proximity(event1, event2, config)
                if not is_proximate:
                    continue
                
                # Create cascade pair
                cascade = CascadePair(
                    primary_event_id=event1['EVENT_ID'],
                    secondary_event_id=event2['EVENT_ID'],
                    primary_event_type=event1['EVENT_TYPE'],
                    secondary_event_type=event2['EVENT_TYPE'],
                    time_gap_hours=time_gap.total_seconds() / 3600,
                    same_county=(
                        event1['STATE_FIPS'] == event2['STATE_FIPS'] and
                        event1['CZ_FIPS'] == event2['CZ_FIPS']
                    ),
                    same_episode=(event1['EPISODE_ID'] == event2['EPISODE_ID']),
                    distance_km=distance
                )
                cascades.append(cascade)
        
        processed += 1
        if verbose and processed % 1000 == 0:
            print(f"  Processed {processed:,} episodes, found {len(cascades):,} cascades...")
    
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
