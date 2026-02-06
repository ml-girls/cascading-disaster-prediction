"""
Feature Engineering Module

Creates features for cascade prediction from storm event data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from event timestamps.
    
    Features created:
    - event_duration_hours: Duration of the event
    - month: Month of event start
    - day_of_week: Day of week (0=Monday)
    - hour: Hour of event start
    - season: Meteorological season (spring, summer, fall, winter)
    - is_weekend: Boolean for weekend events
    """
    df = df.copy()
    
    # Event duration
    if 'BEGIN_DATETIME' in df.columns and 'END_DATETIME' in df.columns:
        df['event_duration_hours'] = (
            (df['END_DATETIME'] - df['BEGIN_DATETIME'])
            .dt.total_seconds() / 3600
        ).clip(lower=0)  # Ensure non-negative
    
    # Extract time components
    if 'BEGIN_DATETIME' in df.columns:
        df['month'] = df['BEGIN_DATETIME'].dt.month
        df['day_of_week'] = df['BEGIN_DATETIME'].dt.dayofweek
        df['hour'] = df['BEGIN_DATETIME'].dt.hour
        df['day_of_year'] = df['BEGIN_DATETIME'].dt.dayofyear
        
        # Season (meteorological)
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df


def engineer_impact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from event impact metrics.
    
    Features created:
    - total_injuries: Direct + indirect injuries
    - total_deaths: Direct + indirect deaths
    - has_fatalities: Boolean for fatal events
    - has_damage: Boolean for events with property/crop damage
    - log_damage: Log-transformed total damage (for modeling)
    """
    df = df.copy()
    
    # Combined metrics
    injury_cols = ['INJURIES_DIRECT', 'INJURIES_INDIRECT']
    death_cols = ['DEATHS_DIRECT', 'DEATHS_INDIRECT']
    
    if all(col in df.columns for col in injury_cols):
        df['total_injuries'] = df[injury_cols].sum(axis=1)
    
    if all(col in df.columns for col in death_cols):
        df['total_deaths'] = df[death_cols].sum(axis=1)
        df['has_fatalities'] = df['total_deaths'] > 0
    
    # Damage features
    if 'TOTAL_DAMAGE_USD' in df.columns:
        df['has_damage'] = df['TOTAL_DAMAGE_USD'] > 0
        df['log_damage'] = np.log1p(df['TOTAL_DAMAGE_USD'])
    
    return df


def engineer_episode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on episode groupings.
    
    Features created:
    - episode_event_count: Total events in same episode
    - episode_event_position: Position of this event in episode (by time)
    - is_first_in_episode: Boolean for first event in episode
    - episode_duration_hours: Total duration of episode
    """
    df = df.copy()
    
    if 'EPISODE_ID' not in df.columns:
        return df
    
    # Count events per episode
    episode_counts = df.groupby('EPISODE_ID').size().rename('episode_event_count')
    df = df.merge(episode_counts, on='EPISODE_ID', how='left')
    
    # Position within episode
    if 'BEGIN_DATETIME' in df.columns:
        df = df.sort_values(['EPISODE_ID', 'BEGIN_DATETIME'])
        df['episode_event_position'] = df.groupby('EPISODE_ID').cumcount() + 1
        df['is_first_in_episode'] = df['episode_event_position'] == 1
        
        # Episode duration (first to last event)
        episode_durations = df.groupby('EPISODE_ID').agg({
            'BEGIN_DATETIME': 'min',
            'END_DATETIME': 'max'
        })
        episode_durations['episode_duration_hours'] = (
            (episode_durations['END_DATETIME'] - episode_durations['BEGIN_DATETIME'])
            .dt.total_seconds() / 3600
        ).clip(lower=0)
        episode_durations = episode_durations['episode_duration_hours']
        df = df.merge(episode_durations, on='EPISODE_ID', how='left')
    
    return df


def engineer_event_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on event type.
    
    Features created:
    - event_type_encoded: Numeric encoding of event type
    - is_hurricane_type: Boolean for hurricane-related events
    - is_flood_type: Boolean for flood-related events
    - is_winter_type: Boolean for winter weather events
    """
    df = df.copy()
    
    if 'EVENT_TYPE' not in df.columns:
        return df
    
    # Numeric encoding
    event_types = df['EVENT_TYPE'].unique()
    type_to_code = {t: i for i, t in enumerate(sorted(event_types))}
    df['event_type_encoded'] = df['EVENT_TYPE'].map(type_to_code)
    
    # Category flags
    hurricane_types = [
        'Hurricane', 'Hurricane (Typhoon)', 'Tropical Storm', 
        'Tropical Depression', 'Storm Surge/Tide'
    ]
    flood_types = [
        'Flash Flood', 'Flood', 'Coastal Flood', 'Lakeshore Flood'
    ]
    winter_types = [
        'Winter Storm', 'Blizzard', 'Heavy Snow', 'Ice Storm',
        'Cold/Wind Chill', 'Extreme Cold/Wind Chill', 'Frost/Freeze'
    ]
    
    df['is_hurricane_type'] = df['EVENT_TYPE'].isin(hurricane_types)
    df['is_flood_type'] = df['EVENT_TYPE'].isin(flood_types)
    df['is_winter_type'] = df['EVENT_TYPE'].isin(winter_types)
    
    return df


def engineer_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spatial features.
    
    Features created:
    - has_coordinates: Boolean for events with lat/lon
    - latitude: Cleaned latitude
    - longitude: Cleaned longitude
    - is_coastal_state: Approximate flag for coastal states
    """
    df = df.copy()
    
    # Coordinate availability
    df['has_coordinates'] = (
        df['BEGIN_LAT'].notna() & 
        df['BEGIN_LON'].notna()
    )
    
    # Clean coordinates
    df['latitude'] = df['BEGIN_LAT']
    df['longitude'] = df['BEGIN_LON']
    
    # Coastal state approximation (simplified)
    coastal_states_fips = [
        '01', '02', '06', '09', '10', '12', '13', '15', '22', '23',
        '24', '25', '28', '33', '34', '36', '37', '41', '44', '45',
        '48', '51', '53'  # AL, AK, CA, CT, DE, FL, GA, HI, LA, ME, MD, MA, MS, NH, NJ, NY, NC, OR, RI, SC, TX, VA, WA
    ]
    df['is_coastal_state'] = df['STATE_FIPS'].astype(str).str.zfill(2).isin(coastal_states_fips)
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Raw storm events DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    df = engineer_temporal_features(df)
    df = engineer_impact_features(df)
    df = engineer_episode_features(df)
    df = engineer_event_type_features(df)
    df = engineer_spatial_features(df)
    
    return df


def get_feature_columns(include_target: bool = False) -> List[str]:
    """Get list of engineered feature column names for modeling."""
    
    features = [
        # Temporal
        'event_duration_hours',
        'month',
        'day_of_week',
        'hour',
        'day_of_year',
        'is_weekend',
        
        # Impact
        'total_injuries',
        'total_deaths',
        'has_fatalities',
        'has_damage',
        'log_damage',
        
        # Episode
        'episode_event_count',
        'episode_event_position',
        'is_first_in_episode',
        'episode_duration_hours',
        
        # Event type
        'event_type_encoded',
        'is_hurricane_type',
        'is_flood_type',
        'is_winter_type',
        
        # Spatial
        'has_coordinates',
        'latitude',
        'longitude',
        'is_coastal_state',
    ]
    
    if include_target:
        features.append('target')
    
    return features
