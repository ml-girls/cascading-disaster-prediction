"""
Feature Engineering Module for Cascade Prediction

FIXED: Duplicate timestamp handling in historical features
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from event timestamps."""
    df = df.copy()
    if 'BEGIN_DATETIME' not in df.columns:
        return df
    
    if 'END_DATETIME' in df.columns:
        df['event_duration_hours'] = ((df['END_DATETIME'] - df['BEGIN_DATETIME']).dt.total_seconds() / 3600).clip(lower=0)
    
    df['month'] = df['BEGIN_DATETIME'].dt.month
    df['hour'] = df['BEGIN_DATETIME'].dt.hour
    df['day_of_week'] = df['BEGIN_DATETIME'].dt.dayofweek
    df['day_of_year'] = df['BEGIN_DATETIME'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def parse_damage_string(s):
    """Convert damage strings like '10.00K' or '5.00M' to numeric USD."""
    if pd.isna(s) or s == '':
        return 0.0
    s = str(s).strip().upper()
    if s.endswith('K'):
        return float(s[:-1]) * 1_000
    elif s.endswith('M'):
        return float(s[:-1]) * 1_000_000
    elif s.endswith('B'):
        return float(s[:-1]) * 1_000_000_000
    try:
        return float(s)
    except ValueError:
        return 0.0


def engineer_impact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from event impact metrics."""
    df = df.copy()

    # Parse damage strings into USD
    if 'DAMAGE_PROPERTY' in df.columns:
        df['damage_property_usd'] = df['DAMAGE_PROPERTY'].apply(parse_damage_string)
    if 'DAMAGE_CROPS' in df.columns:
        df['damage_crops_usd'] = df['DAMAGE_CROPS'].apply(parse_damage_string)

    # Total damage
    prop = df.get('damage_property_usd', pd.Series(0, index=df.index)).fillna(0)
    crop = df.get('damage_crops_usd', pd.Series(0, index=df.index)).fillna(0)
    df['total_damage_usd'] = prop + crop
    df['log_damage'] = np.log1p(df['total_damage_usd'])

    # Total injuries
    if 'INJURIES_DIRECT' in df.columns:
        injuries_indirect = df.get('INJURIES_INDIRECT', pd.Series(0, index=df.index))
        df['total_injuries'] = df['INJURIES_DIRECT'].fillna(0) + injuries_indirect.fillna(0)

    # Total deaths + fatality flag
    if 'DEATHS_DIRECT' in df.columns:
        deaths_indirect = df.get('DEATHS_INDIRECT', pd.Series(0, index=df.index))
        df['total_deaths'] = df['DEATHS_DIRECT'].fillna(0) + deaths_indirect.fillna(0)
        df['has_fatalities'] = (df['total_deaths'] > 0).astype(int)

    # Severity score: financial + human impact
    deaths = df.get('total_deaths', pd.Series(0, index=df.index)).fillna(0)
    injuries = df.get('total_injuries', pd.Series(0, index=df.index)).fillna(0)

    df['severity_score'] = (
        df['total_damage_usd'] / 1e6
        + deaths * 100
        + injuries * 10
    )

    return df

def engineer_event_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on event type."""
    df = df.copy()
    if 'EVENT_TYPE' not in df.columns:
        return df
    
    event_types = sorted(df['EVENT_TYPE'].unique())
    type_to_code = {t: i for i, t in enumerate(event_types)}
    df['event_type_encoded'] = df['EVENT_TYPE'].map(type_to_code)
    
    hurricane_types = ['Hurricane', 'Hurricane (Typhoon)', 'Tropical Storm', 'Tropical Depression', 'Storm Surge/Tide']
    flood_types = ['Flash Flood', 'Flood', 'Coastal Flood', 'Lakeshore Flood']
    winter_types = ['Winter Storm', 'Blizzard', 'Heavy Snow', 'Ice Storm', 'Cold/Wind Chill', 'Extreme Cold/Wind Chill', 'Frost/Freeze', 'Winter Weather']
    convective_types = ['Tornado', 'Thunderstorm Wind', 'Hail', 'Lightning']
    
    df['is_hurricane_type'] = df['EVENT_TYPE'].isin(hurricane_types).astype(int)
    df['is_flood_type'] = df['EVENT_TYPE'].isin(flood_types).astype(int)
    df['is_winter_type'] = df['EVENT_TYPE'].isin(winter_types).astype(int)
    df['is_convective_type'] = df['EVENT_TYPE'].isin(convective_types).astype(int)
    
    event_freq = df['EVENT_TYPE'].value_counts(normalize=True)
    df['event_rarity_score'] = df['EVENT_TYPE'].map(event_freq)
    
    return df


def engineer_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spatial features."""
    df = df.copy()
    df['has_coordinates'] = (df['BEGIN_LAT'].notna() & df['BEGIN_LON'].notna()).astype(int)
    df['latitude'] = df['BEGIN_LAT']
    df['longitude'] = df['BEGIN_LON']
    df['abs_latitude'] = df['latitude'].abs()
    
    coastal_states_fips = ['01', '02', '06', '09', '10', '12', '13', '15', '22', '23',
                           '24', '25', '28', '33', '34', '36', '37', '41', '44', '45',
                           '48', '51', '53']
    
    if 'STATE_FIPS' in df.columns:
        df['is_coastal_state'] = df['STATE_FIPS'].astype(str).str.zfill(2).isin(coastal_states_fips).astype(int)
    
    return df


def engineer_tornado_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create tornado-specific features."""
    df = df.copy()
    if 'TOR_F_SCALE' in df.columns:
        f_scale_map = {'EF0': 0, 'EF1': 1, 'EF2': 2, 'EF3': 3, 'EF4': 4, 'EF5': 5,
                       'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4, 'F5': 5}
        df['tornado_intensity'] = df['TOR_F_SCALE'].map(f_scale_map).fillna(-1)

    # Tornado path dimensions
    if 'TOR_LENGTH' in df.columns:
        df['tor_length'] = pd.to_numeric(df['TOR_LENGTH'], errors='coerce').fillna(0)
    if 'TOR_WIDTH' in df.columns:
        df['tor_width_miles'] = pd.to_numeric(df['TOR_WIDTH'], errors='coerce').fillna(0) / 1760  # yards to miles
    if 'tor_length' in df.columns and 'tor_width_miles' in df.columns:
        df['tor_path_area'] = df['tor_length'] * df['tor_width_miles']

    return df

def engineer_event_detail_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from magnitude, CZ type, flood cause, and event path."""
    df = df.copy()

    # Magnitude (wind speed in knots / hail size in inches)
    if 'MAGNITUDE' in df.columns:
        df['magnitude'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce').fillna(0)
        df['has_magnitude'] = (df['magnitude'] > 0).astype(int)
    if 'MAGNITUDE_TYPE' in df.columns and 'magnitude' in df.columns:
        df['is_severe_wind'] = (
            df['MAGNITUDE_TYPE'].isin(['EG', 'ES', 'MG', 'MS']) & (df['magnitude'] >= 50)
        ).astype(int)
        df['is_significant_hail'] = (
            ~df['MAGNITUDE_TYPE'].isin(['EG', 'ES', 'MG', 'MS']) & (df['magnitude'] >= 1.0)
        ).astype(int)

    # CZ Type
    if 'CZ_TYPE' in df.columns:
        df['is_marine_event'] = (df['CZ_TYPE'] == 'M').astype(int)
        df['is_zone_event'] = (df['CZ_TYPE'] == 'Z').astype(int)

    # Flood cause
    if 'FLOOD_CAUSE' in df.columns:
        df['is_ice_jam_flood'] = df['FLOOD_CAUSE'].str.contains('Ice Jam', case=False, na=False).astype(int)
        df['is_rain_flood'] = df['FLOOD_CAUSE'].str.contains('Rain', case=False, na=False).astype(int)
        df['is_snowmelt_flood'] = df['FLOOD_CAUSE'].str.contains('Melt', case=False, na=False).astype(int)

    # Event path length (haversine between begin/end coords)
    if all(c in df.columns for c in ['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON']):
        lat1 = np.radians(df['BEGIN_LAT'])
        lat2 = np.radians(df['END_LAT'])
        dlat = lat2 - lat1
        dlon = np.radians(df['END_LON'] - df['BEGIN_LON'])
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        df['event_path_length_miles'] = 2 * 3958.8 * np.arcsin(np.sqrt(a.clip(upper=1.0)))
        df['event_path_length_miles'] = df['event_path_length_miles'].fillna(0)

    # Begin range (proximity proxy)
    if 'BEGIN_RANGE' in df.columns:
        df['begin_range'] = pd.to_numeric(df['BEGIN_RANGE'], errors='coerce').fillna(0)

    return df


def engineer_historical_features(df: pd.DataFrame, window_days: List[int] = [7, 30]) -> pd.DataFrame:
    """Create historical context features"""
    df = df.copy()
    if 'LOCATION_KEY' not in df.columns or 'BEGIN_DATETIME' not in df.columns or len(df) == 0:
        return df
    
    df = df.sort_values(['LOCATION_KEY', 'BEGIN_DATETIME']).reset_index(drop=True)
    
    # Days since last event
    df['prev_event_time'] = df.groupby('LOCATION_KEY')['BEGIN_DATETIME'].shift(1)
    df['days_since_last_event'] = ((df['BEGIN_DATETIME'] - df['prev_event_time']).dt.total_seconds() / 86400).fillna(999)

    if 'triggered_any_cascade' in df.columns:
        df['_cascade_time'] = df['BEGIN_DATETIME'].where(df['triggered_any_cascade'] == 1, np.nan)
        df['_last_cascade_time'] = df.groupby('LOCATION_KEY')['_cascade_time'].ffill().shift(1)
        df['days_since_last_cascade_trigger'] = ((df['BEGIN_DATETIME'] - df['_last_cascade_time']).dt.total_seconds() / 86400).fillna(999)
        df.drop(columns=['_cascade_time', '_last_cascade_time'], inplace=True, errors='ignore')
    
    
    #Make timestamps unique by adding microseconds
    df['_unique_offset'] = df.groupby(['LOCATION_KEY', 'BEGIN_DATETIME']).cumcount()
    df['_datetime_unique'] = df['BEGIN_DATETIME'] + pd.to_timedelta(df['_unique_offset'], unit='us')
    
    # Rolling window features
    for days in window_days:
        window_str = f'{days}D'
        df_indexed = df.set_index('_datetime_unique')
        loc_groups = df_indexed.groupby('LOCATION_KEY', group_keys=False)
        
        # Use .values to avoid reindex issues
        rolled_count = loc_groups['LOCATION_KEY'].rolling(window_str, closed='left').count()
        df[f'events_last_{days}d'] = rolled_count.values
        
        if 'TOTAL_DAMAGE_USD' in df.columns:
            rolled_damage = loc_groups['TOTAL_DAMAGE_USD'].rolling(window_str, closed='left').sum()
            df[f'damage_last_{days}d'] = rolled_damage.values
        
        if 'severity_score' in df.columns:
            rolled_severity = loc_groups['severity_score'].rolling(window_str, closed='left').max()
            df[f'max_severity_last_{days}d'] = rolled_severity.values
        
        if 'triggered_any_cascade' in df.columns:
            for days in window_days:
                window_str = f'{days}D'
                df_indexed = df.set_index('_datetime_unique')
                df_indexed['_cascade_int'] = df['triggered_any_cascade'].astype(int)
                loc_groups_cascade = df_indexed.groupby('LOCATION_KEY', group_keys=False)
                rolled_cascades = loc_groups_cascade['_cascade_int'].rolling(window_str, closed='left').sum()
                df[f'cascades_triggered_last_{days}d'] = rolled_cascades.values
    
    if 'events_last_30d' in df.columns:
        df['recent_event_density'] = df['events_last_30d'] / 30.0
    
    # Cleanup
    df.drop(columns=['prev_event_time', '_unique_offset', '_datetime_unique'], inplace=True, errors='ignore')
    
    # Fill NaNs
    for days in window_days:
        for col in [f'events_last_{days}d', f'damage_last_{days}d', f'max_severity_last_{days}d', f'cascades_last_{days}d']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
    return df


def engineer_base_features(df: pd.DataFrame, include_historical: bool = True, historical_windows: List[int] = [7, 30]) -> pd.DataFrame:
    """Apply base feature engineering."""
    if len(df) == 0:
        return df
    
    if 'BEGIN_DATETIME' in df.columns:
        df = df.sort_values('BEGIN_DATETIME').reset_index(drop=True)
    
    if 'is_cascade_result' in df.columns:
        # Robust conversion: handle bool, int, or string "True"/"False"
        df['is_cascade_result'] = df['is_cascade_result'].map({'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)
    
    df = engineer_temporal_features(df)
    df = engineer_impact_features(df)
    df = engineer_event_type_features(df)
    df = engineer_spatial_features(df)
    df = engineer_tornado_features(df)
    df = engineer_event_detail_features(df)
    
    if include_historical:
        df = engineer_historical_features(df, window_days=historical_windows)
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Get list of numeric feature columns suitable for modeling."""
    if exclude_cols is None:
        exclude_cols = []
    
    critical_exclude = [
        'EVENT_ID', 'EPISODE_ID', 'LOCATION_KEY',
        'target', 'is_cascade_result', 'triggered_any_cascade',
    ]
    
    datetime_exclude = [
        'BEGIN_DATETIME', 'END_DATETIME', 'BEGIN_DATE_TIME', 'END_DATE_TIME',
    ]
    
    text_exclude = [
        'EVENT_NARRATIVE', 'EPISODE_NARRATIVE',
        'BEGIN_LOCATION', 'END_LOCATION', 'CZ_NAME',
        'SOURCE', 'WFO', 'CZ_TIMEZONE', 'DATA_SOURCE',
        'MONTH_NAME', 'STATE',
        'EVENT_TYPE', 'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'CATEGORY', 'TOR_F_SCALE',
        'BEGIN_AZIMUTH', 'END_AZIMUTH', 
        'TOR_OTHER_WFO', 'TOR_OTHER_CZ_STATE', 'TOR_OTHER_CZ_FIPS', 'TOR_OTHER_CZ_NAME',
        'DAMAGE_PROPERTY', 'DAMAGE_CROPS',
    ]
    
    redundant_exclude = [
        'BEGIN_YEARMONTH', 'END_YEARMONTH',  # Have month_sin/month_cos
        'BEGIN_DAY', 'END_DAY',              # Have day_of_year
        'BEGIN_TIME', 'END_TIME',            # Have hour_sin/hour_cos
        'YEAR',                              # Have temporal features
    ]
    
    all_exclude = set(critical_exclude + datetime_exclude + text_exclude + redundant_exclude + exclude_cols)
    
    # Get numeric columns not in exclusion list
    feature_cols = [col for col in df.columns if col not in all_exclude]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols