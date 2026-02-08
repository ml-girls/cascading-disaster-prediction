"""
Data Loader for NOAA Storm Events Database

Handles loading and parsing of NOAA Storm Events CSV files with proper
date parsing and data type handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime
import glob


def parse_noaa_datetime(date_str: str) -> Optional[datetime]:
    """
    Parse NOAA datetime string format.
    
    Examples: "25-OCT-23 02:30:00", "07/29/2023 00:00:00"
    """
    if pd.isna(date_str) or date_str == "":
        return None
    
    # Try different formats
    formats = [
        "%d-%b-%y %H:%M:%S",  # 25-OCT-23 02:30:00
        "%m/%d/%Y %H:%M:%S",  # 07/29/2023 00:00:00
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def parse_damage_value(value: str) -> float:
    """
    Parse NOAA damage values which may contain K (thousands) or M (millions).
    
    Examples: "5K", "1.5M", "100", ""
    Returns value in dollars.
    """
    if pd.isna(value) or value == "" or value is None:
        return 0.0
    
    value = str(value).strip().upper()
    
    if value == "":
        return 0.0
    
    try:
        if value.endswith("K"):
            return float(value[:-1]) * 1_000
        elif value.endswith("M"):
            return float(value[:-1]) * 1_000_000
        elif value.endswith("B"):
            return float(value[:-1]) * 1_000_000_000
        else:
            return float(value)
    except ValueError:
        return 0.0


def load_single_year(
    filepath: Union[str, Path],
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load a single year's NOAA Storm Events details file.
    
    Args:
        filepath: Path to the CSV file
        parse_dates: Whether to parse datetime columns
        
    Returns:
        DataFrame with parsed storm events
    """
    df = pd.read_csv(filepath, low_memory=False)
    
    # Parse datetime columns
    if parse_dates:
        df['BEGIN_DATETIME'] = df['BEGIN_DATE_TIME'].apply(parse_noaa_datetime)
        df['END_DATETIME'] = df['END_DATE_TIME'].apply(parse_noaa_datetime)
    
    # Parse damage values
    df['DAMAGE_PROPERTY_USD'] = df['DAMAGE_PROPERTY'].apply(parse_damage_value)
    df['DAMAGE_CROPS_USD'] = df['DAMAGE_CROPS'].apply(parse_damage_value)
    df['TOTAL_DAMAGE_USD'] = df['DAMAGE_PROPERTY_USD'] + df['DAMAGE_CROPS_USD']
    
    # Ensure numeric columns
    numeric_cols = ['INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Create composite location key (STATE_FIPS + CZ_FIPS)
    df['LOCATION_KEY'] = df['STATE_FIPS'].astype(str).str.zfill(2) + '_' + df['CZ_FIPS'].astype(str).str.zfill(3)
    
    return df


def load_storm_events(
    data_dir: Union[str, Path],
    years: Optional[List[int]] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load NOAA Storm Events from multiple year files.
    
    Args:
        data_dir: Directory containing the details/ subdirectory
        years: List of years to load (e.g., [2020, 2021, 2022]). If None, loads all.
        parse_dates: Whether to parse datetime columns
        
    Returns:
        Combined DataFrame with all storm events
    """
    data_path = Path(data_dir)
    details_dir = data_path / "details"
    
    if not details_dir.exists():
        raise FileNotFoundError(f"Details directory not found: {details_dir}")
    
    # Find all CSV files
    all_files = sorted(glob.glob(str(details_dir / "StormEvents_details-ftp_*.csv")))
    
    if not all_files:
        raise FileNotFoundError(f"No storm events files found in {details_dir}")
    
    # Filter by years if specified
    if years is not None:
        filtered_files = []
        for f in all_files:
            # Extract year from filename (e.g., _d2023_)
            for year in years:
                if f"_d{year}_" in f:
                    filtered_files.append(f)
                    break
        all_files = filtered_files
    
    print(f"Loading {len(all_files)} files...")
    
    dfs = []
    for filepath in all_files:
        try:
            df = load_single_year(filepath, parse_dates=parse_dates)
            dfs.append(df)
            print(f"  Loaded {Path(filepath).name}: {len(df):,} events")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    if not dfs:
        raise ValueError("No files successfully loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal events loaded: {len(combined):,}")
    
    return combined


def load_fatalities(
    data_dir: Union[str, Path],
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load NOAA Storm Events fatalities data.
    
    Args:
        data_dir: Directory containing the fatalities/ subdirectory
        years: List of years to load. If None, loads all.
        
    Returns:
        DataFrame with fatality records
    """
    data_path = Path(data_dir)
    fatalities_dir = data_path / "fatalities"
    
    if not fatalities_dir.exists():
        raise FileNotFoundError(f"Fatalities directory not found: {fatalities_dir}")
    
    all_files = sorted(glob.glob(str(fatalities_dir / "StormEvents_fatalities-ftp_*.csv")))
    
    if years is not None:
        filtered_files = []
        for f in all_files:
            for year in years:
                if f"_d{year}_" in f:
                    filtered_files.append(f)
                    break
        all_files = filtered_files
    
    dfs = []
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def get_event_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics by event type."""
    return df.groupby('EVENT_TYPE').agg({
        'EVENT_ID': 'count',
        'INJURIES_DIRECT': 'sum',
        'DEATHS_DIRECT': 'sum',
        'TOTAL_DAMAGE_USD': 'sum'
    }).rename(columns={'EVENT_ID': 'count'}).sort_values('count', ascending=False)


def load_storm_events_with_fatalities(
    data_dir: Union[str, Path],
    years: Optional[List[int]] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load NOAA Storm Events with fatality counts joined.
    
    Adds columns:
    - FATALITY_COUNT: Number of fatalities for this event
    - FATALITY_LOCATIONS: Comma-separated fatality locations
    - FATALITY_TYPES: Comma-separated fatality types (D=Direct, I=Indirect)
    
    Args:
        data_dir: Directory containing details/ and fatalities/ subdirectories
        years: List of years to load
        parse_dates: Whether to parse datetime columns
        
    Returns:
        DataFrame with storm events and aggregated fatality info
    """
    # Load events
    df = load_storm_events(data_dir, years=years, parse_dates=parse_dates)
    
    # Load fatalities
    print("\nLoading fatalities...")
    fatalities = load_fatalities(data_dir, years=years)
    
    if fatalities.empty:
        print("No fatalities data found")
        df['FATALITY_COUNT'] = 0
        df['FATALITY_LOCATIONS'] = ""
        df['FATALITY_TYPES'] = ""
        return df
    
    print(f"Loaded {len(fatalities):,} fatality records")
    
    # Aggregate fatalities by EVENT_ID
    fatality_agg = fatalities.groupby('EVENT_ID').agg({
        'FATALITY_ID': 'count',
        'FATALITY_LOCATION': lambda x: ','.join(x.dropna().astype(str).unique()),
        'FATALITY_TYPE': lambda x: ','.join(x.dropna().astype(str).unique()),
    }).rename(columns={
        'FATALITY_ID': 'FATALITY_COUNT',
        'FATALITY_LOCATION': 'FATALITY_LOCATIONS',
        'FATALITY_TYPE': 'FATALITY_TYPES'
    })
    
    # Merge with events
    df = df.merge(fatality_agg, on='EVENT_ID', how='left')
    df['FATALITY_COUNT'] = df['FATALITY_COUNT'].fillna(0).astype(int)
    df['FATALITY_LOCATIONS'] = df['FATALITY_LOCATIONS'].fillna('')
    df['FATALITY_TYPES'] = df['FATALITY_TYPES'].fillna('')
    
    print(f"Events with fatalities: {(df['FATALITY_COUNT'] > 0).sum():,}")
    
    return df


def read_all_csvs_from_gdrive(drive, folder_id, folder_name):
    """
    Fetches all CSV files from a GDrive folder and combines them.
    """
    query = f"'{folder_id}' in parents and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    if not file_list:
        print(f"No files found in {folder_name}")
        return pd.DataFrame()

    li = []
    for file in file_list:
        if file['title'].endswith('.csv') or file['title'].endswith('.csv.gz'):
            if file['title'].endswith('.gz'):
                content = file.GetContentBinary()
                df = pd.read_csv(io.BytesIO(content), compression='gzip', low_memory=False)
            else:
                content = file.GetContentString()
                df = pd.read_csv(io.StringIO(content), low_memory=False)
            li.append(df)

    return pd.concat(li, axis=0, ignore_index=True) if li else pd.DataFrame()