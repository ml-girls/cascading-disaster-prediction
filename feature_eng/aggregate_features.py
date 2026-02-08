import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

class AggregateFeatureTransformer:
    """Learn and apply aggregate statistics from training data."""
    def __init__(self):
        self.location_stats: Dict = {}
        self.state_stats: Dict = {}
        self.cascade_probs: Dict = {}
        self.cascade_time_gaps: Dict = {}
        self.cascade_distances: Dict = {}
        self.cascade_same_county_probs: Dict = {}
        self.all_secondary_types: set = set()
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame, cascade_pairs_df: Optional[pd.DataFrame] = None):
        if 'LOCATION_KEY' in df.columns:
            location_groups = df.groupby('LOCATION_KEY')
            self.location_stats = {
                loc: {
                    'event_count': len(group),
                    'avg_damage': group['TOTAL_DAMAGE_USD'].mean() if 'TOTAL_DAMAGE_USD' in group else 0,
                    'avg_severity': group['severity_score'].mean() if 'severity_score' in group else 0,
                    'cascade_rate': group['is_cascade_result'].mean() if 'is_cascade_result' in group else 0
                }
                for loc, group in location_groups
            }
        
        if 'STATE' in df.columns:
            state_groups = df.groupby('STATE')
            self.state_stats = {
                state: {
                    'avg_damage': group['TOTAL_DAMAGE_USD'].mean() if 'TOTAL_DAMAGE_USD' in group else 0,
                    'cascade_rate': group['is_cascade_result'].mean() if 'is_cascade_result' in group else 0
                }
                for state, group in state_groups
            }
        
        if cascade_pairs_df is not None and len(cascade_pairs_df) > 0:
            cascade_counts = defaultdict(lambda: defaultdict(int))
            time_gaps = defaultdict(lambda: defaultdict(list))
            distances = defaultdict(lambda: defaultdict(list))
            same_county_flags = defaultdict(lambda: defaultdict(list))
            
            for _, row in cascade_pairs_df.iterrows():
                primary = row['primary_event_type']
                secondary = row['secondary_event_type']
                cascade_counts[primary][secondary] += 1
                self.all_secondary_types.add(secondary)
                
                if 'time_gap_hours' in row:
                    time_gaps[primary][secondary].append(row['time_gap_hours'])
                if 'distance_km' in row and not pd.isna(row['distance_km']):
                    distances[primary][secondary].append(row['distance_km'])
                if 'same_county' in row:
                    same_county_flags[primary][secondary].append(int(row['same_county']))
            
            for primary in cascade_counts:
                total_for_primary = sum(cascade_counts[primary].values())
                self.cascade_probs[primary] = {
                    sec: count / total_for_primary 
                    for sec, count in cascade_counts[primary].items()
                }
                
                self.cascade_time_gaps[primary] = {
                    sec: np.mean(gaps) for sec, gaps in time_gaps[primary].items()
                }
                self.cascade_distances[primary] = {
                    sec: np.mean(dists) if dists else 0.0 for sec, dists in distances[primary].items()
                }
                self.cascade_same_county_probs[primary] = {
                    sec: np.mean(flags) for sec, flags in same_county_flags[primary].items()
                }
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        all_location_cascades = [s['cascade_rate'] for s in self.location_stats.values()]
        all_location_damage = [s['avg_damage'] for s in self.location_stats.values()]
        global_cascade_rate = np.mean(all_location_cascades) if all_location_cascades else 0
        global_avg_damage = np.mean(all_location_damage) if all_location_damage else 0
        
        if 'LOCATION_KEY' in df.columns:
            df['location_event_count'] = df['LOCATION_KEY'].map(
                lambda x: self.location_stats.get(x, {}).get('event_count', 0)
            )
            df['location_avg_damage'] = df['LOCATION_KEY'].map(
                lambda x: self.location_stats.get(x, {}).get('avg_damage', global_avg_damage)
            )
            df['location_avg_severity'] = df['LOCATION_KEY'].map(
                lambda x: self.location_stats.get(x, {}).get('avg_severity', 0)
            )
            df['location_cascade_rate'] = df['LOCATION_KEY'].map(
                lambda x: self.location_stats.get(x, {}).get('cascade_rate', global_cascade_rate)
            )
        
        if 'STATE' in df.columns:
            df['state_avg_damage'] = df['STATE'].map(
                lambda x: self.state_stats.get(x, {}).get('avg_damage', global_avg_damage)
            )
            df['state_cascade_rate'] = df['STATE'].map(
                lambda x: self.state_stats.get(x, {}).get('cascade_rate', global_cascade_rate)
            )
        
        if 'EVENT_TYPE' in df.columns and self.cascade_probs:
            for secondary_type in sorted(self.all_secondary_types):
                sec_clean = secondary_type.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                
                prob_col = f'p_{sec_clean}_given_primary'
                df[prob_col] = df['EVENT_TYPE'].map(
                    lambda x: self.cascade_probs.get(x, {}).get(secondary_type, 0.0)
                )
                
                time_col = f'avg_time_gap_{sec_clean}'
                df[time_col] = df['EVENT_TYPE'].map(
                    lambda x: self.cascade_time_gaps.get(x, {}).get(secondary_type, 0.0)
                )
                
                dist_col = f'avg_dist_{sec_clean}'
                df[dist_col] = df['EVENT_TYPE'].map(
                    lambda x: self.cascade_distances.get(x, {}).get(secondary_type, 0.0)
                )
                
                county_col = f'prob_same_county_{sec_clean}'
                df[county_col] = df['EVENT_TYPE'].map(
                    lambda x: self.cascade_same_county_probs.get(x, {}).get(secondary_type, 0.0)
                )
            
            prob_cols = [c for c in df.columns if c.startswith('p_') and c.endswith('_given_primary')]
            if prob_cols:
                df['total_cascade_probability'] = df[prob_cols].sum(axis=1)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, cascade_pairs_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(df, cascade_pairs_df)
        return self.transform(df)
