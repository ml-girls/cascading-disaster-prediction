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
        self.event_type_stats: Dict = {}
        self.all_secondary_types: set = set()
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame, labels_df: Optional[pd.DataFrame] = None):
        """Fit probabilities and location stats."""
        
        # ADD: Compute triggered flag if not present
        if labels_df is not None and 'triggered_any_cascade' not in df.columns:
            df = df.copy()
            df['triggered_any_cascade'] = (labels_df.sum(axis=1) > 0).astype(int)
    
        if 'LOCATION_KEY' in df.columns:
            location_groups = df.groupby('LOCATION_KEY')
            self.location_stats = {
                loc: {
                    'event_count': len(group),
                    'avg_damage': group['total_damage_usd'].mean() if 'total_damage_usd' in group.columns else 0,
                    'avg_severity': group['severity_score'].mean() if 'severity_score' in group.columns else 0,
                    'cascade_rate': group['triggered_any_cascade'].mean() if 'triggered_any_cascade' in group.columns else 0
                }
                for loc, group in location_groups
            }
        
        if 'STATE' in df.columns:
            state_groups = df.groupby('STATE')
            self.state_stats = {
                state: {
                    'avg_damage': group['total_damage_usd'].mean() if 'total_damage_usd' in group.columns else 0,
                    'cascade_rate': group['triggered_any_cascade'].mean() if 'triggered_any_cascade' in group.columns else 0,
                    'avg_severity': group['severity_score'].mean() if 'severity_score' in group.columns else 0,
                }
                for state, group in state_groups
            }
        
        if labels_df is not None and 'EVENT_TYPE' in df.columns:
            # Combine to get probabilities per event type
            combined = pd.concat([df['EVENT_TYPE'], labels_df], axis=1)
            type_counts = combined.groupby('EVENT_TYPE').size()
            type_cascades = combined.groupby('EVENT_TYPE')[labels_df.columns].sum()
            
            self.all_secondary_types = set(labels_df.columns)
            
            # P(Secondary | Primary) = Count(Primary triggers Secondary) / Count(Primary)
            # Note: One primary can trigger multiple secondaries. 
            # We are calculating the percentage of times this primary type results in that secondary type.
            for primary_type in type_counts.index:
                total_events = type_counts[primary_type]
                self.cascade_probs[primary_type] = {
                    sec: type_cascades.loc[primary_type, sec] / total_events
                    for sec in labels_df.columns
                    if type_cascades.loc[primary_type, sec] > 0
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
            df['state_avg_severity'] = df['STATE'].map(
                lambda x: self.state_stats.get(x, {}).get('avg_severity', 0)
            )

        if 'EVENT_TYPE' in df.columns:
            type_groups = df.groupby('EVENT_TYPE')
            self.event_type_stats = {
                et: {
                    'avg_damage': group['total_damage_usd'].mean() if 'total_damage_usd' in group.columns else 0,
                    'avg_severity': group['severity_score'].mean() if 'severity_score' in group.columns else 0,
                }
                for et, group in type_groups
            }
        
        if 'EVENT_TYPE' in df.columns and self.cascade_probs:
            for secondary_type in sorted(self.all_secondary_types):
                sec_clean = secondary_type.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                prob_col = f'p_{sec_clean}_given_primary'
                df[prob_col] = df['EVENT_TYPE'].map(
                    lambda x: self.cascade_probs.get(x, {}).get(secondary_type, 0.0)
                )
            
            prob_cols = [c for c in df.columns if c.startswith('p_') and c.endswith('_given_primary')]
            if prob_cols:
                df['total_cascade_probability'] = df[prob_cols].sum(axis=1)
        
        if 'EVENT_TYPE' in df.columns:
            df['event_type_avg_damage'] = df['EVENT_TYPE'].map(
                lambda x: self.event_type_stats.get(x, {}).get('avg_damage', global_avg_damage)
            )
            df['event_type_avg_severity'] = df['EVENT_TYPE'].map(
                lambda x: self.event_type_stats.get(x, {}).get('avg_severity', 0)
            )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, labels_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(df, labels_df)
        return self.transform(df)
