# timezone_normalizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz
from typing import Dict, List, Any, Optional, Union
import re

class TimezoneNormalizer:
    """
    Normalize timezone information in datetime data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.common_timezones = self._get_common_timezones()
        self.timezone_aliases = self._get_timezone_aliases()
    
    def _get_common_timezones(self) -> Dict[str, str]:
        """Get mapping of common timezone names to pytz timezones."""
        return {
            'UTC': 'UTC',
            'GMT': 'UTC',
            'EST': 'US/Eastern',
            'EDT': 'US/Eastern',
            'CST': 'US/Central',
            'CDT': 'US/Central',
            'MST': 'US/Mountain',
            'MDT': 'US/Mountain',
            'PST': 'US/Pacific',
            'PDT': 'US/Pacific',
            'CET': 'Europe/Paris',
            'CEST': 'Europe/Paris',
            'EET': 'Europe/Athens',
            'EEST': 'Europe/Athens',
            'WET': 'Europe/Lisbon',
            'WEST': 'Europe/Lisbon',
        }
    
    def _get_timezone_aliases(self) -> Dict[str, str]:
        """Get aliases for common timezone misspellings."""
        return {
            'utc': 'UTC',
            'gmt': 'UTC',
            'est': 'US/Eastern',
            'edt': 'US/Eastern',
            'cst': 'US/Central',
            'cdt': 'US/Central',
            'pst': 'US/Pacific',
            'pdt': 'US/Pacific',
            'eastern': 'US/Eastern',
            'central': 'US/Central',
            'mountain': 'US/Mountain',
            'pacific': 'US/Pacific',
        }
    
    def detect_timezone(self, dt: pd.Timestamp) -> Optional[str]:
        """Detect timezone from a datetime object."""
        if pd.isna(dt):
            return None
        
        # If datetime already has timezone info
        if dt.tz is not None:
            return str(dt.tz)
        
        # Try to infer from string representation
        dt_str = str(dt)
        tz_patterns = {
            r'([+-]\d{2}:?\d{2})$': 'UTC',  # UTC offset
            r'\b(UTC|GMT)\b': 'UTC',
            r'\b(EST|EDT)\b': 'US/Eastern',
            r'\b(CST|CDT)\b': 'US/Central',
            r'\b(MST|MDT)\b': 'US/Mountain',
            r'\b(PST|PDT)\b': 'US/Pacific',
            r'\bCET\b': 'Europe/Paris',
            r'\bCEST\b': 'Europe/Paris',
        }
        
        for pattern, tz in tz_patterns.items():
            if re.search(pattern, dt_str, re.IGNORECASE):
                return tz
        
        return None
    
    def extract_timezone_from_string(self, date_string: str) -> Optional[str]:
        """Extract timezone information from date string."""
        if not isinstance(date_string, str):
            return None
        
        # Common timezone patterns
        patterns = {
            r'([+-]\d{4})\b': 'UTC',  # +0800, -0500
            r'([+-]\d{2}:?\d{2})\b': 'UTC',  # +08:00, -05:00
            r'\b(UTC|GMT)\b': 'UTC',
            r'\b(EST|EDT)\b': 'US/Eastern',
            r'\b(CST|CDT)\b': 'US/Central',
            r'\b(MST|MDT)\b': 'US/Mountain',
            r'\b(PST|PDT)\b': 'US/Pacific',
            r'\b(AEST|AEDT)\b': 'Australia/Sydney',
            r'\b(ACST|ACDT)\b': 'Australia/Adelaide',
            r'\b(AWST)\b': 'Australia/Perth',
            r'\b(CET|CEST)\b': 'Europe/Paris',
            r'\b(EET|EEST)\b': 'Europe/Athens',
            r'\b(WET|WEST)\b': 'Europe/Lisbon',
            r'\b(IST)\b': 'Asia/Kolkata',  # Indian Standard Time
            r'\b(JST)\b': 'Asia/Tokyo',
            r'\b(KST)\b': 'Asia/Seoul',
            r'\b(CST)\b': 'Asia/Shanghai',  # China Standard Time
        }
        
        for pattern, default_tz in patterns.items():
            match = re.search(pattern, date_string, re.IGNORECASE)
            if match:
                tz_str = match.group(1).upper()
                
                # Handle UTC offsets
                if tz_str in ['UTC', 'GMT']:
                    return 'UTC'
                elif tz_str in self.common_timezones:
                    return self.common_timezones[tz_str]
                else:
                    return default_tz
        
        return None
    
    def normalize_timezone(self, series: pd.Series, target_timezone: str = 'UTC') -> pd.Series:
        """Normalize timezone of a datetime series."""
        print(f"    🌍 Normalizing timezone to: {target_timezone}")
        
        if series.empty:
            return series
        
        # Ensure target timezone is valid
        try:
            target_tz = pytz.timezone(target_timezone)
        except pytz.UnknownTimeZoneError:
            print(f"    ⚠️  Unknown timezone: {target_timezone}, falling back to UTC")
            target_tz = pytz.UTC
            target_timezone = 'UTC'
        
        # Handle different timezone scenarios
        normalized_series = series.copy()
        
        # Scenario 1: Series already has timezone info
        if hasattr(series, 'dt') and series.dt.tz is not None:
            print("    🔄 Converting from existing timezone...")
            normalized_series = series.dt.tz_convert(target_tz)
        
        # Scenario 2: Series is timezone-naive, assume UTC and localize
        else:
            print("    🔄 Localizing naive datetime to target timezone...")
            try:
                # First try to localize as UTC, then convert
                utc_series = series.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                normalized_series = utc_series.dt.tz_convert(target_tz)
            except (TypeError, ValueError):
                # Fallback: direct localization to target timezone
                try:
                    normalized_series = series.dt.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
                except Exception as e:
                    print(f"    ⚠️  Timezone normalization failed: {e}")
                    # Return original series if normalization fails
                    return series
        
        return normalized_series
    
    def infer_and_normalize_timezone(self, series: pd.Series, default_timezone: str = 'UTC') -> pd.Series:
        """Infer timezone from data and normalize."""
        print("    🔍 Inferring timezone from data...")
        
        if series.empty:
            return series
        
        # Sample data for timezone inference
        sample_size = min(100, len(series))
        sample = series.dropna().head(sample_size)
        
        detected_timezones = []
        
        for dt in sample:
            tz = self.detect_timezone(dt)
            if tz:
                detected_timezones.append(tz)
        
        # Determine most common timezone
        if detected_timezones:
            from collections import Counter
            tz_counter = Counter(detected_timezones)
            most_common_tz = tz_counter.most_common(1)[0][0]
            print(f"    📊 Inferred timezone: {most_common_tz} (confidence: {tz_counter[most_common_tz]}/{len(detected_timezones)})")
            
            # Normalize using inferred timezone
            return self.normalize_timezone(series, most_common_tz)
        else:
            print(f"    ⚠️  No timezone detected, using default: {default_timezone}")
            return self.normalize_timezone(series, default_timezone)
    
    def get_timezone_info(self, series: pd.Series) -> Dict[str, Any]:
        """Get timezone information about a datetime series."""
        info = {
            'has_timezone': False,
            'timezone': None,
            'is_naive': True,
            'sample_timezones': []
        }
        
        if series.empty:
            return info
        
        # Check if series has timezone info
        if hasattr(series, 'dt') and series.dt.tz is not None:
            info['has_timezone'] = True
            info['timezone'] = str(series.dt.tz)
            info['is_naive'] = False
        
        # Sample timezone detection
        sample = series.dropna().head(10)
        for dt in sample:
            tz = self.detect_timezone(dt)
            if tz and tz not in info['sample_timezones']:
                info['sample_timezones'].append(tz)
        
        return info
