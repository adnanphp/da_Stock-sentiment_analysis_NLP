# data_type_converter.py
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import re
from typing import Dict, List, Any, Optional

class DataTypeConverter:
    def __init__(self, output_dir="data_type_conversion_reports"):
        self.output_dir = output_dir
        self.conversion_report = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        # Handle numpy arrays and pandas Series first
        if isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        # Handle scalar values
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            # If pd.isna fails (e.g., for arrays), treat as not null
            pass
        
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        else:
            # For any other type, try to convert to string
            try:
                return str(obj)
            except:
                return None
    
    def _ensure_numeric(self, value, default=0):
        """Ensure value is numeric, convert if possible"""
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return default
        else:
            return default
    
    def detect_data_type(self, series: pd.Series) -> Dict[str, Any]:
        """
        Automatically detect the most appropriate data type for a pandas Series
        """
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {"detected_type": "unknown", "confidence": 0.0, "reason": "All values are null"}
        
        original_dtype = str(series.dtype)
        sample_values = self._convert_to_serializable(non_null_series.head(10).tolist())
        
        # Test for different data types
        type_scores = {
            "integer": self._score_integer(non_null_series, sample_values),
            "float": self._score_float(non_null_series, sample_values),
            "boolean": self._score_boolean(non_null_series, sample_values),
            "datetime": self._score_datetime(non_null_series, sample_values),
            "categorical": self._score_categorical(non_null_series, sample_values),
            "text": self._score_text(non_null_series, sample_values)
        }
        
        # Get the best matching type
        best_type = max(type_scores.items(), key=lambda x: x[1]['score'])
        detected_type = best_type[0]
        confidence = best_type[1]['score']
        reason = best_type[1]['reason']
        
        return {
            "detected_type": detected_type,
            "confidence": round(confidence, 3),
            "reason": reason,
            "original_dtype": original_dtype,
            "unique_values": len(non_null_series.unique()),
            "sample_values": sample_values[:3]
        }
    
    def _score_integer(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits integer type"""
        try:
            # Try to convert to integer
            int_series = pd.to_numeric(series, errors='coerce')
            valid_count = int_series.notna().sum()
            total_count = len(series)
            
            if valid_count == 0:
                return {"score": 0.0, "reason": "No valid integer values"}
            
            score = valid_count / total_count
            
            # Check if all values are actually integers (no decimals)
            if score > 0.9:
                has_decimals = any(float(x) != int(float(x)) for x in sample_values if pd.notna(x))
                if not has_decimals:
                    score = min(score + 0.1, 1.0)
                    return {"score": score, "reason": "High percentage of valid integers without decimals"}
            
            return {"score": score, "reason": f"{valid_count}/{total_count} valid integer values"}
        except:
            return {"score": 0.0, "reason": "Error in integer conversion"}
    
    def _score_float(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits float type"""
        try:
            float_series = pd.to_numeric(series, errors='coerce')
            valid_count = float_series.notna().sum()
            total_count = len(series)
            
            if valid_count == 0:
                return {"score": 0.0, "reason": "No valid float values"}
            
            score = valid_count / total_count
            
            # Check if values have decimals (indicating they're truly floats)
            has_decimals = any(float(x) != int(float(x)) for x in sample_values if pd.notna(x))
            if has_decimals and score > 0.8:
                score = min(score + 0.1, 1.0)
                return {"score": score, "reason": "High percentage of valid floats with decimal values"}
            
            return {"score": score, "reason": f"{valid_count}/{total_count} valid float values"}
        except:
            return {"score": 0.0, "reason": "Error in float conversion"}
    
    def _score_boolean(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits boolean type"""
        try:
            # Common boolean representations
            true_values = ['true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n']
            str_series = series.astype(str).str.lower().str.strip()
            
            boolean_like = str_series.isin(true_values)
            valid_count = boolean_like.sum()
            total_count = len(series)
            
            if valid_count == 0:
                return {"score": 0.0, "reason": "No common boolean representations"}
            
            score = valid_count / total_count
            
            # Bonus if only 2 unique values (typical for booleans)
            unique_count = len(str_series.unique())
            if unique_count == 2 and score > 0.8:
                score = min(score + 0.2, 1.0)
                return {"score": score, "reason": "Perfect boolean pattern (2 unique values)"}
            
            return {"score": score, "reason": f"{valid_count}/{total_count} boolean-like values"}
        except:
            return {"score": 0.0, "reason": "Error in boolean detection"}
    
    def _score_datetime(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits datetime type"""
        try:
            # Try multiple date formats
            date_formats = [
                '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
                '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M'
            ]
            
            valid_count = 0
            for fmt in date_formats:
                try:
                    date_series = pd.to_datetime(series, format=fmt, errors='coerce')
                    valid_count = max(valid_count, date_series.notna().sum())
                except:
                    continue
            
            total_count = len(series)
            
            if valid_count == 0:
                return {"score": 0.0, "reason": "No valid datetime values"}
            
            score = valid_count / total_count
            
            # High confidence if most values are valid dates
            if score > 0.9:
                return {"score": min(score + 0.1, 1.0), "reason": "High percentage of valid datetime values"}
            
            return {"score": score, "reason": f"{valid_count}/{total_count} valid datetime values"}
        except:
            return {"score": 0.0, "reason": "Error in datetime detection"}
    
    def _score_categorical(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits categorical type"""
        try:
            total_count = len(series)
            unique_count = len(series.unique())
            
            # Calculate cardinality ratio
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Good for categorical if low cardinality and reasonable number of unique values
            if cardinality_ratio < 0.1 and unique_count < 100:
                score = 1.0 - cardinality_ratio
                return {"score": min(score, 0.9), "reason": f"Low cardinality ({unique_count} unique values)"}
            elif cardinality_ratio < 0.3 and unique_count < 1000:
                score = 0.8 - (cardinality_ratio * 2)
                return {"score": max(score, 0.3), "reason": f"Moderate cardinality ({unique_count} unique values)"}
            else:
                return {"score": 0.1, "reason": f"High cardinality ({unique_count} unique values)"}
        except:
            return {"score": 0.0, "reason": "Error in categorical detection"}
    
    def _score_text(self, series: pd.Series, sample_values: List) -> Dict[str, Any]:
        """Score how well the series fits text type"""
        try:
            # Check if values are strings and have varying lengths
            str_series = series.astype(str)
            avg_length = str_series.str.len().mean()
            
            # Text typically has longer strings and high variability
            if avg_length > 10:
                score = 0.8
                reason = "Long string values typical of text"
            elif avg_length > 5:
                score = 0.6
                reason = "Moderate length string values"
            else:
                score = 0.3
                reason = "Short string values"
            
            # Check for text patterns (spaces, punctuation)
            has_spaces = str_series.str.contains(' ').any()
            has_punctuation = str_series.str.contains(r'[.,!?;:]').any()
            
            if has_spaces and has_punctuation:
                score = min(score + 0.2, 1.0)
                reason += " with text patterns"
            
            return {"score": score, "reason": reason}
        except:
            return {"score": 0.0, "reason": "Error in text detection"}
    
    def automatic_type_conversion(self, df: pd.DataFrame, dataset_name: str, 
                                 confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Automatically detect and convert data types based on content analysis
        """
        converted_df = df.copy()
        conversion_details = {}
        
        print(f"🔍 Automatically detecting types for: {dataset_name}")
        
        for column in df.columns:
            try:
                # Detect the best data type
                detection_result = self.detect_data_type(df[column])
                detected_type = detection_result["detected_type"]
                confidence = detection_result["confidence"]
                
                original_dtype = str(df[column].dtype)
                conversion_applied = False
                new_dtype = original_dtype
                conversion_notes = []
                
                # Apply conversion if confidence is above threshold
                if confidence >= confidence_threshold:
                    try:
                        if detected_type == "integer":
                            converted_df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                            new_dtype = "Int64"
                            conversion_applied = True
                            conversion_notes.append("Converted to nullable integer")
                            
                        elif detected_type == "float":
                            converted_df[column] = pd.to_numeric(df[column], errors='coerce')
                            new_dtype = "float64"
                            conversion_applied = True
                            conversion_notes.append("Converted to float")
                            
                        elif detected_type == "boolean":
                            # Map common boolean representations
                            bool_map = {
                                'true': True, 'false': False,
                                'yes': True, 'no': False,
                                '1': True, '0': False,
                                't': True, 'f': False,
                                'y': True, 'n': False
                            }
                            str_series = df[column].astype(str).str.lower().str.strip()
                            converted_df[column] = str_series.map(bool_map).astype('boolean')
                            new_dtype = "boolean"
                            conversion_applied = True
                            conversion_notes.append("Converted to boolean")
                            
                        elif detected_type == "datetime":
                            converted_df[column] = pd.to_datetime(df[column], errors='coerce')
                            new_dtype = "datetime64[ns]"
                            conversion_applied = True
                            conversion_notes.append("Converted to datetime")
                            
                        elif detected_type == "categorical":
                            converted_df[column] = df[column].astype('category')
                            new_dtype = "category"
                            conversion_applied = True
                            conversion_notes.append("Converted to category")
                            
                    except Exception as conversion_error:
                        conversion_notes.append(f"Conversion failed: {str(conversion_error)}")
                        conversion_applied = False
                
                # Record conversion details
                conversion_details[column] = {
                    "original_dtype": original_dtype,
                    "detected_type": detected_type,
                    "confidence": confidence,
                    "conversion_applied": conversion_applied,
                    "new_dtype": new_dtype,
                    "conversion_notes": conversion_notes,
                    "unique_values": detection_result["unique_values"],
                    "sample_original": detection_result["sample_values"]
                }
                
                status = "✓" if conversion_applied else "⏭️"
                print(f"  {status} {column}: {original_dtype} → {detected_type} (conf: {confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error processing {column}: {e}")
                conversion_details[column] = {
                    "error": str(e),
                    "original_dtype": str(df[column].dtype),
                    "conversion_applied": False
                }
        
        # Count successful conversions
        columns_converted = len([c for c in conversion_details.values() if c.get("conversion_applied", False)])
        
        # Store conversion report
        self.conversion_report[dataset_name] = {
            "conversion_strategy": "automatic",
            "confidence_threshold": confidence_threshold,
            "original_shape": list(df.shape),
            "converted_shape": list(converted_df.shape),
            "columns_converted": columns_converted,  # Store as integer
            "conversion_details": conversion_details,
            "timestamp": datetime.now().isoformat()
        }
        
        return converted_df
    
    def manual_type_conversion(self, df: pd.DataFrame, dataset_name: str, 
                              type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert data types based on manual specification
        """
        converted_df = df.copy()
        conversion_details = {}
        
        print(f"🔧 Applying manual type conversion for: {dataset_name}")
        
        for column, target_type in type_mapping.items():
            if column not in df.columns:
                print(f"  ⚠️  Column '{column}' not found in dataset")
                continue
                
            try:
                original_dtype = str(df[column].dtype)
                conversion_applied = False
                conversion_notes = []
                
                # Apply the specified conversion
                if target_type.lower() in ['int', 'integer', 'int64']:
                    converted_df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    new_dtype = "Int64"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to integer")
                    
                elif target_type.lower() in ['float', 'float64', 'double']:
                    converted_df[column] = pd.to_numeric(df[column], errors='coerce')
                    new_dtype = "float64"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to float")
                    
                elif target_type.lower() in ['bool', 'boolean']:
                    # For manual boolean, we try direct conversion first
                    try:
                        converted_df[column] = df[column].astype('boolean')
                    except:
                        # Fallback to string mapping
                        bool_map = {
                            'true': True, 'false': False, 'yes': True, 'no': False,
                            '1': True, '0': False, 't': True, 'f': False
                        }
                        str_series = df[column].astype(str).str.lower().str.strip()
                        converted_df[column] = str_series.map(bool_map).astype('boolean')
                    new_dtype = "boolean"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to boolean")
                    
                elif target_type.lower() in ['datetime', 'date', 'timestamp']:
                    converted_df[column] = pd.to_datetime(df[column], errors='coerce')
                    new_dtype = "datetime64[ns]"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to datetime")
                    
                elif target_type.lower() in ['category', 'categorical']:
                    converted_df[column] = df[column].astype('category')
                    new_dtype = "category"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to category")
                    
                elif target_type.lower() in ['string', 'str', 'text']:
                    converted_df[column] = df[column].astype('string')
                    new_dtype = "string"
                    conversion_applied = True
                    conversion_notes.append("Manually converted to string")
                    
                else:
                    conversion_notes.append(f"Unsupported target type: {target_type}")
                
                # Record conversion details
                conversion_details[column] = {
                    "original_dtype": original_dtype,
                    "target_type": target_type,
                    "conversion_applied": conversion_applied,
                    "new_dtype": new_dtype if conversion_applied else original_dtype,
                    "conversion_notes": conversion_notes,
                    "unique_values": len(df[column].unique()),
                    "sample_original": self._convert_to_serializable(df[column].head(3).tolist())
                }
                
                status = "✓" if conversion_applied else "❌"
                print(f"  {status} {column}: {original_dtype} → {target_type}")
                
            except Exception as e:
                print(f"  ❌ Error converting {column} to {target_type}: {e}")
                conversion_details[column] = {
                    "error": str(e),
                    "original_dtype": str(df[column].dtype),
                    "target_type": target_type,
                    "conversion_applied": False,
                    "conversion_notes": [f"Conversion failed: {str(e)}"]
                }
        
        # Count successful conversions
        columns_converted = len([c for c in conversion_details.values() if c.get("conversion_applied", False)])
        
        # Store conversion report
        self.conversion_report[dataset_name] = {
            "conversion_strategy": "manual",
            "type_mapping": type_mapping,
            "original_shape": list(df.shape),
            "converted_shape": list(converted_df.shape),
            "columns_converted": columns_converted,  # Store as integer
            "conversion_details": conversion_details,
            "timestamp": datetime.now().isoformat()
        }
        
        return converted_df
    
    def analyze_dataset_types(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive type analysis without conversion
        """
        print(f"📊 Analyzing data types for: {dataset_name}")
        
        type_analysis = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "total_columns": len(df.columns),
                "total_rows": len(df),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            "column_analysis": {},
            "recommendations": []
        }
        
        for column in df.columns:
            try:
                detection_result = self.detect_data_type(df[column])
                current_dtype = str(df[column].dtype)
                
                type_analysis["column_analysis"][column] = {
                    "current_dtype": current_dtype,
                    "detected_type": detection_result["detected_type"],
                    "confidence": detection_result["confidence"],
                    "reason": detection_result["reason"],
                    "unique_values": detection_result["unique_values"],
                    "null_count": int(df[column].isnull().sum()),  # Convert to Python int
                    "null_percentage": round((df[column].isnull().sum() / len(df)) * 100, 2),
                    "sample_values": self._convert_to_serializable(detection_result["sample_values"]),
                    "recommended_action": self._get_recommended_action(detection_result, current_dtype)
                }
                
                print(f"  📋 {column}: {current_dtype} → {detection_result['detected_type']} (conf: {detection_result['confidence']:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {column}: {e}")
                type_analysis["column_analysis"][column] = {
                    "error": str(e),
                    "current_dtype": str(df[column].dtype)
                }
        
        # Generate overall recommendations
        type_analysis["recommendations"] = self._generate_overall_recommendations(type_analysis)
        
        return type_analysis
    
    def _get_recommended_action(self, detection_result: Dict, current_dtype: str) -> str:
        """Get recommended action for a column"""
        detected_type = detection_result["detected_type"]
        confidence = detection_result["confidence"]
        
        # Map current dtype to simplified type
        current_simple = self._simplify_dtype(current_dtype)
        detected_simple = detected_type
        
        if current_simple == detected_simple and confidence > 0.8:
            return "keep_current"
        elif confidence > 0.7:
            return f"convert_to_{detected_type}"
        else:
            return "review_manually"
    
    def _simplify_dtype(self, dtype: str) -> str:
        """Simplify pandas dtype to basic type"""
        if 'int' in dtype.lower():
            return 'integer'
        elif 'float' in dtype.lower():
            return 'float'
        elif 'bool' in dtype.lower():
            return 'boolean'
        elif 'datetime' in dtype.lower():
            return 'datetime'
        elif 'category' in dtype.lower():
            return 'categorical'
        elif 'object' in dtype.lower():
            return 'text'
        else:
            return 'unknown'
    
    def _generate_overall_recommendations(self, analysis: Dict) -> List[str]:
        """Generate overall recommendations for the dataset"""
        recommendations = []
        column_analysis = analysis["column_analysis"]
        
        # Count conversion recommendations
        conversion_counts = {}
        for col_info in column_analysis.values():
            if "recommended_action" in col_info:
                action = col_info["recommended_action"]
                conversion_counts[action] = conversion_counts.get(action, 0) + 1
        
        # Generate recommendations based on counts
        if conversion_counts.get("review_manually", 0) > 0:
            recommendations.append(f"Review {conversion_counts['review_manually']} columns manually due to low confidence detection")
        
        potential_conversions = sum(count for action, count in conversion_counts.items() if action.startswith("convert_to_"))
        if potential_conversions > 0:
            recommendations.append(f"Consider automatic conversion for {potential_conversions} columns with high confidence")
        
        # Memory optimization recommendation
        current_memory = analysis["overview"]["memory_usage_mb"]
        if current_memory > 100:  # If dataset is larger than 100MB
            recommendations.append("Consider converting object columns to category for memory optimization")
        
        return recommendations
    
    def save_conversion_report(self, output_path: str = "data_type_conversion_report.json"):
        """Save conversion report to JSON"""
        # Convert to serializable format only when saving to JSON
        serializable_report = self._convert_to_serializable(self.conversion_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Conversion report saved to: {output_path}")
    
    def get_conversion_summary(self) -> Dict[str, Any]:
        """Get summary of all conversions performed"""
        summary = {
            "total_datasets_processed": len(self.conversion_report),
            "total_columns_converted": 0,
            "conversion_strategies": {},
            "success_rate": 0
        }
        
        total_columns_processed = 0
        successful_conversions = 0
        
        for dataset, report in self.conversion_report.items():
            strategy = report.get("conversion_strategy", "unknown")
            summary["conversion_strategies"][strategy] = summary["conversion_strategies"].get(strategy, 0) + 1
            
            # Ensure columns_converted is an integer
            columns_converted = report.get("columns_converted", 0)
            columns_converted = self._ensure_numeric(columns_converted, 0)
            summary["total_columns_converted"] += columns_converted
            
            # Count all columns processed
            if "conversion_details" in report:
                total_columns_processed += len(report["conversion_details"])
                successful_conversions += columns_converted
        
        if total_columns_processed > 0:
            summary["success_rate"] = round((successful_conversions / total_columns_processed) * 100, 2)
        
        return summary
