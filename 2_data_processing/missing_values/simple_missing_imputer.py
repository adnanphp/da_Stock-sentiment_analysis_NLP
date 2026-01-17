# simple_missing_imputer.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class SimpleMissingValueImputer:
    def __init__(self):
        self.imputation_report = {}

    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        if pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        else:
            return obj

    def ultra_safe_has_missing(self, df):
        """Ultra-safe check for missing values using numpy"""
        try:
            total_missing = 0
            for col in df.columns:
                try:
                    col_array = df[col].values
                    col_missing = np.sum(pd.isna(col_array))
                    total_missing += col_missing
                except:
                    continue
            return total_missing > 0
        except:
            return False

    def ultra_safe_column_has_missing(self, df, column):
        """Ultra-safe check for missing values in a specific column"""
        try:
            col_array = df[column].values
            return np.sum(pd.isna(col_array)) > 0
        except:
            return False

    def ultra_safe_get_missing_count(self, df, column):
        """Ultra-safe get missing count for a column"""
        try:
            col_array = df[column].values
            return np.sum(pd.isna(col_array))
        except:
            return 0

    def simple_impute(self, df, dataset_name):
        """Simple imputation that avoids all complex operations"""
        imputed_df = df.copy()
        imputation_details = {}

        # Check if we have missing values
        if not self.ultra_safe_has_missing(df):
            return imputed_df

        for column in df.columns:
            try:
                # Ultra-safe missing check for this column
                if not self.ultra_safe_column_has_missing(df, column):
                    continue

                original_missing = self.ultra_safe_get_missing_count(df, column)
                data_type = str(df[column].dtype)
                missing_percentage = (original_missing / len(df)) * 100

                # Simple imputation strategies
                if data_type in ["int64", "float64"]:
                    # For numeric columns, use median
                    try:
                        imputed_value = float(df[column].median())
                        imputed_df[column] = df[column].fillna(imputed_value)
                        action = "median_imputation"
                    except:
                        # If median fails, use 0
                        imputed_value = 0.0
                        imputed_df[column] = df[column].fillna(imputed_value)
                        action = "zero_imputation_fallback"
                    
                elif data_type == "object":
                    # For text columns, use "Unknown"
                    imputed_value = "Unknown"
                    imputed_df[column] = df[column].fillna(imputed_value)
                    action = "constant_imputation"
                    
                else:
                    # For other types, try simple fill
                    try:
                        imputed_df[column] = df[column].fillna(method="ffill").fillna(method="bfill")
                        action = "forward_backward_fill"
                        imputed_value = "forward/backward filled"
                    except:
                        # If that fails, use "Unknown"
                        imputed_value = "Unknown"
                        imputed_df[column] = df[column].fillna(imputed_value)
                        action = "constant_imputation_fallback"

                # Check remaining missing
                remaining_missing = self.ultra_safe_get_missing_count(imputed_df, column)

                imputation_details[column] = {
                    "original_missing": int(original_missing),
                    "missing_percentage": float(missing_percentage),
                    "data_type": data_type,
                    "action_taken": action,
                    "imputed_value": str(imputed_value),
                    "remaining_missing": int(remaining_missing),
                    "success": bool(remaining_missing == 0),
                }

            except Exception as e:
                print(f"    Imputation error for column {column}: {e}")
                imputation_details[column] = {
                    "error": str(e),
                    "original_missing": int(original_missing) if 'original_missing' in locals() else 0,
                    "success": False,
                }

        # Store report
        self.imputation_report[dataset_name] = self._convert_to_serializable({
            "original_shape": list(df.shape),
            "imputed_shape": list(imputed_df.shape),
            "total_columns_imputed": len(imputation_details),
            "imputation_details": imputation_details,
        })

        return imputed_df

    def get_imputation_summary(self):
        """Get summary of all imputations performed"""
        summary = {
            "total_datasets_imputed": len(self.imputation_report),
            "success_rate": 0,
        }

        total_columns = 0
        successful_imputations = 0

        for dataset, report in self.imputation_report.items():
            for column, details in report["imputation_details"].items():
                total_columns += 1
                if details.get("success", False):
                    successful_imputations += 1

        if total_columns > 0:
            summary["success_rate"] = round((successful_imputations / total_columns) * 100, 2)

        return summary
