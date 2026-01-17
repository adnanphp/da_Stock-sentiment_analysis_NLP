# manual_type_mapper.py
import pandas as pd
import json
from data_type_converter import DataTypeConverter

class ManualTypeMapper:
    """
    Interactive tool for creating manual type mappings based on data analysis
    """
    def __init__(self):
        self.converter = DataTypeConverter()
    
    def create_type_mapping_interactive(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Interactive tool to create manual type mappings
        """
        print(f"\n🎯 Creating manual type mapping for: {dataset_name}")
        print("=" * 50)
        
        # First, analyze the dataset
        analysis = self.converter.analyze_dataset_types(df, dataset_name)
        
        type_mapping = {}
        
        for column, col_info in analysis["column_analysis"].items():
            if "error" in col_info:
                print(f"❌ Skipping {column} due to error: {col_info['error']}")
                continue
            
            current_dtype = col_info["current_dtype"]
            detected_type = col_info["detected_type"]
            confidence = col_info["confidence"]
            recommended_action = col_info["recommended_action"]
            
            print(f"\n📝 Column: {column}")
            print(f"   Current: {current_dtype}")
            print(f"   Detected: {detected_type} (confidence: {confidence:.2f})")
            print(f"   Recommendation: {recommended_action}")
            print(f"   Sample: {col_info['sample_values']}")
            
            # Show available type options
            type_options = {
                '1': 'keep_current',
                '2': 'integer',
                '3': 'float', 
                '4': 'boolean',
                '5': 'datetime',
                '6': 'category',
                '7': 'string',
                '8': 'skip'
            }
            
            print("\n   Type options:")
            for key, option in type_options.items():
                print(f"   {key}. {option}")
            
            while True:
                choice = input(f"\n   Choose type for '{column}' (1-8): ").strip()
                
                if choice in type_options:
                    selected_type = type_options[choice]
                    
                    if selected_type == 'skip':
                        print(f"   ⏭️  Skipping {column}")
                        break
                    elif selected_type == 'keep_current':
                        print(f"   ✅ Keeping current type for {column}")
                        break
                    else:
                        type_mapping[column] = selected_type
                        print(f"   ✅ Mapping {column} → {selected_type}")
                        break
                else:
                    print("   ❌ Invalid choice. Please enter 1-8.")
        
        return type_mapping
    
    def generate_mapping_from_analysis(self, analysis: dict, confidence_threshold: float = 0.8) -> dict:
        """
        Generate type mapping automatically from analysis results
        """
        type_mapping = {}
        column_analysis = analysis["column_analysis"]
        
        for column, col_info in column_analysis.items():
            if "error" in col_info:
                continue
            
            current_dtype = col_info["current_dtype"]
            detected_type = col_info["detected_type"]
            confidence = col_info["confidence"]
            recommended_action = col_info["recommended_action"]
            
            # Only map if confidence is high and conversion is recommended
            if (confidence >= confidence_threshold and 
                recommended_action.startswith("convert_to")):
                type_mapping[column] = detected_type
        
        return type_mapping
    
    def save_type_mappings(self, mappings: dict, output_path: str = "manual_type_mappings.json"):
        """Save type mappings to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"✓ Type mappings saved to: {output_path}")
    
    def load_type_mappings(self, input_path: str = "manual_type_mappings.json") -> dict:
        """Load type mappings from JSON file"""
        try:
            with open(input_path, 'r') as f:
                mappings = json.load(f)
            print(f"✓ Type mappings loaded from: {input_path}")
            return mappings
        except FileNotFoundError:
            print(f"⚠️  No existing mappings found at: {input_path}")
            return {}

def test_manual_mapping():
    """Test function for manual mapping"""
    # Example usage
    mapper = ManualTypeMapper()
    
    # Create sample data for testing
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': ['25', '30', '35', '40', '45'],  # Strings that should be integers
        'is_active': ['true', 'false', 'true', 'false', 'true'],  # Boolean strings
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'date_joined': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Generate mapping interactively
    mapping = mapper.create_type_mapping_interactive(df, "sample_dataset")
    
    print(f"\n🎯 Final type mapping:")
    for col, col_type in mapping.items():
        print(f"  {col}: {col_type}")
    
    # Save mappings
    mapper.save_type_mappings({"sample_dataset": mapping})

if __name__ == "__main__":
    test_manual_mapping()
