# missing_value_visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
from matplotlib import gridspec

class MissingValueVisualizer:
    def __init__(self, output_dir="missing_value_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_missingness_matrix(self, df, dataset_name):
        """Create missingness matrix visualization"""
        plt.figure(figsize=(12, 8))
        msno.matrix(df)
        plt.title(f'Missing Values Matrix - {dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_missingness_barplot(self, df, dataset_name):
        """Create bar plot of missing values by column"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Only show columns with missing values
        missing_data = pd.DataFrame({
            'column': missing_counts.index,
            'missing_count': missing_counts.values,
            'missing_percentage': missing_percentages.values
        })
        missing_data = missing_data[missing_data['missing_count'] > 0]
        
        if len(missing_data) == 0:
            return
            
        plt.figure(figsize=(12, 8))
        bars = plt.barh(missing_data['column'], missing_data['missing_percentage'])
        plt.xlabel('Missing Percentage (%)')
        plt.title(f'Missing Values by Column - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_barplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_missingness_heatmap(self, df, dataset_name):
        """Create correlation heatmap of missing values"""
        # Calculate correlation of missingness
        missing_corr = df.isnull().corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0,
                   cbar_kws={'label': 'Missing Value Correlation'})
        plt.title(f'Missing Value Correlation Heatmap - {dataset_name}', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_imputation_comparison(self, original_df, imputed_df, dataset_name):
        """Create before/after imputation comparison"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Before imputation
        ax1 = fig.add_subplot(gs[0, 0])
        before_missing = original_df.isnull().sum()
        before_missing = before_missing[before_missing > 0]
        ax1.barh(before_missing.index, before_missing.values)
        ax1.set_title('Before Imputation - Missing Counts')
        ax1.set_xlabel('Missing Count')
        
        # After imputation  
        ax2 = fig.add_subplot(gs[0, 1])
        after_missing = imputed_df.isnull().sum()
        after_missing = after_missing[after_missing > 0]
        if len(after_missing) > 0:
            ax2.barh(after_missing.index, after_missing.values)
        ax2.set_title('After Imputation - Missing Counts')
        ax2.set_xlabel('Missing Count')
        
        # Overall comparison
        ax3 = fig.add_subplot(gs[1, :])
        comparison_data = pd.DataFrame({
            'Before': original_df.isnull().sum().sum(),
            'After': imputed_df.isnull().sum().sum()
        }, index=['Total Missing Cells'])
        comparison_data.plot(kind='bar', ax=ax3)
        ax3.set_title('Overall Missing Values Comparison')
        ax3.set_ylabel('Missing Cell Count')
        
        plt.suptitle(f'Imputation Results - {dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_imputation_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_comprehensive_dashboard(self, df, dataset_name, analysis_results=None):
        """Create a comprehensive missing values dashboard"""
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # 1. Missingness matrix
        ax1 = fig.add_subplot(gs[0, 0])
        msno.matrix(df, ax=ax1)
        ax1.set_title('Missing Values Matrix')
        
        # 2. Missingness bar plot
        ax2 = fig.add_subplot(gs[0, 1])
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        missing_data = pd.DataFrame({
            'column': missing_counts.index,
            'missing_percentage': missing_percentages.values
        })
        missing_data = missing_data[missing_data['missing_percentage'] > 0]
        
        if len(missing_data) > 0:
            bars = ax2.barh(missing_data['column'], missing_data['missing_percentage'])
            ax2.set_xlabel('Missing Percentage (%)')
            ax2.set_title('Missing Values by Column')
            
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', ha='left', va='center')
        
        # 3. Data types with missing values
        ax3 = fig.add_subplot(gs[1, 0])
        dtype_missing = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                dtype = str(df[col].dtype)
                dtype_missing[dtype] = dtype_missing.get(dtype, 0) + 1
        
        if dtype_missing:
            ax3.pie(dtype_missing.values(), labels=dtype_missing.keys(), autopct='%1.1f%%')
            ax3.set_title('Data Types with Missing Values')
        
        # 4. Records with missing values
        ax4 = fig.add_subplot(gs[1, 1])
        records_with_missing = df.isnull().any(axis=1).sum()
        records_complete = len(df) - records_with_missing
        
        ax4.pie([records_complete, records_with_missing], 
               labels=['Complete Records', 'Records with Missing Values'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax4.set_title('Record Completeness')
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
        Dataset: {dataset_name}
        Total Records: {len(df):,}
        Total Columns: {len(df.columns)}
        Total Missing Cells: {df.isnull().sum().sum():,}
        Overall Missing Percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%
        Columns with Missing Values: {len([col for col in df.columns if df[col].isnull().sum() > 0])}
        Records with Missing Values: {records_with_missing:,} ({records_with_missing/len(df)*100:.1f}%)
        """
        
        if analysis_results and dataset_name in analysis_results:
            analysis = analysis_results[dataset_name]
            if 'recommendations' in analysis:
                summary_text += "\nKey Recommendations:\n"
                for i, rec in enumerate(analysis['recommendations'][:3], 1):
                    summary_text += f"{i}. {rec}\n"
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Missing Values Analysis Dashboard - {dataset_name}', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{dataset_name}_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
