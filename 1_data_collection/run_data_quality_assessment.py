# run_data_quality_assessment.py
from data_quality_checker import DataQualityChecker
from data_profiler import DataProfiler
import json
import os
from datetime import datetime

def main():
    data_directory = "financial_data_large"  # Your 90MB data directory
    
    print("=" * 60)
    print("COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    # 1. Run Data Quality Checker
    print("\n1. RUNNING DATA QUALITY CHECKER...")
    quality_checker = DataQualityChecker(data_directory)
    quality_report = quality_checker.check_all_datasets()
    
    # Save quality report
    quality_checker.save_quality_report("data_quality_report.json")
    
    # Print summary
    quality_summary = quality_checker.get_summary_statistics()
    print("\nQUALITY CHECK SUMMARY:")
    print(json.dumps(quality_summary, indent=2))
    
    # 2. Run Data Profiler (with error handling)
    print("\n2. RUNNING DATA PROFILER...")
    try:
        data_profiler = DataProfiler(data_directory)
        profile_reports = data_profiler.profile_all_datasets()
        
        # Save profiles
        data_profiler.save_profiles("profiling_reports")
        
        # Print profiling summary
        profiling_summary = data_profiler.generate_summary_report()
        print("\nPROFILING SUMMARY:")
        print(json.dumps(profiling_summary, indent=2))
        
    except Exception as e:
        print(f"⚠️  Data profiling encountered an error: {e}")
        print("Continuing with quality assessment results...")
    
    # 3. Generate combined insights
    print("\n3. KEY INSIGHTS & RECOMMENDATIONS:")
    print("-" * 40)
    
    # Analyze quality scores
    quality_scores = []
    for dataset, report in quality_report.items():
        if 'overall_quality_score' in report:
            quality_scores.append(report['overall_quality_score'])
    
    if quality_scores:
        avg_score = sum(quality_scores) / len(quality_scores)
        print(f"Average Data Quality Score: {avg_score:.1f}/100")
        
        if avg_score < 70:
            print("⚠️  Data quality needs improvement")
        elif avg_score < 85:
            print("✅ Data quality is acceptable")
        else:
            print("🎉 Excellent data quality!")
    
    print(f"\nAssessment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Generated reports:")
    print("  - data_quality_report.json")
    print("  - profiling_reports/ (directory with individual dataset profiles)")

if __name__ == "__main__":
    main()
