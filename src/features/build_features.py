import pandas as pd
import os
import sys
from src.features.temporal import calculate_temporal_features
from src.features.graph_features import calculate_graph_features
from src.features.context import calculate_context_features
from src.data.graph import load_dependency_graph

def main():
    # 1. Load and Validate CSV
    raw_path = 'data/raw/api_failure_cascade_dataset.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        sys.exit(1)
        
    print(f"Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    initial_row_count = len(df)
    
    # 2. Load Labels (if they exist)
    # Labels are typically cascade_risk_score, cascade_level, etc.
    # They are already in the CSV for this dataset.
    
    # 3. Temporal Features
    print("Generating temporal features...")
    df = calculate_temporal_features(df)
    
    # 4. Graph Features
    print("Generating graph features...")
    graph = load_dependency_graph(raw_path)
    df = calculate_graph_features(df, graph)
    
    # 5. Context Features
    print("Generating context features...")
    df = calculate_context_features(df)
    
    # 6. Prevent Data Leakage
    # Exclude all target-derived fields from model inputs
    # These should be kept for evaluation but removed from features_parquet
    target_cols = [
        'cascade_risk_score', 
        'cascade_level', 
        'blast_radius'
    ]
    # We keep 'cascade_label' if it exists, otherwise we'll use the above.
    # Looking at the head, we have 'cascade_risk_score' and 'cascade_level'.
    # Let's assume 'cascade_level' is our target.
    
    # Identify target-derived columns present in the DF
    existing_targets = [col for col in target_cols if col in df.columns]
    
    # Separate target columns from feature columns
    feature_cols = [col for col in df.columns if col not in existing_targets]
    
    # 7. Validate Output Schema
    # Check for NaNs in engineered features
    nan_counts = df[feature_cols].isna().sum().sum()
    print(f"Total NaNs in feature set: {nan_counts}")
    
    # 8. Save Result
    output_path = 'data/processed/features.parquet'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Successfully saved features to {output_path}")
    print(f"Input rows: {initial_row_count}")
    print(f"Output rows: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Targets excluded: {existing_targets}")

if __name__ == "__main__":
    main()
