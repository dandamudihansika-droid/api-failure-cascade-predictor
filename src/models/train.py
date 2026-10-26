import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(file_path):
    df = pd.read_parquet(file_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    target_col = 'cascade_level'
    leakage_cols = ['cascade_risk_score', 'cascade_level', 'blast_radius']
    identifier_cols = ['timestamp', 'service_name']
    features = [col for col in df.columns if col not in leakage_cols and col not in identifier_cols]
    return df, features, target_col

def chronological_split(df, train_size=0.7, val_size=0.15, test_size=0.15, *args, **kwargs):
    if isinstance(train_size, (list, tuple, pd.Index, str)):
        train_size = kwargs.get('train_ratio', 0.7)
    if isinstance(val_size, (list, tuple, pd.Index, str)):
        val_size = kwargs.get('val_ratio', 0.15)
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    pr_auc_scores = {}
    if y_prob is not None:
        for i, name in enumerate(target_names):
            # Binary labels for current class i
            y_binary = (y_test == i).astype(int)
            # Probabilities for class i (matching class IDs with probability columns)
            if hasattr(model, "classes_") and i in model.classes_:
                col_idx = int(np.where(model.classes_ == i)[0][0])
                prob = y_prob[:, col_idx]
            else:
                prob = y_prob[:, i]
            precision, recall, _ = precision_recall_curve(y_binary, prob)
            pr_auc_scores[name] = auc(recall, precision)
            
    return {
        'macro_f1': macro_f1,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'pr_auc': pr_auc_scores
    }

def main():
    data_path = 'data/processed/features.parquet'
    df, features, target_col = load_and_prepare_data(data_path)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    target_names = le.classes_
    
    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    
    # Baseline
    baseline = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, target_names)
    
    # Challenger
    challenger = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight='balanced_subsample')
    challenger.fit(X_train, y_train)
    challenger_metrics = evaluate_model(challenger, X_test, y_test, target_names)
    
    importances = challenger.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    joblib.dump(challenger, 'models/cascade_predictor.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    report_data = {
        'baseline_metrics': baseline_metrics,
        'challenger_metrics': challenger_metrics,
        'feature_importance': feat_imp.to_dict(),
        'class_distribution': df[target_col].value_counts().to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    with open('reports/evaluation_report.json', 'w') as f:
        json.dump(report_data, f, indent=4)

if __name__ == "__main__":
    import json
    from datetime import datetime
    main()
