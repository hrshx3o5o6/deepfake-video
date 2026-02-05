"""
Evaluation script for deepfake detection predictions.
Calculates accuracy and log loss by comparing predictions with ground truth labels.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report

# ==========================================
# CONFIGURATION
# ==========================================
GROUND_TRUTH_PATH = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test_labels.csv"
PREDICTIONS_PATH = "test_predictions_final.csv"

# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================
def evaluate_predictions():
    """
    Evaluates model predictions against ground truth labels.
    Calculates accuracy, log loss, and other metrics.
    """
    
    print("=" * 70)
    print("🎯 DEEPFAKE DETECTION - MODEL EVALUATION")
    print("=" * 70)
    
    # 1. Load ground truth labels
    print(f"\n📂 Loading ground truth from: {GROUND_TRUTH_PATH}")
    df_truth = pd.read_csv(GROUND_TRUTH_PATH)
    print(f"   ✓ Loaded {len(df_truth)} ground truth labels")
    
    # 2. Load predictions
    print(f"\n📂 Loading predictions from: {PREDICTIONS_PATH}")
    df_pred = pd.read_csv(PREDICTIONS_PATH)
    print(f"   ✓ Loaded {len(df_pred)} predictions")
    
    # 3. Merge on filename
    print(f"\n🔗 Merging predictions with ground truth by filename...")
    df_merged = pd.merge(
        df_truth, 
        df_pred, 
        on='filename', 
        suffixes=('_true', '_pred')
    )
    print(f"   ✓ Matched {len(df_merged)} videos")
    
    # Check if all videos were matched
    if len(df_merged) != len(df_truth):
        missing = len(df_truth) - len(df_merged)
        print(f"   ⚠️  Warning: {missing} videos from ground truth not found in predictions")
    
    if len(df_merged) != len(df_pred):
        extra = len(df_pred) - len(df_merged)
        print(f"   ⚠️  Warning: {extra} predictions not found in ground truth")
    
    # 4. Extract arrays
    y_true = df_merged['label_true'].values
    y_pred = df_merged['label_pred'].values
    y_prob = df_merged['probability'].values
    
    # 5. Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 6. Calculate Log Loss
    # Clip probabilities to avoid log(0) errors
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logloss = log_loss(y_true, y_prob_clipped)
    
    # 7. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 8. Calculate additional metrics
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 9. Print Results
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Log Loss:    {logloss:.4f}")
    print(f"   Total Videos: {len(df_merged)}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Real    Fake")
    print(f"   Actual Real   {tn:4d}    {fp:4d}")
    print(f"         Fake    {fn:4d}    {tp:4d}")
    
    print(f"\n🔍 Per-Class Metrics:")
    print(f"   Real Videos (Label 0):")
    print(f"      Precision: {precision_real:.4f}")
    print(f"      Recall:    {recall_real:.4f}")
    print(f"   ")
    print(f"   Fake Videos (Label 1):")
    print(f"      Precision: {precision_fake:.4f}")
    print(f"      Recall:    {recall_fake:.4f}")
    
    # 10. Distribution analysis
    real_count_true = (y_true == 0).sum()
    fake_count_true = (y_true == 1).sum()
    real_count_pred = (y_pred == 0).sum()
    fake_count_pred = (y_pred == 1).sum()
    
    print(f"\n📊 Label Distribution:")
    print(f"   Ground Truth:  Real={real_count_true}, Fake={fake_count_true}")
    print(f"   Predictions:   Real={real_count_pred}, Fake={fake_count_pred}")
    
    # 11. Probability distribution analysis
    print(f"\n📈 Probability Statistics:")
    print(f"   Min:  {y_prob.min():.4f}")
    print(f"   Max:  {y_prob.max():.4f}")
    print(f"   Mean: {y_prob.mean():.4f}")
    print(f"   Std:  {y_prob.std():.4f}")
    
    # 12. Show some misclassified examples
    df_merged['correct'] = (df_merged['label_true'] == df_merged['label_pred'])
    df_wrong = df_merged[~df_merged['correct']].copy()
    
    if len(df_wrong) > 0:
        print(f"\n❌ Misclassified Videos: {len(df_wrong)}")
        print(f"\n   Sample of misclassified videos (first 10):")
        print("   " + "-" * 66)
        print(f"   {'Filename':<40} {'True':>4} {'Pred':>4} {'Prob':>6}")
        print("   " + "-" * 66)
        for _, row in df_wrong.head(10).iterrows():
            print(f"   {row['filename']:<40} {row['label_true']:>4} {row['label_pred']:>4} {row['probability']:>6.2f}")
    else:
        print(f"\n✅ Perfect predictions! No misclassifications!")
    
    # 13. Save detailed results
    output_file = "evaluation_results.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    # Return metrics for programmatic use
    return {
        'accuracy': accuracy,
        'log_loss': logloss,
        'confusion_matrix': cm,
        'total_samples': len(df_merged)
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    metrics = evaluate_predictions()
