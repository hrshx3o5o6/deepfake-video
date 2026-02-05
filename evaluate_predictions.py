"""
Evaluation script for deepfake detection predictions.
Calculates accuracy, log loss, and weighted leaderboard score using NumPy.
Formula: Score = 0.7 * LogLoss + 0.3 * (100 - Accuracy%)
"""

import pandas as pd
import numpy as np
import sys

# ==========================================
# CONFIGURATION
# ==========================================
GROUND_TRUTH_PATH = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test_labels.csv"
PREDICTIONS_PATH = "test_predictions_final.csv"

def calculate_metrics_manually(y_true, y_pred, y_prob):
    """
    Calculate Accuracy and Log Loss using pure NumPy.
    """
    # Accuracy
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    # Log Loss
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    
    # Log loss formula: -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
    log_loss_vals = -(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    log_loss = np.mean(log_loss_vals)
    
    return accuracy, log_loss

def calculate_leaderboard_score(accuracy, log_loss):
    """
    Calculate the final weighted score based on the provided formula.
    finalScore = 0.7 * errorRate + 0.3 * (100 - accuracy)
    
    Note: Lower score is better.
    """
    accuracy_percent = accuracy * 100
    
    # Formula components
    term1 = 0.7 * log_loss
    term2 = 0.3 * (100 - accuracy_percent)
    
    final_score = term1 + term2
    
    return final_score

def evaluate_predictions():
    print("=" * 60)
    print("� DEEPFAKE LEADERBOARD EVALUATION")
    print("=" * 60)
    
    # 1. Load Data
    try:
        print(f"📂 Loading ground truth...")
        df_truth = pd.read_csv(GROUND_TRUTH_PATH)
        
        print(f"📂 Loading predictions...")
        df_pred = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Merge Data
    # We must ensure we align predictions with ground truth by filename
    df_merged = pd.merge(df_truth, df_pred, on='filename', suffixes=('_true', '_pred'))
    
    print(f"✅ Evaluated {len(df_merged)} videos (out of {len(df_truth)} total test videos)")
    
    if len(df_merged) == 0:
        print("❌ No matching filenames found between predictions and ground truth!")
        return

    # 3. Extract Arrays
    y_true = df_merged['label_true'].values.astype(int)
    y_pred = df_merged['label_pred'].values.astype(int)
    y_prob = df_merged['probability'].values.astype(float)
    
    # 4. Calculate Metrics
    accuracy, log_loss = calculate_metrics_manually(y_true, y_pred, y_prob)
    
    # 5. Calculate Final Score
    final_score = calculate_leaderboard_score(accuracy, log_loss)
    
    # 6. Display Results
    print("\n" + "-" * 30)
    print("📊 METRICS")
    print("-" * 30)
    print(f"✅ Accuracy:   {accuracy*100:.2f}%")
    print(f"❌ Log Loss:   {log_loss:.4f}")
    print("-" * 30)
    
    print("\n" + "=" * 30)
    print(f"⭐️ FINAL SCORE: {final_score:.4f}")
    print("=" * 30)
    print("(Lower is better)")
    
    # Optional: Per-class breakdown
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    
    acc_real = np.mean(y_true[real_mask] == y_pred[real_mask])
    acc_fake = np.mean(y_true[fake_mask] == y_pred[fake_mask])
    
    print(f"\n🔍 Breakdown:")
    print(f"   Real Video Accuracy: {acc_real*100:.1f}%")
    print(f"   Fake Video Accuracy: {acc_fake*100:.1f}%")

if __name__ == "__main__":
    evaluate_predictions()
