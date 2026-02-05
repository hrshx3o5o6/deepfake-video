# Yantra ML Hack - Deepfake Detection Toolkit

This toolkit provides scripts for seeding data, running deepfake detection inference on test videos, and evaluating model performance.

## 🛠️ Setup

1. **Install Dependencies**
   ```bash
   uv sync
   ```

2. **Environment Variables**
   Create a `.env` file with your Supabase credentials (see `.env.example` or existing `.env`).

## 🚀 Scripts Overview

### 1. Seed Supabase Database (`seed_supabase.py`)
Seeds the `User` table in Supabase with participant data from a CSV file.

**Features:**
- Handles specific column mapping (`Reg. No.`, `Name`, `Email` → `regNo`, `name`, `email`)
- Deduplicates emails (keeps last occurrence)
- Uses `upsert` to update existing records
- Batched processing for efficiency

**Usage:**
```bash
uv run seed_supabase.py path/to/users.csv
```

### 2. Deepfake Inference (`predict_test_videos.py`)
Runs the trained EfficientNet-LSTM model on test videos to generate predictions.

**Pipeline:**
1. **Extract Faces**: Uses MTCNN to extract 20 frames per video (saved to `processed_test_faces/`)
2. **Inference**: Loads `efficientnet_b0_detector.pth` and predicts "fake" probability
3. **Output**: Generates `test_predictions_final.csv`

**Usage:**
```bash
uv run predict_test_videos.py
```
*Note: Optimized for Mac M-series chips (Metal Performance Shaders).*

### 3. Evaluation & Leaderboard (`evaluate_predictions.py`)
Calculates the final leaderboard score using the competition metric.

**Formula:**
`Score = 0.7 * LogLoss + 0.3 * (100 - Accuracy%)`
*(Lower is better)*

**Usage:**
```bash
uv run evaluate_predictions.py
```

## 📂 Project Structure

- `seed_supabase.py`: Data seeding utility
- `predict_test_videos.py`: Main inference engine
- `evaluate_predictions.py`: Scoring and metrics
- `main_efficientnetlstm.py`: Original training code
- `efficientnet_b0_detector.pth`: Trained model weights
