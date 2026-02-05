"""
Inference script for deepfake detection on test videos.
Loads the trained EfficientNet-LSTM model and generates predictions.
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Test data paths
    TEST_VIDEO_DIR = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test"
    PROCESSED_TEST_DIR = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/processed_test_faces"
    
    # Model path
    MODEL_PATH = "efficientnet_b0_detector.pth"
    
    # Output
    OUTPUT_CSV = "test_predictions_final.csv"
    
    # Model hyperparameters (must match training)
    SEQ_LENGTH = 20
    IMG_SIZE = 224
    BATCH_SIZE = 8  # Lower batch size for inference on M-series chips
    
    # Device configuration (optimized for Mac M-series)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"✅ Using Device: {Config.DEVICE}")

# ==========================================
# 2. MODEL ARCHITECTURE (SAME AS TRAINING)
# ==========================================
class EfficientNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super(EfficientNetLSTM, self).__init__()
        
        # Load EfficientNet-B0 (Pretrained)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Freeze CNN parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # EfficientNet-B0 output size is 1280
        input_size = 1280
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten time dimension for CNN
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Pass through EfficientNet
        x = self.features(c_in)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Restore time dimension for LSTM
        lstm_in = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out)

# ==========================================
# 3. FACE EXTRACTION FOR TEST VIDEOS
# ==========================================
def extract_faces_from_test_videos():
    """Extract faces from all test videos."""
    print(f"🚀 Starting Face Extraction for Test Videos...")
    
    # Use CPU for MTCNN to avoid MPS issues
    mtcnn = MTCNN(keep_all=False, select_largest=True, device='cpu', post_process=False)
    
    os.makedirs(Config.PROCESSED_TEST_DIR, exist_ok=True)
    
    # Get all test videos
    test_videos = glob.glob(os.path.join(Config.TEST_VIDEO_DIR, "*.mp4"))
    print(f"📹 Found {len(test_videos)} test videos")
    
    for video_path in tqdm(test_videos, desc="Extracting Faces"):
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        
        save_folder = os.path.join(Config.PROCESSED_TEST_DIR, video_id)
        os.makedirs(save_folder, exist_ok=True)
        
        # Skip if already processed
        if len(os.listdir(save_folder)) >= Config.SEQ_LENGTH:
            continue
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            continue
        
        # Sample frames uniformly
        frame_idxs = np.linspace(0, total_frames - 1, Config.SEQ_LENGTH, dtype=int)
        
        count = 0
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in frame_idxs:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                try:
                    mtcnn(frame_pil, save_path=os.path.join(save_folder, f"{count}.png"))
                    count += 1
                except Exception:
                    pass
            
            current_frame += 1
        
        cap.release()
    
    print("✅ Face extraction complete!")

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_on_test_videos():
    """Load model and generate predictions for all test videos."""
    
    # 1. Load the trained model
    print(f"📥 Loading model from {Config.MODEL_PATH}...")
    model = EfficientNetLSTM().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # 2. Prepare transforms
    tfms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Get all test videos
    test_videos = sorted(glob.glob(os.path.join(Config.TEST_VIDEO_DIR, "*.mp4")))
    
    results = []
    
    print(f"🔮 Generating predictions for {len(test_videos)} videos...")
    
    # Process in batches
    batch_videos = []
    batch_filenames = []
    
    for video_path in tqdm(test_videos, desc="Predicting"):
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        
        # Load processed faces
        video_folder = os.path.join(Config.PROCESSED_TEST_DIR, video_id)
        images = []
        
        if os.path.isdir(video_folder):
            image_files = sorted(
                glob.glob(os.path.join(video_folder, "*.png")),
                key=lambda x: int(os.path.basename(x).split('.')[0])
            )
        else:
            image_files = []
        
        # Load and transform images
        for img_path in image_files[:Config.SEQ_LENGTH]:
            try:
                img = Image.open(img_path).convert('RGB')
                img = tfms(img)
                images.append(img)
            except:
                pass
        
        # Handle padding
        if len(images) == 0:
            blank = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)
            images = [blank for _ in range(Config.SEQ_LENGTH)]
        elif len(images) < Config.SEQ_LENGTH:
            while len(images) < Config.SEQ_LENGTH:
                images.append(images[-1])
        
        images_tensor = torch.stack(images)
        batch_videos.append(images_tensor)
        batch_filenames.append(video_filename)
        
        # Process batch
        if len(batch_videos) == Config.BATCH_SIZE or video_path == test_videos[-1]:
            batch_tensor = torch.stack(batch_videos).to(Config.DEVICE)
            
            with torch.no_grad():
                predictions = model(batch_tensor)
            
            # Extract results
            for i, filename in enumerate(batch_filenames):
                probability = predictions[i].item()
                label = 1 if probability > 0.5 else 0
                
                results.append({
                    'filename': filename,
                    'label': label,
                    'probability': round(probability, 2)
                })
            
            # Reset batch
            batch_videos = []
            batch_filenames = []
    
    # 4. Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(Config.OUTPUT_CSV, index=False)
    
    print(f"\n✅ Predictions saved to: {Config.OUTPUT_CSV}")
    print(f"📊 Total predictions: {len(df_results)}")
    print(f"   Real (0): {(df_results['label'] == 0).sum()}")
    print(f"   Fake (1): {(df_results['label'] == 1).sum()}")
    print(f"\n📋 First 10 predictions:")
    print(df_results.head(10).to_string(index=False))

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    print("=" * 60)
    print("🎬 DEEPFAKE DETECTION - TEST VIDEO INFERENCE")
    print("=" * 60)
    
    # Step 1: Extract faces from test videos
    if not os.path.exists(Config.PROCESSED_TEST_DIR) or len(os.listdir(Config.PROCESSED_TEST_DIR)) == 0:
        extract_faces_from_test_videos()
    else:
        print(f"ℹ️  Processed test faces already exist at {Config.PROCESSED_TEST_DIR}")
        print("   Skipping face extraction. Delete the folder to re-process.")
    
    # Step 2: Generate predictions
    predict_on_test_videos()
    
    print("\n" + "=" * 60)
    print("🎉 INFERENCE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
