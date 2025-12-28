import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import glob
import sys
from ultralytics import YOLO  # Required for YOLOv12

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # --- UPDATE THESE PATHS FOR YOUR LOCAL MAC ---
    # Point this to the folder containing 'real' and 'fake' subfolders
    VIDEO_ROOT = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/train"  
    
    # We will save processed faces here
    PROCESSED_DIR = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/processed_faces_yolo"
    
    # PATH TO YOUR YOLO WEIGHTS
    YOLO_PATH = "face_models/yolov12m-face.pt"  # Make sure this file exists locally
    
    # HYPERPARAMETERS
    SEQ_LENGTH = 20        
    IMG_SIZE = 224         
    BATCH_SIZE = 8        
    EPOCHS = 10
    LR = 1e-4
    
    # MAC M5 OPTIMIZATION (MPS)
    # We explicitly force MPS as requested
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✅ Apple Silicon (MPS) detected & enabled.")
    else:
        print("⚠️ MPS not available. Falling back to CPU (Slow).")
        DEVICE = torch.device("cpu")

# ==========================================
# 2. HELPER: BUILD DATASET FROM FOLDERS
# ==========================================
def create_dataset_dataframe():
    data_list = []
    classes = {'real': 0, 'fake': 1}
    
    print(f"📂 Scanning for data in: {Config.VIDEO_ROOT}")
    
    # Validate YOLO path
    if not os.path.exists(Config.YOLO_PATH):
        print(f"❌ Error: YOLO weights not found at {Config.YOLO_PATH}")
        sys.exit()

    found_something = False
    
    for folder_name, label in classes.items():
        folder_path = os.path.join(Config.VIDEO_ROOT, folder_name)
        if os.path.isdir(folder_path):
            found_something = True
            print(f"   Found class '{folder_name}' -> Label {label}")
            
            # Extensions to look for
            videos = []
            for ext in ['*.mp4', '*.avi', '*.mov']:
                videos.extend(glob.glob(os.path.join(folder_path, ext)))
                
            for vid_path in videos:
                vid_name = os.path.basename(vid_path)
                vid_id = os.path.splitext(vid_name)[0]
                
                data_list.append({
                    'video_path': vid_path,
                    'video_id': vid_id,
                    'label': label
                })
    
    if not found_something:
        print("❌ Error: Could not find 'real' or 'fake' folders.")
        print("   Please check Config.VIDEO_ROOT path.")
        sys.exit()
        
    df = pd.DataFrame(data_list)
    print(f"✅ Found {len(df)} total videos.")
    return df

# ==========================================
# 3. PREPROCESSING (YOLOv12 on MPS)
# ==========================================
def preprocess_videos(df):
    if os.path.exists(Config.PROCESSED_DIR) and len(os.listdir(Config.PROCESSED_DIR)) > 0:
        print(f"ℹ️  Processed directory exists. Skipping face extraction.")
        # NOTE: If you want to force re-run, delete the folder manually
        return

    print(f"🚀 Starting Face Extraction using {Config.YOLO_PATH} on {Config.DEVICE}...")
    
    # LOAD YOLO MODEL
    # Ultralytics handles device placement automatically if we pass device='mps' during inference
    try:
        yolo_model = YOLO(Config.YOLO_PATH)
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        sys.exit()
    
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        video_path = row['video_path']
        video_id = row['video_id']
        
        save_folder = os.path.join(Config.PROCESSED_DIR, video_id)
        os.makedirs(save_folder, exist_ok=True)
        
        if len(os.listdir(save_folder)) >= Config.SEQ_LENGTH:
            continue
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0: 
            cap.release()
            continue 
            
        frame_idxs = np.linspace(0, total_frames - 1, Config.SEQ_LENGTH, dtype=int)
        
        count = 0
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame in frame_idxs:
                # RUN YOLO INFERENCE ON MPS
                # conf=0.5 filters out low confidence detections
                results = yolo_model(frame, device=Config.DEVICE, verbose=False, conf=0.5)
                
                best_face = None
                max_area = 0
                
                # Parse Results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # xyxy format: x1, y1, x2, y2
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Calculate area to pick the largest face (main subject)
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        
                        if area > max_area:
                            max_area = area
                            # Ensure coordinates are within frame bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            best_face = frame[y1:y2, x1:x2]

                # Save if a face was found
                if best_face is not None and best_face.size > 0:
                    try:
                        face_rgb = cv2.cvtColor(best_face, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(face_rgb)
                        # Resize here to save disk space and loading time
                        img_pil = img_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE))
                        img_pil.save(os.path.join(save_folder, f"{count}.png"))
                        count += 1
                    except Exception as e:
                        pass
            
            current_frame += 1
        cap.release()

# ==========================================
# 4. DATASET LOADER
# ==========================================
class DeepfakeDataset(Dataset):
    def __init__(self, processed_dir, df, transform=None):
        self.processed_dir = processed_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        label = float(row['label'])
        
        video_folder = os.path.join(self.processed_dir, video_id)
        images = []
        
        if os.path.isdir(video_folder):
            # Sort numerically (0.png, 1.png...)
            image_files = sorted(glob.glob(os.path.join(video_folder, "*.png")), 
                               key=lambda x: int(os.path.basename(x).split('.')[0]))
        else:
            image_files = []
        
        for img_path in image_files[:Config.SEQ_LENGTH]:
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except:
                pass

        # Padding logic (if YOLO missed faces in some frames)
        if len(images) == 0:
            blank = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)
            images = [blank for _ in range(Config.SEQ_LENGTH)]
        elif len(images) < Config.SEQ_LENGTH:
            while len(images) < Config.SEQ_LENGTH:
                images.append(images[-1])
            
        images_tensor = torch.stack(images) 
        return images_tensor, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
class EfficientNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super(EfficientNetLSTM, self).__init__()
        
        print("📥 Loading EfficientNet-B0...")
        weights = models.EfficientNet_B0_Weights.DEFAULT 
        backbone = models.efficientnet_b0(weights=weights)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Freeze CNN
        for param in self.features.parameters():
            param.requires_grad = False
            
        input_size = 1280
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        x = self.features(c_in)     
        x = self.avgpool(x)         
        x = torch.flatten(x, 1)     
        
        lstm_in = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(lstm_in)
        last_out = lstm_out[:, -1, :] 
        out = self.fc(last_out)
        return self.sigmoid(out)

# ==========================================
# 6. TRAINING ENGINE
# ==========================================
def train_model():
    # 1. Setup Data
    df = create_dataset_dataframe()
    preprocess_videos(df)
    
    # 2. Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # 3. Transforms
    tfms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Loaders (num_workers=0 is safer on Mac to avoid multiprocessing spawn issues)
    train_ds = DeepfakeDataset(Config.PROCESSED_DIR, train_df, transform=tfms)
    val_ds = DeepfakeDataset(Config.PROCESSED_DIR, val_df, transform=tfms)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 5. Model
    print(f"🏗️  Building Model on {Config.DEVICE}...")
    model = EfficientNetLSTM().to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    # 6. Training
    print("🔥 Starting Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for videos, labels in loop:
            # Move data to MPS
            videos = videos.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).unsqueeze(1)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Log Loss : {avg_train_loss:.4f}")
        print(f"   Val Log Loss   : {avg_val_loss:.4f}")
        print(f"   Val Accuracy   : {accuracy:.2f}%")
        print("-" * 40)

    torch.save(model.state_dict(), "mac_m5_detector.pth")
    print("✅ Model saved as mac_m5_detector.pth")

if __name__ == "__main__":
    train_model()