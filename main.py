import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import glob
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # DATA_ROOT should point to where 'real' and 'fake' folders are located
    # Based on your structure: ieee-ml-hack-data -> archive -> train -> (real/fake)
    VIDEO_ROOT = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/train"
    
    # We will save processed faces here
    PROCESSED_DIR = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/processed_faces"
    
    # HYPERPARAMETERS
    SEQ_LENGTH = 20        
    IMG_SIZE = 224         
    BATCH_SIZE = 16        
    EPOCHS = 10
    LR = 1e-4
    
    # APPLE SILICON OPTIMIZATION
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"✅ Using Device: {Config.DEVICE}")

# ==========================================
# 2. HELPER: BUILD DATASET FROM FOLDERS
# ==========================================
def create_dataset_dataframe():
    """
    Scans VIDEO_ROOT for 'real' and 'fake' folders.
    Returns a pandas DataFrame with video paths and labels.
    """
    data_list = []
    classes = {
        'real': 0,
        'fake': 1,
    }
    
    print(f"📂 Scanning for data in: {Config.VIDEO_ROOT}")
    
    found_something = False
    
    for folder_name, label in classes.items():
        folder_path = os.path.join(Config.VIDEO_ROOT, folder_name)
        if os.path.isdir(folder_path):
            found_something = True
            print(f"   Found class '{folder_name}' -> Label {label}")
            
            videos = glob.glob(os.path.join(folder_path, "*.mp4"))
            for vid_path in videos:
                vid_name = os.path.basename(vid_path)
                
                # --- CHANGED HERE ---
                # Now using ONLY the filename (without extension) as the ID
                vid_id = os.path.splitext(vid_name)[0]
                # --------------------
                
                data_list.append({
                    'video_path': vid_path,
                    'video_id': vid_id,
                    'label': label
                })
    
    if not found_something:
        print("❌ Error: Could not find 'real' or 'fake' folders.")
        sys.exit()
        
    df = pd.DataFrame(data_list)
    print(f"✅ Found {len(df)} total videos.")
    return df

# ==========================================
# 3. PREPROCESSING (FIXED FOR M5)
# ==========================================
def preprocess_videos(df):
    """Extracts faces using CPU to avoid MPS Adaptive Pool bug."""
    if os.path.exists(Config.PROCESSED_DIR) and len(os.listdir(Config.PROCESSED_DIR)) > 0:
        print(f"ℹ️  Processed directory exists. Skipping face extraction.")
        return

    print(f"🚀 Starting Face Extraction...")
    # NOTE: We force CPU here because MTCNN crashes on MPS due to a PyTorch bug
    print("⚠️  Forcing MTCNN to CPU to avoid 'Adaptive Pool MPS' crash.")
    
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    # CHANGE IS HERE: device='cpu'
    mtcnn = MTCNN(keep_all=False, select_largest=True, device='cpu', post_process=False)
    
    # Iterate through our dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        video_path = row['video_path']
        video_id = row['video_id']
        
        save_folder = os.path.join(Config.PROCESSED_DIR, video_id)
        os.makedirs(save_folder, exist_ok=True)
        
        # Check if already done
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                try:
                    # This now runs on CPU, so no crash!
                    mtcnn(frame_pil, save_path=os.path.join(save_folder, f"{count}.png"))
                    count += 1
                except Exception:
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

        # Padding logic
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
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super(ResNetLSTM, self).__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1] 
        self.cnn = nn.Sequential(*modules)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(c_in)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out)

# ==========================================
# 6. TRAINING ENGINE
# ==========================================
def train_model():
    # 1. Build DataFrame from Folders
    df = create_dataset_dataframe()
    
    # 2. Extract Faces (if not done)
    preprocess_videos(df)
    
    # 3. Train/Val Split (80/20)
    # Shuffle first to mix reals and fakes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Transforms
    tfms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Loaders
    train_ds = DeepfakeDataset(Config.PROCESSED_DIR, train_df, transform=tfms)
    val_ds = DeepfakeDataset(Config.PROCESSED_DIR, val_df, transform=tfms)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model Setup
    print(f"🏗️  Building Model on {Config.DEVICE}...")
    model = ResNetLSTM().to(Config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    # Training Loop
    print("🔥 Starting Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for videos, labels in loop:
            videos = videos.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Evaluation Loop
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

    torch.save(model.state_dict(), "deepfake_detector_m5.pth")
    print("✅ Model saved as deepfake_detector_m5.pth")

if __name__ == "__main__":
    train_model()