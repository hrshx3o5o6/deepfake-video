#optimized for t4-gpus
#not optimised for Apple ARM chips

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
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # DATA_ROOT should point to where 'real' and 'fake' folders are located
    VIDEO_ROOT = "/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/train"
    
    # We will save processed faces here
    PROCESSED_DIR = '/Volumes/Harshas ssd/development/ieee-ml-hack-data'
    
    # HYPERPARAMETERS
    SEQ_LENGTH = 20        
    IMG_SIZE = 224         # EfficientNet-B0 expects 224x224
    BATCH_SIZE = 16        # Larger batch for stability
    EPOCHS = 5             # 5 epochs for small dataset
    
    # TRANSFORMER HYPERPARAMETERS (smaller for 600 videos)
    D_MODEL = 256          # Smaller dimension to prevent overfitting
    NHEAD = 4              # Fewer attention heads
    NUM_ENCODER_LAYERS = 2 # 2 layers is enough for small data
    DIM_FEEDFORWARD = 512  # Smaller FFN
    DROPOUT = 0.3          # Higher dropout for regularization
    
    # LEARNING RATE & WARMUP (aggressive for small dataset)
    LR = 1e-4              # Higher LR for faster learning
    WARMUP_STEPS = 50      # Minimal warmup
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0    # Gradient clipping
    
    # DEVICE CONFIGURATION
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using Device: {Config.DEVICE}")

# ==========================================
# 2. POSITIONAL ENCODING
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Uses sinusoidal encoding as in 'Attention is All You Need'.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==========================================
# 3. MODEL ARCHITECTURE (EFFICIENTNET + TRANSFORMER)
# ==========================================
class EfficientNetTransformer(nn.Module):
    """
    Hybrid architecture combining:
    - EfficientNet-B0 for spatial feature extraction (per-frame)
    - Transformer Encoder for temporal modeling (across frames)
    - BERT-style CLS token for sequence classification
    """
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(EfficientNetTransformer, self).__init__()
        
        # ===== SPATIAL FEATURE EXTRACTOR (EfficientNet-B0) =====
        print("📥 Loading EfficientNet-B0 Weights...")
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Freeze CNN parameters initially (can unfreeze later for fine-tuning)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ===== PROJECTION LAYER =====
        # EfficientNet-B0 outputs 1280-dim features
        # Project to transformer dimension
        self.feature_projection = nn.Linear(1280, d_model)
        
        # ===== CLS TOKEN (BERT-style) =====
        # Learnable token prepended to sequence for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # ===== POSITIONAL ENCODING =====
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # ===== TRANSFORMER ENCODER =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            activation='gelu'   # GELU activation (used in BERT)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===== CLASSIFICATION HEAD =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Video tensor of shape (batch, seq_len, channels, height, width)
        Returns:
            Predictions of shape (batch, 1) with values in [0, 1]
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # ===== SPATIAL ENCODING =====
        # Flatten batch and sequence dimensions for CNN processing
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Extract spatial features using EfficientNet
        x = self.features(c_in)      # (batch*seq, 1280, 7, 7)
        x = self.avgpool(x)           # (batch*seq, 1280, 1, 1)
        x = torch.flatten(x, 1)       # (batch*seq, 1280)
        
        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1)  # (batch, seq, 1280)
        
        # Project to transformer dimension
        x = self.feature_projection(x)  # (batch, seq, d_model)
        
        # ===== PREPEND CLS TOKEN =====
        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)
        
        # ===== ADD POSITIONAL ENCODING =====
        x = self.positional_encoding(x)
        
        # ===== TEMPORAL ENCODING (TRANSFORMER) =====
        x = self.transformer_encoder(x)  # (batch, seq+1, d_model)
        
        # ===== EXTRACT CLS TOKEN OUTPUT =====
        # Only use the CLS token (first position) for classification
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # ===== CLASSIFICATION =====
        out = self.classifier(cls_output)  # (batch, 1)
        return out
    
    def unfreeze_efficientnet(self):
        """Unfreeze EfficientNet layers for fine-tuning"""
        print("🔓 Unfreezing EfficientNet layers for fine-tuning...")
        for param in self.features.parameters():
            param.requires_grad = True

# ==========================================
# 4. HELPER: BUILD DATASET FROM FOLDERS
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
                vid_id = os.path.splitext(vid_name)[0]
                
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
# 5. PREPROCESSING
# ==========================================
def preprocess_videos(df):
    """Extracts faces using MTCNN"""
    if os.path.exists(Config.PROCESSED_DIR) and len(os.listdir(Config.PROCESSED_DIR)) > 0:
        print(f"ℹ️  Processed directory exists. Skipping face extraction.")
        return

    print(f"🚀 Starting Face Extraction...")
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=Config.DEVICE, post_process=False)
    
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                try:
                    mtcnn(frame_pil, save_path=os.path.join(save_folder, f"{count}.png"))
                    count += 1
                except Exception:
                    pass 
            
            current_frame += 1
        cap.release()

# ==========================================
# 6. DATASET LOADER
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
# 7. LEARNING RATE SCHEDULER WITH WARMUP
# ==========================================
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a learning rate scheduler with linear warmup and linear decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

# ==========================================
# 8. TRAINING ENGINE
# ==========================================
def train_model():
    # 1. Build DataFrame from Folders
    df = create_dataset_dataframe()
    
    # 2. Extract Faces (if not done)
    preprocess_videos(df)
    
    # 3. Train/Val Split (80/20)
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
    print(f"🏗️  Building EfficientNet-Transformer Model on {Config.DEVICE}...")
    model = EfficientNetTransformer(
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_ENCODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    # Training Loop
    print("🔥 Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # ===== TRAINING PHASE =====
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
            
            # Gradient clipping (important for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
        # ===== VALIDATION PHASE =====
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
        print(f"   Learning Rate  : {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 40)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "efficientnet_transformer_best.pth")
            print(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Optional: Unfreeze EfficientNet after epoch 2 for fine-tuning
        if epoch == 2:
            model.unfreeze_efficientnet()
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5

    # Save final model
    torch.save(model.state_dict(), "efficientnet_transformer_final.pth")
    print("✅ Final model saved as efficientnet_transformer_final.pth")

if __name__ == "__main__":
    train_model()
