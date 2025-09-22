# ============================
# 1. SETUP AND INSTALLATIONS
# ============================

# Install required packages
!pip install torch torchvision torchaudio --quiet
!pip install librosa matplotlib seaborn scikit-learn pandas --quiet

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import gc
import sys
import multiprocessing
warnings.filterwarnings('ignore')

# Fix multiprocessing issues in Colab
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Clear memory and set device
torch.cuda.empty_cache() if torch.cuda.is_available() else None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================
# 2. DATASET DOWNLOAD (FIXED)
# ============================

def download_gtzan_dataset():
    """Download GTZAN dataset with proper error handling"""
    try:
        # Try multiple download sources
        print("Downloading GTZAN dataset...")
        
        # Method 1: Direct download
        !wget -O gtzan.zip -q --no-check-certificate https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification
        
        if not os.path.exists('gtzan.zip'):
            print("Kaggle download failed, trying alternative source...")
            # Alternative download method
            !wget -O gtzan.zip -q --no-check-certificate https://github.com/mdeff/fma/releases/download/gtzan/gtzan.zip
        
        if os.path.exists('gtzan.zip'):
            print("✓ Dataset downloaded successfully")
            !unzip -o -q gtzan.zip
            return True
        else:
            print("❌ Download failed")
            return False
            
    except Exception as e:
        print(f"Download error: {e}")
        return False

# Download and setup dataset
if not os.path.exists('Data') and not os.path.exists('genres_original'):
    success = download_gtzan_dataset()
    if not success:
        print("Creating dummy dataset for testing...")
        # Create a minimal test dataset structure
        os.makedirs('Data/genres_original', exist_ok=True)
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop']
        for genre in genres:
            os.makedirs(f'Data/genres_original/{genre}', exist_ok=True)

# Find data directory
data_dir = None
possible_paths = [
    'Data/genres_original',
    'Data/genres', 
    'genres_original',
    'genres'
]

for path in possible_paths:
    if os.path.exists(path):
        data_dir = path
        print(f"Found dataset at: {data_dir}")
        break

if data_dir is None:
    # Search for any directory with .wav files
    for root, dirs, files in os.walk('.'):
        if any(f.endswith('.wav') for f in files):
            data_dir = root
            print(f"Found audio files in: {data_dir}")
            break

# ============================
# 3. OPTIMIZED DATASET CLASS
# ============================

class MusicDataset(Dataset):
    def __init__(self, file_paths, labels, sr=22050, duration=10, n_mfcc=20, 
                 augment=False, mixup_alpha=0.2):
        """
        Optimized Dataset with enhanced features and augmentation
        - Increased MFCCs from 13 to 20
        - Added delta and delta-delta features
        - Reduced duration to 10 seconds for more diverse sampling
        - Added mixup augmentation
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        
        # Known corrupted files in GTZAN
        corrupted_files = [
            'jazz.00054.wav',
            'jazz.00054.au'
        ]
        
        # Filter and validate files
        print("Validating audio files...")
        self.file_paths = []
        self.labels = []
        
        valid_count = 0
        for path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            # Skip corrupted files
            if any(corrupt in os.path.basename(path) for corrupt in corrupted_files):
                print(f"  Skipping corrupted file: {os.path.basename(path)}")
                continue
                
            # Quick validation - check if file exists and has reasonable size
            if os.path.exists(path) and os.path.getsize(path) > 10000:  # At least 10KB
                self.file_paths.append(path)
                self.labels.append(label)
                valid_count += 1
            else:
                print(f"  Invalid or missing file: {os.path.basename(path)}")
        
        print(f"✓ Valid files: {len(self.file_paths)}/{len(file_paths)} ({len(self.file_paths)/len(file_paths)*100:.1f}%)")
        
        # Pre-compute some statistics
        if len(self.file_paths) > 0:
            from collections import Counter
            label_counts = Counter(self.labels)
            print(f"  Files per class: {dict(label_counts)}")
        
        # Store all file paths and labels for mixup
        self.all_file_paths = self.file_paths.copy()
        self.all_labels = self.labels.copy()
    
    def __len__(self):
        return len(self.file_paths)
    
    def extract_features(self, y, sr):
        """Extract enhanced MFCC features with delta and delta-delta"""
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512,
            n_mels=128  # More mel filters for better representation
        )
        
        # Add delta and delta-delta features for temporal dynamics
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack all features (60 total features: 20 + 20 + 20)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load with librosa for better compatibility
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Ensure consistent length
            target_length = self.sr * self.duration
            if len(y) > target_length:
                # Random crop for training, center crop for validation
                if self.augment:
                    start = np.random.randint(0, len(y) - target_length + 1)
                    y = y[start:start + target_length]
                else:
                    # Center crop for validation
                    start = (len(y) - target_length) // 2
                    y = y[start:start + target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            # Mixup augmentation (very effective for small datasets!)
            mixup_label = label
            if self.augment and np.random.random() > 0.5:  # 50% chance
                # Get another random sample
                idx2 = np.random.randint(0, len(self.all_file_paths))
                y2, _ = librosa.load(self.all_file_paths[idx2], sr=self.sr, duration=self.duration)
                
                # Ensure same length
                if len(y2) > target_length:
                    y2 = y2[:target_length]
                else:
                    y2 = np.pad(y2, (0, target_length - len(y2)), mode='constant')
                
                # Mixup with random alpha
                alpha = np.random.uniform(0.1, self.mixup_alpha)
                y = (1 - alpha) * y + alpha * y2
                # Note: For simplicity, we keep the original label (could implement soft labels)
            
            # Other augmentations
            elif self.augment and np.random.random() > 0.3:  # 70% chance of other augmentations
                aug_choice = np.random.choice(['noise', 'shift', 'pitch'], p=[0.4, 0.3, 0.3])
                
                if aug_choice == 'noise':
                    # Add small amount of noise
                    noise_factor = np.random.uniform(0.002, 0.008)
                    y = y + np.random.normal(0, noise_factor, y.shape)
                
                elif aug_choice == 'shift':
                    # Time shift (smaller shifts for 10-second clips)
                    shift_samples = int(np.random.uniform(-0.05, 0.05) * len(y))
                    y = np.roll(y, shift_samples)
                
                elif aug_choice == 'pitch':
                    # Pitch shift
                    steps = np.random.choice([-2, -1, 1, 2])
                    y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
            
            # Extract enhanced features
            features = self.extract_features(y, self.sr)
            
            return torch.FloatTensor(features), mixup_label
            
        except Exception as e:
            print(f"Error loading {os.path.basename(audio_path)}: {e}")
            # Return zero features as fallback
            dummy_features = np.zeros((self.n_mfcc * 3, (self.sr * self.duration) // 512 + 1))
            return torch.FloatTensor(dummy_features), label

# ============================
# 4. OPTIMIZED CNN MODEL
# ============================

class MusicCNN(nn.Module):
    def __init__(self, input_channels=60, num_classes=10, dropout_conv=0.2, dropout_fc=0.4):
        """
        Optimized CNN with wider layers but same depth
        - Increased filter counts for better feature extraction
        - Adjusted dropout rates for better generalization
        """
        super(MusicCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block - wider filters
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Increased from 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increased from 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Increased from 128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_conv + 0.1)  # Slightly higher dropout in last conv
        )
        
        # Classifier - wider for richer representations
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),  # Increased from 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================
# 5. LABEL SMOOTHING LOSS
# ============================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing to improve generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        return -(one_hot * log_prob).sum(dim=-1).mean()

# ============================
# 6. DATA PREPARATION
# ============================

# Collect audio files
audio_files = []
labels = []
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

if data_dir and os.path.exists(data_dir):
    # Check for genre subdirectories
    genre_dirs_found = False
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.exists(genre_path):
            genre_dirs_found = True
            for file in os.listdir(genre_path):
                if file.endswith(('.wav', '.au')):
                    audio_files.append(os.path.join(genre_path, file))
                    labels.append(genre)
    
    # If no subdirectories, try flat structure
    if not genre_dirs_found:
        for file in os.listdir(data_dir):
            if file.endswith(('.wav', '.au')):
                genre = file.split('.')[0]
                if genre in genres:
                    audio_files.append(os.path.join(data_dir, file))
                    labels.append(genre)

print(f"Total audio files found: {len(audio_files)}")

# Create dummy data if no files found (for testing)
if len(audio_files) == 0:
    print("⚠️  No audio files found. Creating dummy data for testing.")
    # This allows the script to run for testing purposes
    for i in range(50):  # Create 50 dummy entries
        genre = genres[i % len(genres)]
        audio_files.append(f"dummy_{genre}_{i}.wav")
        labels.append(genre)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Check dataset balance
from collections import Counter
genre_counts = Counter(labels)
print(f"\nFiles per genre:")
for genre in genres:
    count = genre_counts.get(genre, 0)
    print(f"  {genre}: {count}")

# Check if dataset is severely imbalanced (difference > 20%)
max_count = max(genre_counts.values()) if genre_counts else 0
min_count = min(genre_counts.values()) if genre_counts else 0
imbalance_ratio = (max_count - min_count) / max_count if max_count > 0 else 0

if imbalance_ratio > 0.2:  # More than 20% imbalance
    print(f"Dataset is imbalanced (ratio: {imbalance_ratio:.2f}). Balancing to minimum count: {min_count}")
    balanced_files = []
    balanced_labels = []
    
    for genre in genres:
        genre_files = [f for f, l in zip(audio_files, labels) if l == genre]
        genre_labels = [l for l in labels if l == genre]
        
        # Use minimum count across all genres
        if len(genre_files) > min_count:
            genre_files = genre_files[:min_count]
            genre_labels = genre_labels[:min_count]
        
        balanced_files.extend(genre_files)
        balanced_labels.extend(genre_labels)
    
    audio_files = balanced_files
    labels = balanced_labels
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"Balanced dataset: {len(audio_files)} files")
else:
    print("✓ Dataset is reasonably balanced, using all available files")

print(f"Using {len(audio_files)} files for training")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    audio_files, encoded_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=encoded_labels
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ============================
# 7. OPTIMIZED DATALOADER SETUP
# ============================

# Create datasets with optimized parameters
train_dataset = MusicDataset(
    X_train, y_train, 
    sr=22050, 
    duration=10,  # Reduced from 30 seconds
    n_mfcc=20,    # Increased from 13
    augment=True,
    mixup_alpha=0.2
)

val_dataset = MusicDataset(
    X_val, y_val,
    sr=22050,
    duration=10,
    n_mfcc=20,
    augment=False
)

# Optimized batch size with gradient accumulation
batch_size = 16  # Smaller batch for better gradients

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0,  # Fixed for Colab
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print("✓ DataLoaders created successfully")
print(f"  Feature dimensions: 60 x ~430 (3x20 MFCCs with deltas)")

# ============================
# 8. TRAINING FUNCTIONS WITH GRADIENT ACCUMULATION
# ============================

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=2):
    """Training with gradient accumulation for effective larger batch size"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, (features, labels) in enumerate(tqdm(train_loader, desc='Training')):
        try:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Accumulate gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        except Exception as e:
            print(f"Training batch error: {e}")
            continue
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Standard validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='Validation'):
            try:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"Validation batch error: {e}")
                continue
    
    val_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_acc = 100 * correct / total if total > 0 else 0
    return val_loss, val_acc

def validate_with_tta(model, val_loader, device, n_aug=3):
    """Test-Time Augmentation for better validation accuracy"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='TTA Validation'):
            batch_preds = []
            features = features.to(device)
            
            # Original prediction
            outputs = model(features)
            batch_preds.append(F.softmax(outputs, dim=1))
            
            # Augmented predictions (simple time shifts)
            for _ in range(n_aug - 1):
                # Random time shift
                shift_amount = np.random.randint(-20, 20)
                shifted = torch.roll(features, shifts=shift_amount, dims=-1)
                outputs = model(shifted)
                batch_preds.append(F.softmax(outputs, dim=1))
            
            # Average predictions
            avg_preds = torch.stack(batch_preds).mean(dim=0)
            all_preds.append(avg_preds)
            all_labels.append(labels)
    
    # Calculate accuracy from averaged predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels).to(device)
    _, predicted = all_preds.max(1)
    accuracy = (predicted == all_labels).float().mean() * 100
    
    return accuracy.item()

# ============================
# 9. INITIALIZE AND TRAIN MODEL
# ============================

# Initialize model with optimized architecture
model = MusicCNN(
    input_channels=60,  # 3x20 features
    num_classes=len(genres),
    dropout_conv=0.2,   # Optimized dropout
    dropout_fc=0.4
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimized loss and optimizer
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  # Label smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # AdamW optimizer

# Cosine annealing with warm restarts for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,     # Restart every 10 epochs
    T_mult=2,   # Double the period after each restart
    eta_min=1e-6
)

# Training parameters optimized for 80% accuracy
num_epochs = 100  # More epochs with cosine annealing
patience = 20     # Increased patience
best_val_acc = 0
best_tta_acc = 0
patience_counter = 0
accumulation_steps = 2  # Gradient accumulation

train_losses = []
train_accs = []
val_losses = []
val_accs = []
tta_accs = []

print("\n" + "="*50)
print("Starting Optimized Training for 80% Target")
print("="*50)
print("Optimizations enabled:")
print("  ✓ 20 MFCCs + Delta + Delta-Delta (60 features)")
print("  ✓ Mixup augmentation")
print("  ✓ Label smoothing")
print("  ✓ AdamW optimizer with cosine annealing")
print("  ✓ Gradient accumulation")
print("  ✓ Test-Time Augmentation")
print("="*50 + "\n")

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Test-Time Augmentation validation (every 5 epochs after epoch 10)
        tta_acc = 0
        if epoch >= 10 and epoch % 5 == 0:
            tta_acc = validate_with_tta(model, val_loader, device, n_aug=3)
            tta_accs.append(tta_acc)
            print(f"  TTA Val Acc: {tta_acc:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Track best model (use TTA accuracy if available)
        current_best = tta_acc if tta_acc > 0 else val_acc
        
        # Save best model
        if val_acc > best_val_acc or (tta_acc > best_tta_acc and tta_acc > 0):
            best_val_acc = max(val_acc, best_val_acc)
            best_tta_acc = max(tta_acc, best_tta_acc)
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_tta_acc': best_tta_acc,
                'label_encoder_classes': label_encoder.classes_.tolist(),
                'config': {
                    'n_mfcc': 20,
                    'duration': 10,
                    'features': 60,
                    'batch_size': batch_size,
                    'accumulation_steps': accumulation_steps
                }
            }, 'music_genre_classifier_optimized.pt')
            print(f"  ✓ Saved best model (Val: {val_acc:.2f}%, TTA: {best_tta_acc:.2f}%)")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*50}")
print(f"Training Completed!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
if best_tta_acc > 0:
    print(f"Best TTA Accuracy: {best_tta_acc:.2f}%")
print(f"{'='*50}")

# ============================
# 10. PLOT RESULTS
# ============================

if len(train_losses) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History - Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Acc', linewidth=2)
    if len(tta_accs) > 0:
        # Plot TTA points
        tta_epochs = [i for i in range(10, len(train_accs), 5)][:len(tta_accs)]
        axes[0, 1].scatter(tta_epochs, tta_accs, color='red', s=50, zorder=5, label='TTA Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training History - Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule visualization
    lrs = []
    for epoch in range(min(100, len(train_losses))):
        lrs.append(0.0005 * (1 + np.cos(np.pi * (epoch % 10) / 10)) / 2)
    axes[1, 0].plot(lrs[:len(train_losses)], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Cosine Annealing Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    overfitting_gap = [t - v for t, v in zip(train_accs, val_accs)]
    axes[1, 1].plot(overfitting_gap, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train-Val Gap (%)')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Analysis - Optimized for 80% Accuracy', fontsize=14)
    plt.tight_layout()
    plt.show()

# ============================
# 11. FINAL EVALUATION WITH TTA
# ============================

if os.path.exists('music_genre_classifier_optimized.pt'):
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load('music_genre_classifier_optimized.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Standard evaluation
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Standard Validation Accuracy: {val_acc:.2f}%")
    
    # TTA evaluation with more augmentations
    print("\nRunning Test-Time Augmentation (5 augmentations)...")
    tta_acc = validate_with_tta(model, val_loader, device, n_aug=5)
    print(f"TTA Validation Accuracy: {tta_acc:.2f}%")
    
    # Detailed evaluation
    print("\nDetailed Performance Analysis:")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Optimized Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    class_correct = np.diag(cm)
    class_total = cm.sum(axis=1)
    for i, genre in enumerate(label_encoder.classes_):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {genre:12s}: {acc:.1f}%")
    
    # Model size
    model_size = os.path.getsize('music_genre_classifier_optimized.pt') / (1024 * 1024)
    print(f"\nModel size: {model_size:.2f} MB")
    
    # Try to download model file
    try:
        from google.colab import files
        files.download('music_genre_classifier_optimized.pt')
        print("✓ Model downloaded successfully!")
    except:
        print("Model saved as 'music_genre_classifier_optimized.pt' in Colab")

# ============================
# 12. INFERENCE FUNCTION
# ============================

def predict_genre(audio_path, model, label_encoder, device, use_tta=True):
    """
    Predict genre for a single audio file with optional TTA
    """
    model.eval()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=10)
    
    # Ensure consistent length
    target_length = 22050 * 10
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            # Test-Time Augmentation
            predictions = []
            
            # Original
            output = model(features_tensor)
            predictions.append(F.softmax(output, dim=1))
            
            # Time-shifted versions
            for shift in [-20, -10, 10, 20]:
                shifted = torch.roll(features_tensor, shifts=shift, dims=-1)
                output = model(shifted)
                predictions.append(F.softmax(output, dim=1))
            
            # Average predictions
            avg_pred = torch.stack(predictions).mean(dim=0)
            confidence, predicted = torch.max(avg_pred, 1)
        else:
            output = model(features_tensor)
            confidence, predicted = torch.max(F.softmax(output, dim=1), 1)
    
    genre = label_encoder.classes_[predicted.item()]
    confidence = confidence.item() * 100
    
    return genre, confidence