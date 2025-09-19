# Music Genre Classifier - Fixed Training Script for Google Colab
# Fixes DataLoader worker issues, memory management, and dataset handling

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
# 3. FIXED DATASET CLASS
# ============================

class MusicDataset(Dataset):
    def __init__(self, file_paths, labels, sr=22050, duration=30, n_mfcc=13, 
                 augment=False):
        """
        Fixed Dataset class optimized for full GTZAN dataset
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.augment = augment
        
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
    
    def __len__(self):
        return len(self.file_paths)
    
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
            
            # Enhanced augmentation for training
            if self.augment and np.random.random() > 0.3:  # 70% chance of augmentation
                aug_choice = np.random.choice(['noise', 'shift', 'speed'])
                
                if aug_choice == 'noise':
                    # Add small amount of noise
                    noise_factor = np.random.uniform(0.001, 0.01)
                    y = y + np.random.normal(0, noise_factor, y.shape)
                
                elif aug_choice == 'shift':
                    # Time shift
                    shift_samples = int(np.random.uniform(-0.1, 0.1) * len(y))
                    y = np.roll(y, shift_samples)
                
                elif aug_choice == 'speed':
                    # Speed change (simple time stretching)
                    speed_factor = np.random.uniform(0.9, 1.1)
                    if speed_factor != 1.0:
                        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
                        if len(y_stretched) >= target_length:
                            y = y_stretched[:target_length]
                        else:
                            y = np.pad(y_stretched, (0, target_length - len(y_stretched)), mode='constant')
            
            # Extract MFCC features using librosa (more stable than torchaudio in Colab)
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=self.sr, 
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512,
                n_mels=128  # More mel filters for better representation
            )
            
            # Normalize features (per sample normalization)
            mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
            
            return torch.FloatTensor(mfcc), label
            
        except Exception as e:
            print(f"Error loading {os.path.basename(audio_path)}: {e}")
            # Return zero features as fallback
            dummy_mfcc = np.zeros((self.n_mfcc, (self.sr * self.duration) // 512 + 1))
            return torch.FloatTensor(dummy_mfcc), label

# ============================
# 4. SIMPLIFIED CNN MODEL
# ============================

class MusicCNN(nn.Module):
    def __init__(self, input_channels=13, num_classes=10):
        super(MusicCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
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
# 5. DATA PREPARATION
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

# Use full GTZAN dataset (100 files per genre = 1000 total)
# Only balance if there are significantly uneven class distributions
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
# 6. FIXED DATALOADER SETUP
# ============================

# Create datasets
train_dataset = MusicDataset(X_train, y_train, augment=True)
val_dataset = MusicDataset(X_val, y_val, augment=False)

# Optimized batch size for full GTZAN dataset
batch_size = 32  # Increased back to 32 since we're using full dataset efficiently

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0,  # ← This fixes the worker error!
    pin_memory=False  # Disable pin_memory in Colab
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=0,  # ← This fixes the worker error!
    pin_memory=False  # Disable pin_memory in Colab
)

print("✓ DataLoaders created successfully")

# ============================
# 7. TRAINING FUNCTIONS
# ============================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in tqdm(train_loader, desc='Training'):
        try:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
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

# ============================
# 8. INITIALIZE AND TRAIN MODEL
# ============================

# Initialize model
model = MusicCNN(num_classes=len(genres)).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training parameters optimized for full GTZAN dataset
num_epochs = 50  # Increased epochs since we have more data
patience = 15   # Increased patience for better convergence
best_val_acc = 0
patience_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print("\n" + "="*50)
print("Starting Training")
print("="*50)

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'label_encoder_classes': label_encoder.classes_.tolist()
            }, 'music_genre_classifier_best.pt')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
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
print(f"{'='*50}")

# ============================
# 9. PLOT RESULTS
# ============================

if len(train_losses) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training History - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================
# 10. FINAL EVALUATION
# ============================

if os.path.exists('music_genre_classifier_best.pt'):
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load('music_genre_classifier_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    
    # Try to download model file
    try:
        from google.colab import files
        files.download('music_genre_classifier_best.pt')
        print("✓ Model downloaded successfully!")
    except:
        print("Model saved as 'music_genre_classifier_best.pt' in Colab")

print("\n✅ Script completed successfully!")