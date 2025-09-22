import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# Try to import librosa, fall back to torchaudio if not available
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    import torchaudio
    import torchaudio.transforms as T
    print("Warning: librosa not found, using torchaudio instead")

# ============================
# FLEXIBLE MODEL DEFINITION
# ============================

class MusicCNN(nn.Module):
    def __init__(self, input_channels=60, num_classes=10, dropout_conv=0.2, dropout_fc=0.4):
        """
        Flexible CNN that can handle both the original and optimized versions
        """
        super(MusicCNN, self).__init__()
        
        self.input_channels = input_channels
        
        # Determine architecture based on input channels
        if input_channels == 13:
            # Original simple architecture (13 MFCCs only)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Dropout2d(0.3)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        else:
            # Optimized architecture (60 channels: 20 MFCCs + deltas)
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_conv),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_conv),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Dropout2d(dropout_conv + 0.1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_fc),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_fc),
                nn.Linear(256, num_classes)
            )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================
# FLEXIBLE AUDIO PROCESSOR
# ============================

class AudioProcessor:
    def __init__(self, sr=22050, duration=30, n_mfcc=13, n_fft=2048, hop_length=512, use_deltas=False):
        """
        Flexible audio processor that can work with both model versions
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_deltas = use_deltas
        
    def extract_features_librosa(self, y, sr):
        """Extract features using librosa (preferred method)"""
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        )
        
        if self.use_deltas:
            # Add delta and delta-delta features for enhanced model
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        else:
            features = mfcc
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def extract_features_torchaudio(self, waveform, sr):
        """Extract features using torchaudio (fallback method)"""
        # Convert numpy to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.FloatTensor(waveform)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract MFCCs
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': 128
            }
        )
        
        mfcc = mfcc_transform(waveform)
        
        if self.use_deltas:
            # Simple approximation of deltas for torchaudio
            # Not as good as librosa but workable
            mfcc_np = mfcc.numpy()
            mfcc_delta = np.zeros_like(mfcc_np)
            mfcc_delta[:, :, 1:] = mfcc_np[:, :, 1:] - mfcc_np[:, :, :-1]
            mfcc_delta2 = np.zeros_like(mfcc_np)
            mfcc_delta2[:, :, 1:] = mfcc_delta[:, :, 1:] - mfcc_delta[:, :, :-1]
            features = np.vstack([mfcc_np.squeeze(0), mfcc_delta.squeeze(0), mfcc_delta2.squeeze(0)])
        else:
            features = mfcc.squeeze(0).numpy()
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
        
    def process_audio(self, audio_path):
        try:
            if HAS_LIBROSA:
                # Load with librosa
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
                
                # Ensure consistent length
                target_length = self.sr * self.duration
                if len(y) > target_length:
                    start = (len(y) - target_length) // 2
                    y = y[start:start + target_length]
                else:
                    y = np.pad(y, (0, target_length - len(y)), mode='constant')
                
                features = self.extract_features_librosa(y, self.sr)
                return features, y
            else:
                # Load with torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != self.sr:
                    resampler = T.Resample(sample_rate, self.sr)
                    waveform = resampler(waveform)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Adjust length
                num_samples = self.sr * self.duration
                if waveform.shape[1] > num_samples:
                    waveform = waveform[:, :num_samples]
                else:
                    padding = num_samples - waveform.shape[1]
                    waveform = F.pad(waveform, (0, padding))
                
                features = self.extract_features_torchaudio(waveform, self.sr)
                return features, waveform.numpy().flatten()
                
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

# ============================
# AUTO-DETECTING MUSIC CLASSIFIER
# ============================

class MusicGenreClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get label encoder classes
        self.genre_labels = checkpoint['label_encoder_classes']
        
        # Auto-detect model architecture from state dict
        state_dict = checkpoint['model_state_dict']
        
        # Check first conv layer to determine input channels
        first_conv_key = 'features.0.weight'
        if first_conv_key in state_dict:
            # Shape is [out_channels, in_channels, kernel_h, kernel_w]
            # But we have a channel dimension, so we need to check the actual feature size
            first_conv_weight = state_dict[first_conv_key]
            print(f"First conv layer shape: {first_conv_weight.shape}")
        
        # Check classifier to determine architecture version
        classifier_key = 'classifier.0.weight'
        if classifier_key in state_dict:
            classifier_weight = state_dict[classifier_key]
            classifier_input = classifier_weight.shape[1]
            
            if classifier_input == 128 * 4 * 4:  # 2048
                # Original simple model
                print("Detected: Simple model architecture (13 MFCCs)")
                input_channels = 13
                model_type = 'simple'
            elif classifier_input == 256 * 4 * 4:  # 4096
                # Optimized model
                print("Detected: Optimized model architecture (60 features)")
                input_channels = 60
                model_type = 'optimized'
            else:
                print(f"Warning: Unknown classifier input size {classifier_input}")
                # Try to infer from config
                config = checkpoint.get('config', {})
                input_channels = config.get('features', config.get('n_mfcc', 13))
                model_type = 'optimized' if input_channels > 20 else 'simple'
        else:
            # Fallback to config
            config = checkpoint.get('config', {})
            input_channels = config.get('features', config.get('n_mfcc', 13))
            model_type = 'optimized' if input_channels > 20 else 'simple'
        
        # Initialize model with detected architecture
        self.model = MusicCNN(
            input_channels=input_channels,
            num_classes=len(self.genre_labels),
            dropout_conv=0.2 if model_type == 'optimized' else 0.25,
            dropout_fc=0.4 if model_type == 'optimized' else 0.5
        )
        
        # Load model weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Get configuration
        config = checkpoint.get('config', {})
        
        # Initialize processor based on detected model type
        if model_type == 'optimized':
            # Enhanced model with deltas
            self.processor = AudioProcessor(
                sr=22050,
                duration=config.get('duration', 10),
                n_mfcc=config.get('n_mfcc', 20),
                n_fft=2048,
                hop_length=512,
                use_deltas=True
            )
        else:
            # Simple model
            self.processor = AudioProcessor(
                sr=22050,
                duration=30,  # Original uses 30 seconds
                n_mfcc=13,
                n_fft=2048,
                hop_length=512,
                use_deltas=False
            )
        
        print("✓ Model loaded successfully!")
        print(f"  Model type: {model_type}")
        print(f"  Genres: {self.genre_labels}")
        print(f"  Input features: {input_channels} channels")
        if 'best_val_acc' in checkpoint:
            print(f"  Model accuracy: {checkpoint['best_val_acc']:.2f}%")
        if 'best_tta_acc' in checkpoint and checkpoint['best_tta_acc'] > 0:
            print(f"  TTA accuracy: {checkpoint['best_tta_acc']:.2f}%")

    def predict(self, audio_path, use_tta=False, analyze_full_song=False, hop_ratio=0.5):
        """
        Predict genre with optional Test-Time Augmentation and full song analysis
        
        Args:
            audio_path: Path to audio file
            use_tta: Whether to use Test-Time Augmentation
            analyze_full_song: Whether to analyze the entire song or just one segment
            hop_ratio: Overlap ratio for sliding window (0.5 = 50% overlap)
        """
        if analyze_full_song:
            return self._predict_full_song(audio_path, use_tta, hop_ratio)
        else:
            # Original single-segment prediction
            features, waveform = self.processor.process_audio(audio_path)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if use_tta:
                    # Test-Time Augmentation for better accuracy
                    predictions = []
                    
                    # Original prediction
                    outputs = self.model(features_tensor)
                    predictions.append(F.softmax(outputs, dim=1))
                    
                    # Time-shifted versions
                    for shift in [-20, -10, 10, 20]:
                        shifted = torch.roll(features_tensor, shifts=shift, dims=-1)
                        outputs = self.model(shifted)
                        predictions.append(F.softmax(outputs, dim=1))
                    
                    # Average predictions
                    probabilities = torch.stack(predictions).mean(dim=0)
                else:
                    outputs = self.model(features_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                
                confidence, predicted = torch.max(probabilities, 1)

            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, min(3, len(self.genre_labels)))
            top3_results = []
            for i in range(min(3, len(self.genre_labels))):
                genre = self.genre_labels[top3_indices[0][i]]
                prob = top3_probs[0][i].item() * 100
                top3_results.append((genre, prob))

            return {
                'predicted_genre': self.genre_labels[predicted.item()],
                'confidence': confidence.item() * 100,
                'top3_predictions': top3_results,
                'features': features,
                'waveform': waveform,
                'segments_analyzed': 1
            }
    
    def _predict_full_song(self, audio_path, use_tta, hop_ratio=0.5):
        """
        Analyze the entire song using sliding window approach
        """
        print("Analyzing full song with sliding window...")
        
        # Load the entire audio file
        if HAS_LIBROSA:
            y, sr = librosa.load(audio_path, sr=self.processor.sr, duration=None)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.processor.sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.processor.sr)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            y = waveform.numpy().flatten()
            sr = self.processor.sr
        
        # Calculate sliding window parameters
        segment_length = self.processor.sr * self.processor.duration
        hop_length = int(segment_length * (1 - hop_ratio))
        total_length = len(y)
        
        # Extract segments
        segments = []
        positions = []
        for start in range(0, total_length - segment_length + 1, hop_length):
            end = start + segment_length
            segment = y[start:end]
            segments.append(segment)
            positions.append(start / self.processor.sr)  # Position in seconds
            
            # Limit number of segments to avoid memory issues
            if len(segments) >= 20:  # Max 20 segments
                print(f"  Limiting to 20 segments (song too long)")
                break
        
        if len(segments) == 0:
            # Song is shorter than required duration, pad it
            segment = np.pad(y, (0, segment_length - len(y)), mode='constant')
            segments = [segment]
            positions = [0]
        
        print(f"  Analyzing {len(segments)} segments from the song")
        
        # Analyze each segment
        all_predictions = []
        segment_results = []
        
        with torch.no_grad():
            for i, segment in enumerate(segments):
                # Extract features for this segment
                if HAS_LIBROSA:
                    features = self.processor.extract_features_librosa(segment, sr)
                else:
                    features = self.processor.extract_features_torchaudio(
                        torch.FloatTensor(segment).unsqueeze(0), sr
                    )
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # Get prediction for this segment
                outputs = self.model(features_tensor)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append(probs)
                
                # Store per-segment results
                conf, pred = torch.max(probs, 1)
                segment_results.append({
                    'position': positions[i],
                    'genre': self.genre_labels[pred.item()],
                    'confidence': conf.item() * 100
                })
        
        # Average predictions across all segments
        avg_probabilities = torch.stack(all_predictions).mean(dim=0)
        
        # Apply TTA on the averaged predictions if requested
        if use_tta and len(segments) > 1:
            # Additional augmentation by weighting different segments
            weighted_preds = []
            for i, pred in enumerate(all_predictions):
                # Weight center segments slightly higher
                weight = 1.0 if i == len(all_predictions) // 2 else 0.9
                weighted_preds.append(pred * weight)
            avg_probabilities = torch.stack(weighted_preds).sum(dim=0)
            avg_probabilities = avg_probabilities / avg_probabilities.sum(dim=1, keepdim=True)
        
        confidence, predicted = torch.max(avg_probabilities, 1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(avg_probabilities, min(3, len(self.genre_labels)))
        top3_results = []
        for i in range(min(3, len(self.genre_labels))):
            genre = self.genre_labels[top3_indices[0][i]]
            prob = top3_probs[0][i].item() * 100
            top3_results.append((genre, prob))
        
        # Find most common genre across segments
        genre_counts = {}
        for result in segment_results:
            genre = result['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        most_common_genre = max(genre_counts, key=genre_counts.get)
        
        # Use a representative segment for visualization (middle segment)
        middle_idx = len(segments) // 2
        middle_segment = segments[middle_idx]
        
        if HAS_LIBROSA:
            display_features = self.processor.extract_features_librosa(middle_segment, sr)
        else:
            display_features = self.processor.extract_features_torchaudio(
                torch.FloatTensor(middle_segment).unsqueeze(0), sr
            )
        
        return {
            'predicted_genre': self.genre_labels[predicted.item()],
            'confidence': confidence.item() * 100,
            'top3_predictions': top3_results,
            'features': display_features,
            'waveform': middle_segment,
            'segments_analyzed': len(segments),
            'segment_results': segment_results,
            'most_common_genre': most_common_genre,
            'genre_distribution': genre_counts
        }

# ============================
# GUI APPLICATION
# ============================

class MusicGenreClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Music Genre Classifier (Universal)")
        self.root.geometry("900x800")
        
        self.classifier = None
        self.current_audio_path = None
        self.use_tta = tk.BooleanVar(value=False)
        self.analyze_full = tk.BooleanVar(value=True)  # Default to analyzing full song
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        tk.Label(title_frame, text="🎵 Music Genre Classifier 🎵", 
                font=("Arial", 20, "bold")).pack()
        tk.Label(title_frame, text="Auto-detects model architecture", 
                font=("Arial", 10, "italic")).pack()
        
        # Model loading
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Button(model_frame, text="Load Model (.pt file)", command=self.load_model, 
                 bg="#4CAF50", fg="white", font=("Arial", 12), 
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        self.model_status = tk.Label(model_frame, text="No model loaded", 
                                     font=("Arial", 10), fg="red")
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        # Audio selection
        audio_frame = tk.Frame(self.root)
        audio_frame.pack(pady=10)
        tk.Button(audio_frame, text="Select Audio File", command=self.select_audio, 
                 bg="#2196F3", fg="white", font=("Arial", 12), 
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        self.audio_label = tk.Label(audio_frame, text="No file selected", 
                                    font=("Arial", 10))
        self.audio_label.pack(side=tk.LEFT, padx=10)
        
        # TTA option
        tta_frame = tk.Frame(self.root)
        tta_frame.pack(pady=5)
        tk.Checkbutton(tta_frame, text="Use Test-Time Augmentation (more accurate but slower)", 
                      variable=self.use_tta, font=("Arial", 10)).pack()
        
        # Full song analysis option
        tk.Checkbutton(tta_frame, text="Analyze full song (multiple segments)", 
                      variable=self.analyze_full, font=("Arial", 10), fg="blue").pack()
        
        # Classify button
        self.classify_btn = tk.Button(self.root, text="🎼 Classify Genre", 
                                      command=self.classify_audio, 
                                      bg="#FF9800", fg="white", 
                                      font=("Arial", 14, "bold"), 
                                      padx=30, pady=10, state=tk.DISABLED)
        self.classify_btn.pack(pady=20)
        
        # Results
        results_frame = tk.LabelFrame(self.root, text="Classification Results", 
                                     font=("Arial", 12, "bold"))
        results_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.result_label = tk.Label(results_frame, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)
        self.confidence_label = tk.Label(results_frame, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        self.top3_frame = tk.Frame(results_frame)
        self.top3_frame.pack(pady=10)
        self.viz_frame = tk.Frame(results_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select model file", 
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.classifier = MusicGenreClassifier(file_path)
                self.model_status.config(text="✓ Model loaded", fg="green")
                self.update_classify_button()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.model_status.config(text="Failed to load model", fg="red")
    
    def select_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file", 
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.au"), 
                      ("All files", "*.*")]
        )
        if file_path:
            self.current_audio_path = file_path
            filename = os.path.basename(file_path)
            self.audio_label.config(text=f"Selected: {filename}")
            self.update_classify_button()
    
    def update_classify_button(self):
        if self.classifier and self.current_audio_path:
            self.classify_btn.config(state=tk.NORMAL)
        else:
            self.classify_btn.config(state=tk.DISABLED)
    
    def classify_audio(self):
        if not self.classifier or not self.current_audio_path:
            return
        
        self.progress.pack(pady=5)
        self.progress.start()
        self.result_label.config(text="Classifying...")
        self.confidence_label.config(text="")
        
        # Clear previous results
        for widget in self.top3_frame.winfo_children():
            widget.destroy()
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Run classification in thread
        thread = threading.Thread(target=self.run_classification)
        thread.start()
    
    def run_classification(self):
        try:
            results = self.classifier.predict(
                self.current_audio_path, 
                use_tta=self.use_tta.get(),
                analyze_full_song=self.analyze_full.get()
            )
            self.root.after(0, self.display_results, results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Classification failed:\n{str(e)}"
            ))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
    
    def display_results(self, results):
        genre = results['predicted_genre']
        confidence = results['confidence']
        
        # Color based on confidence
        color = "#4CAF50" if confidence >= 80 else ("#FF9800" if confidence >= 60 else "#F44336")
        
        self.result_label.config(text=f"🎵 Predicted Genre: {genre.upper()}", fg=color)
        
        # Show number of segments analyzed if full song analysis was used
        if results.get('segments_analyzed', 1) > 1:
            conf_text = f"Confidence: {confidence:.1f}% (analyzed {results['segments_analyzed']} segments)"
            self.confidence_label.config(text=conf_text)
            
            # Show genre distribution if available
            if 'genre_distribution' in results:
                dist_text = " | Distribution: "
                for g, count in sorted(results['genre_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                    dist_text += f"{g}:{count} "
                self.confidence_label.config(text=conf_text + dist_text)
        else:
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}% (single segment)")
        
        # Top 3 predictions
        tk.Label(self.top3_frame, text="Top 3 Predictions:", 
                font=("Arial", 11, "bold")).pack()
        for i, (genre, prob) in enumerate(results['top3_predictions']):
            color = "#4CAF50" if i == 0 else "#757575"
            tk.Label(self.top3_frame, text=f"{i+1}. {genre}: {prob:.1f}%", 
                    font=("Arial", 10), fg=color).pack()
        
        # Show per-segment results if available
        if 'segment_results' in results and len(results['segment_results']) > 1:
            tk.Label(self.top3_frame, text="\nSegment Analysis:", 
                    font=("Arial", 10, "bold")).pack()
            # Show first few segment results
            for i, seg in enumerate(results['segment_results'][:5]):
                tk.Label(self.top3_frame, 
                        text=f"  @{seg['position']:.1f}s: {seg['genre']} ({seg['confidence']:.0f}%)",
                        font=("Arial", 9), fg="#666666").pack()
            if len(results['segment_results']) > 5:
                tk.Label(self.top3_frame, 
                        text=f"  ... and {len(results['segment_results'])-5} more segments",
                        font=("Arial", 9, "italic"), fg="#999999").pack()
        
        # Visualization
        fig = plt.Figure(figsize=(10, 4), tight_layout=True)
        
        # MFCC features plot
        ax1 = fig.add_subplot(121)
        # Show only first n_mfcc coefficients (not deltas if present)
        features_to_show = results['features']
        if features_to_show.shape[0] > 20:
            features_to_show = features_to_show[:20]  # Show first 20 MFCCs
        elif features_to_show.shape[0] > 13:
            features_to_show = features_to_show[:13]  # Show first 13 MFCCs
        
        im = ax1.imshow(features_to_show, aspect='auto', cmap='coolwarm')
        ax1.set_title(f'MFCC Features ({features_to_show.shape[0]} coefficients)')
        ax1.set_xlabel('Time Frame')
        ax1.set_ylabel('MFCC Coefficient')
        fig.colorbar(im, ax=ax1)
        
        # Waveform plot
        ax2 = fig.add_subplot(122)
        duration = self.classifier.processor.duration
        time = np.linspace(0, duration, len(results['waveform']))
        ax2.plot(time, results['waveform'], linewidth=0.5, alpha=0.7)
        ax2.set_title(f'Audio Waveform ({duration} seconds)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        self.root.mainloop()

# ============================
# STANDALONE PREDICTION FUNCTION
# ============================

def predict_genre_standalone(audio_path, model_path, use_tta=False, full_song=False):
    """
    Standalone function to predict genre without GUI
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model (.pt file)
        use_tta: Whether to use Test-Time Augmentation
        full_song: Whether to analyze the entire song
    
    Returns:
        Dictionary with prediction results
    """
    classifier = MusicGenreClassifier(model_path)
    results = classifier.predict(audio_path, use_tta=use_tta, analyze_full_song=full_song)
    
    print(f"\n{'='*50}")
    print(f"Audio: {os.path.basename(audio_path)}")
    if results.get('segments_analyzed', 1) > 1:
        print(f"Segments analyzed: {results['segments_analyzed']}")
    print(f"Predicted Genre: {results['predicted_genre']}")
    print(f"Confidence: {results['confidence']:.1f}%")
    
    if 'genre_distribution' in results:
        print(f"\nGenre distribution across segments:")
        for genre, count in sorted(results['genre_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {count} segments")
    
    print(f"\nTop 3 Predictions (averaged):")
    for i, (genre, prob) in enumerate(results['top3_predictions']):
        print(f"  {i+1}. {genre}: {prob:.1f}%")
    print(f"{'='*50}\n")
    
    return results

# ============================
# COMPATIBILITY TEST FUNCTION
# ============================

def test_model_compatibility(model_path):
    """
    Test function to check model architecture and compatibility
    """
    print(f"\n{'='*50}")
    print("Model Compatibility Test")
    print(f"{'='*50}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("\n1. Checkpoint keys:")
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            print(f"  - {key}: {checkpoint[key] if not isinstance(checkpoint[key], dict) else 'dict'}")
    
    print("\n2. Model architecture analysis:")
    state_dict = checkpoint['model_state_dict']
    
    # Analyze first conv layer
    if 'features.0.weight' in state_dict:
        weight = state_dict['features.0.weight']
        print(f"  First conv layer: {weight.shape}")
        print(f"    Output channels: {weight.shape[0]}")
        
    # Analyze classifier
    if 'classifier.0.weight' in state_dict:
        weight = state_dict['classifier.0.weight']
        print(f"  First classifier layer: {weight.shape}")
        print(f"    Input features: {weight.shape[1]}")
        print(f"    Output features: {weight.shape[0]}")
        
        # Determine architecture type
        if weight.shape[1] == 2048:  # 128 * 4 * 4
            print("  → Detected: Simple model (13 MFCCs)")
        elif weight.shape[1] == 4096:  # 256 * 4 * 4
            print("  → Detected: Optimized model (60 features)")
    
    print("\n3. Labels:")
    if 'label_encoder_classes' in checkpoint:
        labels = checkpoint['label_encoder_classes']
        print(f"  Number of classes: {len(labels)}")
        print(f"  Classes: {labels}")
    
    print(f"\n{'='*50}\n")

# ============================
# MAIN
# ============================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test' and len(sys.argv) > 2:
            # Test mode: check model compatibility
            test_model_compatibility(sys.argv[2])
        elif len(sys.argv) > 2:
            # Command line usage
            model_path = sys.argv[1]
            audio_path = sys.argv[2]
            use_tta = '--tta' in sys.argv
            full_song = '--full' in sys.argv
            
            print("Running in command line mode...")
            predict_genre_standalone(audio_path, model_path, use_tta, full_song)
        else:
            print("Usage:")
            print("  GUI mode: python script.py")
            print("  CLI mode: python script.py model.pt audio.wav [--tta] [--full]")
            print("  Test mode: python script.py --test model.pt")
            print("\nOptions:")
            print("  --tta   Use Test-Time Augmentation")
            print("  --full  Analyze entire song (not just one segment)")
    else:
        # GUI mode
        print("Starting GUI application...")
        print("This universal inference script auto-detects model architecture")
        print("Works with both simple (13 MFCC) and optimized (60 feature) models")
        app = MusicGenreClassifierGUI()
        app.run()