"""
Music Genre Classifier - Inference/Demo Script
Run this in VS Code after training the model in Google Colab
"""

import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

# ============================
# MODEL DEFINITION
# ============================

class MusicGenreCNN(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(MusicGenreCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_output = self._forward_conv(test_input)
        flattened_size = test_output.shape[1] * test_output.shape[2] * test_output.shape[3]
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _forward_conv(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        return x
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================
# AUDIO PROCESSOR
# ============================

class AudioProcessor:
    def __init__(self, model_config):
        self.sr = model_config.get('sr', 22050)
        self.duration = model_config.get('duration', 30)
        self.n_mfcc = model_config.get('n_mfcc', 13)
        self.n_fft = model_config.get('n_fft', 2048)
        self.hop_length = model_config.get('hop_length', 512)
        
    def process_audio(self, audio_path):
        """Process audio file and extract features"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.sr:
                resampler = T.Resample(sample_rate, self.sr)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to fixed duration
            num_samples = self.sr * self.duration
            if waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
            elif waveform.shape[1] < num_samples:
                padding = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Extract MFCC features
            mfcc_transform = T.MFCC(
                sample_rate=self.sr,
                n_mfcc=self.n_mfcc,
                melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length}
            )
            mfcc = mfcc_transform(waveform)
            
            return mfcc.squeeze(0), waveform.numpy().flatten()
            
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

# ============================
# MUSIC CLASSIFIER
# ============================

class MusicGenreClassifier:
    def __init__(self, model_path):
        """Initialize the classifier with trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        self.input_shape = checkpoint['input_shape']
        self.genre_labels = checkpoint['label_encoder_classes']
        self.model_config = checkpoint.get('model_config', {})
        
        # Initialize model
        self.model = MusicGenreCNN(input_shape=self.input_shape, num_classes=len(self.genre_labels))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize audio processor
        self.processor = AudioProcessor(self.model_config)
        
        print(f"Model loaded successfully!")
        print(f"Genre labels: {self.genre_labels}")
    
    def predict(self, audio_path):
        """Predict genre for an audio file"""
        # Process audio
        features, waveform = self.processor.process_audio(audio_path)
        
        # Add batch dimension
        features_tensor = features.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        top3_results = []
        for i in range(3):
            genre = self.genre_labels[top3_indices[0][i]]
            prob = top3_probs[0][i].item() * 100
            top3_results.append((genre, prob))
        
        return {
            'predicted_genre': self.genre_labels[predicted.item()],
            'confidence': confidence.item() * 100,
            'top3_predictions': top3_results,
            'features': features.numpy(),
            'waveform': waveform
        }

# ============================
# GUI APPLICATION
# ============================

class MusicGenreClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Music Genre Classifier")
        self.root.geometry("900x700")
        
        # Initialize classifier (will be loaded later)
        self.classifier = None
        self.current_audio_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="🎵 Music Genre Classifier 🎵", 
                               font=("Arial", 20, "bold"))
        title_label.pack(pady=10)
        
        # Model loading frame
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        
        tk.Button(model_frame, text="Load Model (.pt file)", 
                 command=self.load_model, bg="#4CAF50", fg="white",
                 font=("Arial", 12), padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        self.model_status = tk.Label(model_frame, text="No model loaded", 
                                     font=("Arial", 10), fg="red")
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        # Audio selection frame
        audio_frame = tk.Frame(self.root)
        audio_frame.pack(pady=10)
        
        tk.Button(audio_frame, text="Select Audio File", 
                 command=self.select_audio, bg="#2196F3", fg="white",
                 font=("Arial", 12), padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        self.audio_label = tk.Label(audio_frame, text="No file selected", 
                                    font=("Arial", 10))
        self.audio_label.pack(side=tk.LEFT, padx=10)
        
        # Classify button
        self.classify_btn = tk.Button(self.root, text="🎼 Classify Genre", 
                                      command=self.classify_audio,
                                      bg="#FF9800", fg="white", 
                                      font=("Arial", 14, "bold"),
                                      padx=30, pady=10, state=tk.DISABLED)
        self.classify_btn.pack(pady=20)
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="Classification Results", 
                                     font=("Arial", 12, "bold"))
        results_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Main prediction
        self.result_label = tk.Label(results_frame, text="", 
                                    font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = tk.Label(results_frame, text="", 
                                        font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Top 3 predictions
        self.top3_frame = tk.Frame(results_frame)
        self.top3_frame.pack(pady=10)
        
        # Visualization frame
        self.viz_frame = tk.Frame(results_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        
    def load_model(self):
        """Load the trained model"""
        file_path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.classifier = MusicGenreClassifier(file_path)
                self.model_status.config(text="✓ Model loaded", fg="green")
                self.update_classify_button()
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model_status.config(text="Failed to load model", fg="red")
    
    def select_audio(self):
        """Select audio file for classification"""
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_audio_path = file_path
            filename = os.path.basename(file_path)
            self.audio_label.config(text=f"Selected: {filename}")
            self.update_classify_button()
    
    def update_classify_button(self):
        """Enable/disable classify button based on conditions"""
        if self.classifier and self.current_audio_path:
            self.classify_btn.config(state=tk.NORMAL)
        else:
            self.classify_btn.config(state=tk.DISABLED)
    
    def classify_audio(self):
        """Classify the selected audio file"""
        if not self.classifier or not self.current_audio_path:
            return
        
        # Show progress
        self.progress.pack(pady=5)
        self.progress.start()
        
        # Clear previous results
        self.result_label.config(text="Classifying...")
        self.confidence_label.config(text="")
        for widget in self.top3_frame.winfo_children():
            widget.destroy()
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Run classification in separate thread
        thread = threading.Thread(target=self.run_classification)
        thread.start()
    
    def run_classification(self):
        """Run the actual classification"""
        try:
            # Perform classification
            results = self.classifier.predict(self.current_audio_path)
            
            # Update UI in main thread
            self.root.after(0, self.display_results, results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification failed: {str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
    
    def display_results(self, results):
        """Display classification results"""
        # Main result
        genre = results['predicted_genre']
        confidence = results['confidence']
        
        self.result_label.config(
            text=f"🎵 Predicted Genre: {genre.upper()}",
            fg="#4CAF50"
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1f}%"
        )
        
        # Top 3 predictions
        tk.Label(self.top3_frame, text="Top 3 Predictions:", 
                font=("Arial", 11, "bold")).pack()
        
        for i, (genre, prob) in enumerate(results['top3_predictions']):
            color = "#4CAF50" if i == 0 else "#757575"
            tk.Label(self.top3_frame, 
                    text=f"{i+1}. {genre}: {prob:.1f}%",
                    font=("Arial", 10),
                    fg=color).pack()
        
        # Create visualization
        self.create_visualization(results)
    
    def create_visualization(self, results):
        """Create and display feature visualization"""
        fig = plt.Figure(figsize=(10, 4), tight_layout=True)
        
        # MFCC heatmap
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(results['features'], aspect='auto', cmap='coolwarm')
        ax1.set_title('MFCC Features')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('MFCC Coefficients')
        fig.colorbar(im, ax=ax1)
        
        # Waveform
        ax2 = fig.add_subplot(122)
        time = np.linspace(0, len(results['waveform']) / self.classifier.processor.sr, 
                          len(results['waveform']))
        ax2.plot(time, results['waveform'], linewidth=0.5)
        ax2.set_title('Audio Waveform')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# ============================
# COMMAND LINE INTERFACE
# ============================

class MusicGenreClassifierCLI:
    def __init__(self, model_path):
        """Initialize CLI classifier"""
        self.classifier = MusicGenreClassifier(model_path)
    
    def classify_file(self, audio_path):
        """Classify a single audio file"""
        print(f"\nClassifying: {audio_path}")
        print("-" * 50)
        
        try:
            results = self.classifier.predict(audio_path)
            
            print(f"✓ Predicted Genre: {results['predicted_genre'].upper()}")
            print(f"✓ Confidence: {results['confidence']:.2f}%")
            print(f"\nTop 3 Predictions:")
            for i, (genre, prob) in enumerate(results['top3_predictions']):
                print(f"  {i+1}. {genre}: {prob:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None
    
    def classify_directory(self, directory_path):
        """Classify all audio files in a directory"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print("No audio files found in directory")
            return
        
        print(f"\nFound {len(audio_files)} audio files")
        results = []
        
        for audio_file in audio_files:
            result = self.classify_file(audio_file)
            if result:
                results.append({
                    'file': audio_file,
                    'genre': result['predicted_genre'],
                    'confidence': result['confidence']
                })
        
        # Summary
        print("\n" + "=" * 50)
        print("CLASSIFICATION SUMMARY")
        print("=" * 50)
        
        for r in results:
            filename = os.path.basename(r['file'])
            print(f"{filename}: {r['genre']} ({r['confidence']:.1f}%)")
        
        # Genre distribution
        from collections import Counter
        genre_counts = Counter([r['genre'] for r in results])
        
        print("\nGenre Distribution:")
        for genre, count in genre_counts.most_common():
            print(f"  {genre}: {count} files")

# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main function to run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Music Genre Classifier')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pt file)')
    parser.add_argument('--mode', type=str, choices=['gui', 'cli'], 
                       default='gui', help='Run mode: gui or cli')
    parser.add_argument('--audio', type=str, 
                       help='Path to audio file (for CLI mode)')
    parser.add_argument('--directory', type=str, 
                       help='Path to directory with audio files (for CLI mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        # Run GUI application
        app = MusicGenreClassifierGUI()
        app.run()
    else:
        # Run CLI application
        if not args.audio and not args.directory:
            print("Error: Please provide --audio or --directory for CLI mode")
            return
        
        cli = MusicGenreClassifierCLI(args.model)
        
        if args.audio:
            cli.classify_file(args.audio)
        elif args.directory:
            cli.classify_directory(args.directory)

if __name__ == "__main__":
    # Check if running with arguments
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("\n" + "="*60)
        print("MUSIC GENRE CLASSIFIER - DEMO APPLICATION")
        print("="*60)
        print("\nUsage Options:")
        print("\n1. GUI Mode (recommended for demo):")
        print("   python music_classifier_demo.py --model path/to/model.pt")
        print("\n2. CLI Mode (single file):")
        print("   python music_classifier_demo.py --model path/to/model.pt --mode cli --audio path/to/audio.wav")
        print("\n3. CLI Mode (directory):")
        print("   python music_classifier_demo.py --model path/to/model.pt --mode cli --directory path/to/music/")
        print("\n" + "="*60)
        
        # If no arguments, try to run GUI with file dialog for model
        response = input("\nWould you like to run the GUI application? (y/n): ")
        if response.lower() == 'y':
            app = MusicGenreClassifierGUI()
            app.run()