import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# ============================
# SIMPLIFIED MODEL DEFINITION (MATCHES TRAINING SCRIPT)
# ============================

class MusicCNN(nn.Module):
    def __init__(self, input_channels=13, num_classes=10):
        super(MusicCNN, self).__init__()
        
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
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================
# AUDIO PROCESSOR
# ============================

class AudioProcessor:
    def __init__(self, sr=22050, duration=30, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def process_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.sr:
                resampler = T.Resample(sample_rate, self.sr)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            num_samples = self.sr * self.duration
            if waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
            elif waveform.shape[1] < num_samples:
                padding = num_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))

            mfcc_transform = T.MFCC(
                sample_rate=self.sr,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': 128
                }
            )
            mfcc = mfcc_transform(waveform)
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            return mfcc.squeeze(0), waveform.numpy().flatten()
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

# ============================
# MUSIC CLASSIFIER
# ============================

class MusicGenreClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.genre_labels = checkpoint['label_encoder_classes']

        self.model = MusicCNN(num_classes=len(self.genre_labels))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.processor = AudioProcessor()
        print("✓ Model loaded successfully!")
        print(f"  Genres: {self.genre_labels}")
        if 'best_val_acc' in checkpoint:
            print(f"  Model accuracy: {checkpoint['best_val_acc']:.2f}%")

    def predict(self, audio_path):
        features, waveform = self.processor.process_audio(audio_path)
        features_tensor = features.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

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
            'features': features.cpu().numpy(),
            'waveform': waveform
        }

# ============================
# GUI APPLICATION
# ============================

class MusicGenreClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Music Genre Classifier")
        self.root.geometry("900x750")
        
        self.classifier = None
        self.current_audio_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        tk.Label(title_frame, text="🎵 Music Genre Classifier 🎵", font=("Arial", 20, "bold")).pack()
        
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Button(model_frame, text="Load Model (.pt file)", command=self.load_model, bg="#4CAF50", fg="white", font=("Arial", 12), padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        self.model_status = tk.Label(model_frame, text="No model loaded", font=("Arial", 10), fg="red")
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        audio_frame = tk.Frame(self.root)
        audio_frame.pack(pady=10)
        tk.Button(audio_frame, text="Select Audio File", command=self.select_audio, bg="#2196F3", fg="white", font=("Arial", 12), padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        self.audio_label = tk.Label(audio_frame, text="No file selected", font=("Arial", 10))
        self.audio_label.pack(side=tk.LEFT, padx=10)
        
        self.classify_btn = tk.Button(self.root, text="🎼 Classify Genre", command=self.classify_audio, bg="#FF9800", fg="white", font=("Arial", 14, "bold"), padx=30, pady=10, state=tk.DISABLED)
        self.classify_btn.pack(pady=20)
        
        results_frame = tk.LabelFrame(self.root, text="Classification Results", font=("Arial", 12, "bold"))
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
        file_path = filedialog.askopenfilename(title="Select model file", filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")])
        if file_path:
            try:
                self.classifier = MusicGenreClassifier(file_path)
                self.model_status.config(text="✓ Model loaded", fg="green")
                self.update_classify_button()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.model_status.config(text="Failed to load model", fg="red")
    
    def select_audio(self):
        file_path = filedialog.askopenfilename(title="Select audio file", filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")])
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
        for widget in self.top3_frame.winfo_children():
            widget.destroy()
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        thread = threading.Thread(target=self.run_classification)
        thread.start()
    
    def run_classification(self):
        try:
            results = self.classifier.predict(self.current_audio_path)
            self.root.after(0, self.display_results, results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification failed:\n{str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
    
    def display_results(self, results):
        genre = results['predicted_genre']
        confidence = results['confidence']
        color = "#4CAF50" if confidence >= 80 else ("#FF9800" if confidence >= 60 else "#F44336")
        self.result_label.config(text=f"🎵 Predicted Genre: {genre.upper()}", fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        tk.Label(self.top3_frame, text="Top 3 Predictions:", font=("Arial", 11, "bold")).pack()
        for i, (genre, prob) in enumerate(results['top3_predictions']):
            color = "#4CAF50" if i == 0 else "#757575"
            tk.Label(self.top3_frame, text=f"{i+1}. {genre}: {prob:.1f}%", font=("Arial", 10), fg=color).pack()
        fig = plt.Figure(figsize=(10, 4), tight_layout=True)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(results['features'], aspect='auto', cmap='coolwarm')
        ax1.set_title('MFCC Features (Normalized)')
        fig.colorbar(im, ax=ax1)
        ax2 = fig.add_subplot(122)
        time = np.linspace(0, len(results['waveform']) / self.classifier.processor.sr, len(results['waveform']))
        ax2.plot(time, results['waveform'], linewidth=0.5, alpha=0.7)
        ax2.set_title('Audio Waveform')
        ax2.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MusicGenreClassifierGUI()
    app.run()
