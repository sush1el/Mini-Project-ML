# 🎵 Music Genre Classifier

A deep learning-based music genre classification system using CNN and the GTZAN dataset. This project uses PyTorch to classify audio files into 10 different music genres.

## 📋 Features

- **10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **CNN Architecture**: Custom convolutional neural network optimized for audio feature extraction
- **MFCC Features**: Uses Mel-frequency cepstral coefficients for audio representation
- **Dual Interface**: Both GUI and CLI modes for flexibility
- **Real-time Visualization**: Shows waveform and MFCC features during classification
- **Batch Processing**: Can classify entire directories of music files

## 🏗️ Project Structure

```
music-genre-classifier/
│
├── training/
│   └── music_classifier_training.py  # Google Colab training script
│
├── inference/
│   └── music_classifier_demo.py      # VS Code inference script
│
├── models/
│   └── music_genre_classifier_final.pt  # Trained model (downloaded from Colab)
│
├── test_audio/                       # Your test audio files
│   ├── song1.wav
│   ├── song2.mp3
│   └── ...
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Step 1: Train the Model (Google Colab)

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook
3. Copy the entire training script (`music_classifier_training.py`) into the notebook
4. Run all cells sequentially
5. The script will:
   - Download the GTZAN dataset automatically
   - Train the CNN model
   - Save the model as `music_genre_classifier_final.pt`
   - Download the model file to your computer

**Training Tips:**
- Training takes approximately 30-45 minutes on Colab's free GPU
- Expected validation accuracy: 65-75%
- The model file will be about 10-20 MB

### Step 2: Setup Local Environment (VS Code)

1. **Create a project directory:**
```bash
mkdir music-genre-classifier
cd music-genre-classifier
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place files:**
   - Copy `music_classifier_demo.py` to your project directory
   - Copy the downloaded `music_genre_classifier_final.pt` from Colab
   - Add some test audio files (WAV, MP3, FLAC, or M4A format)

### Step 3: Run the Application

#### GUI Mode (Recommended for Demo):
```bash
python music_classifier_demo.py --model music_genre_classifier_final.pt
```

This opens an interactive window where you can:
- Load the model
- Select audio files
- See real-time classification results
- View audio waveforms and MFCC features

#### CLI Mode (Single File):
```bash
python music_classifier_demo.py --model music_genre_classifier_final.pt --mode cli --audio test_audio/song.wav
```

#### CLI Mode (Batch Processing):
```bash
python music_classifier_demo.py --model music_genre_classifier_final.pt --mode cli --directory test_audio/
```

## 📊 Model Architecture

The CNN architecture consists of:
- **Input**: 13 MFCC coefficients × time frames
- **3 Convolutional Blocks**: Each with Conv2D → BatchNorm → ReLU → MaxPool
  - Conv1: 32 filters
  - Conv2: 64 filters  
  - Conv3: 128 filters
- **3 Fully Connected Layers**: 256 → 128 → 10 neurons
- **Output**: 10 genre classes with softmax activation

## 🎯 Performance Metrics

Expected performance on GTZAN dataset:
- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~65-75%
- **Inference Time**: <1 second per 30-second audio clip

Common confusion pairs:
- Rock ↔ Metal (similar characteristics)
- Jazz ↔ Blues (overlapping styles)
- Pop ↔ Country (crossover elements)

## 🛠️ Customization Options

### Adjusting Model Parameters

In the training script, you can modify:
```python
# Audio processing parameters
n_mfcc = 13        # Number of MFCC coefficients
sr = 22050         # Sample rate
duration = 30      # Audio duration in seconds
n_fft = 2048       # FFT window size
hop_length = 512   # Hop length for STFT

# Training parameters
batch_size = 16
learning_rate = 0.001
num_epochs = 50
```

### Adding New Genres

To add new genres:
1. Add audio samples to the GTZAN dataset structure
2. Retrain the model with updated `num_classes`
3. The system will automatically detect new genres

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **"No module named 'torch'"**
   - Solution: Install PyTorch: `pip install torch torchaudio`

2. **"CUDA out of memory" (in Colab)**
   - Solution: Reduce batch_size to 8 or 4

3. **"Audio file not supported"**
   - Solution: Ensure audio files are in WAV, MP3, FLAC, or M4A format

4. **Low accuracy on certain genres**
   - Solution: Increase training epochs or collect more diverse training data

5. **GUI doesn't open on Linux**
   - Solution: Install tkinter: `sudo apt-get install python3-tk`

## 📈 Potential Improvements

- **Data Augmentation**: Add pitch shifting, time stretching, noise injection
- **Advanced Features**: Include spectral centroid, zero-crossing rate, tempo
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Transfer Learning**: Use pre-trained audio models (like VGGish)
- **Real-time Streaming**: Add support for live audio classification
- **Web Deployment**: Create a Flask/FastAPI web application

## 📚 Dataset Information

**GTZAN Music Genre Dataset:**
- 1000 audio tracks (100 per genre)
- 30 seconds each
- 22050 Hz, 16-bit, mono
- Total size: ~1.2 GB
- Source: http://marsyas.info/downloads/datasets.html

## 🤝 Contributing

Feel free to improve the project by:
- Adding more genres
- Implementing new features
- Optimizing the model architecture
- Improving the UI/UX
- Adding unit tests

## 📄 License

This project is for educational purposes. The GTZAN dataset is freely available for academic use.

## 🙏 Acknowledgments

- GTZAN Dataset creators
- PyTorch and Torchaudio teams
- Librosa for audio processing utilities

---

**Happy Classifying! 🎼**