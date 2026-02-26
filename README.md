#🎧 AI -POWERED Music Genre and Style Analyzer
AI -Powered Music Gebre and Style Analyzer is a lightweight system that can analyze an uploaded audio and detect it's primary genre aswell as it's subgenre and whether there is any transformation or style changes
Built using Python ,Librosa, Sckit-Learn and Streamlit
## 🚀 Features

- 🎼 Primary Genre Classification (using SVM)
- 🎧 Subgenre Detection (rule-based intelligence layer)
- ⚡ Nightcore Style Detection (tempo + pitch analysis)
- 📈 BPM (Tempo) Estimation
- 📊 Radar Visualization of Musical Attributes
- 🖥 Interactive Streamlit Web Interface
- 💡 CPU-based lightweight implementation (No GPU required)

---

## 🧠 Technical Approach

### 1️⃣ Feature Extraction (Librosa)
- MFCC (Timbre)
- Chroma Features (Harmony)
- Spectral Contrast (Texture)
- Zero Crossing Rate (Energy)
- Tempo (Beat Tracking)
- Pitch Statistics

### 2️⃣ Machine Learning
- Feature Scaling using StandardScaler
- SVM (RBF Kernel) for Genre Classification
- Secondary SVM for Style Detection

### 3️⃣ Intelligent Layer
- Subgenre classification using tempo and spectral thresholds
- Nightcore detection using tempo and pitch shifts

---

## 📊 System Architecture


Audio Input
↓
Feature Extraction (Librosa)
↓
Feature Scaling
↓
SVM Classifier
↓
Genre + Subgenre + Style Output
↓
Radar Visualization

