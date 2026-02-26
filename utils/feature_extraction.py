import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, duration=30)
    y = librosa.util.normalize(y)

    # MFCC 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Chroma 
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Contrast 
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # Spectral Features 
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Tempo 
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    #  Pitch 
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
    pitch_std = np.std(pitch_vals) if len(pitch_vals) > 0 else 0

    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        contrast_mean,
        centroid,
        bandwidth,
        zcr,
        tempo,
        pitch_mean,
        pitch_std
    ])

    return features, tempo, centroid, zcr, pitch_std
