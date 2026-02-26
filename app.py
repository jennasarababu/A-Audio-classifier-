import streamlit as st
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from utils.feature_extraction import extract_features
from utils.subgenre_logic import classify_subgenre

st.set_page_config(page_title="AI Genre Analyzer", layout="centered")

st.title("🎧 AI Genre Analyzer")
st.markdown("AI-powered Music Genre & Style Detection System")

# ================= LOAD MODELS =================
genre_model = joblib.load("models/genre_model.pkl")
style_model = joblib.load("models/style_model.pkl")
scaler = joblib.load("models/scaler.pkl")

uploaded_file = st.file_uploader("Upload a song", type=["wav", "mp3"])

if uploaded_file:

    # ================= SAVE TEMP FILE (IMPORTANT FIX) =================
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_path = "temp_audio.wav"

    st.audio(uploaded_file)

    # ================= FEATURE EXTRACTION =================
    features, tempo, centroid, zcr, pitch_std = extract_features(audio_path)

    # Ensure numeric safety
    tempo = float(tempo)
    centroid = float(centroid)
    zcr = float(zcr)
    pitch_std = float(pitch_std)

    features_scaled = scaler.transform([features])

    # ================= GENRE PREDICTION =================
    genre_probs = genre_model.predict_proba(features_scaled)[0]
    labels = genre_model.classes_

    genre_index = np.argmax(genre_probs)
    genre = labels[genre_index]
    genre_conf = genre_probs[genre_index]

    top3_idx = np.argsort(genre_probs)[-3:][::-1]

    # ================= STYLE PREDICTION =================
    style_probs = style_model.predict_proba(features_scaled)[0]
    style_index = np.argmax(style_probs)
    style = style_index
    style_conf = style_probs[style_index]

    # Hybrid correction (reduce false Nightcore detection)
    if tempo < 130 or pitch_std < 50:
        style = 0

    subgenre = classify_subgenre(genre, tempo, centroid, zcr, pitch_std)

    # ================= DISPLAY RESULTS =================

    st.subheader("🎼 Prediction Results")

    st.write(f"**Main Genre:** {genre}")
    st.write(f"**Subgenre:** {subgenre}")
    st.write(f"**Style:** {'Nightcore' if style == 1 else 'Original'}")

    st.write(f"🔥 Genre Confidence: {genre_conf*100:.2f}%")
    st.write(f"🔥 Style Confidence: {style_conf*100:.2f}%")
    st.write(f"📈 BPM: {tempo:.2f}")

    # ================= TEMPO CATEGORY =================
    if tempo < 90:
        bpm_category = "Slow"
    elif tempo < 130:
        bpm_category = "Medium"
    else:
        bpm_category = "Fast"

    st.write(f"🎵 Tempo Category: {bpm_category}")

    # ================= TOP 3 GENRES =================
    st.subheader("Top 3 Genre Predictions")

    for i in top3_idx:
        st.write(f"{labels[i]}: {genre_probs[i]*100:.2f}%")

    # ================= PROBABILITY BAR CHART =================
    st.subheader("Genre Probability Distribution")

    df = pd.DataFrame({
        "Genre": labels,
        "Probability": genre_probs
    })

    st.bar_chart(df.set_index("Genre"))

    # ================= RADAR CHART =================
    st.subheader("Audio Feature Radar")

    radar_labels = ['BPM', 'Brightness', 'Energy', 'Pitch Var']

    radar_values = [
        min(tempo / 200, 1),
        min(centroid / 5000, 1),
        min(zcr * 5, 1),
        min(pitch_std / 300, 1)
    ]

    radar_values.append(radar_values[0])

    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, radar_values)
    ax.fill(angles, radar_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)

    st.pyplot(fig)
    

   