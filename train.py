import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from utils.feature_extraction import extract_features

ORIGINAL_PATH = "dataset/gtzan_original"
NIGHTCORE_PATH = "dataset/gtzan_nightcore"

os.makedirs("models", exist_ok=True)

# =========================
# GENRE TRAINING
# =========================

print("Starting Genre Training...")

X_genre = []
y_genre = []

for genre in os.listdir(ORIGINAL_PATH):
    genre_path = os.path.join(ORIGINAL_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            features, _, _, _, _ = extract_features(file_path)
            X_genre.append(features)
            y_genre.append(genre)
        except:
            print(f"Skipping corrupted file: {file_path}")
            continue

X_genre = np.array(X_genre)

print(f"Total genre samples: {len(X_genre)}")

# ===== SCALE FEATURES =====
scaler = StandardScaler()
X_genre = scaler.fit_transform(X_genre)

joblib.dump(scaler, "models/scaler.pkl")

# ===== STRATIFIED SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X_genre,
    y_genre,
    test_size=0.2,
    random_state=42,
    stratify=y_genre
)

# ===== SVM MODEL =====
genre_model = SVC(
    kernel='rbf',
    C=50,
    gamma='scale',
    probability=True,
    random_state=42
)

genre_model.fit(X_train, y_train)

pred = genre_model.predict(X_test)
genre_accuracy = accuracy_score(y_test, pred)

print(f"Genre Model Accuracy (SVM): {genre_accuracy*100:.2f}%")

joblib.dump(genre_model, "models/genre_model.pkl")
print("Genre model saved.")


# =========================
# STYLE TRAINING (Nightcore)
# =========================

print("\nStarting Style (Nightcore) Training...")

X_style = []
y_style = []

# Original = 0
for genre in os.listdir(ORIGINAL_PATH):
    genre_path = os.path.join(ORIGINAL_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            features, _, _, _, _ = extract_features(file_path)
            X_style.append(features)
            y_style.append(0)
        except:
            continue

# Nightcore = 1
for genre in os.listdir(NIGHTCORE_PATH):
    genre_path = os.path.join(NIGHTCORE_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            features, _, _, _, _ = extract_features(file_path)
            X_style.append(features)
            y_style.append(1)
        except:
            continue

X_style = np.array(X_style)

print(f"Total style samples: {len(X_style)}")

# Use same scaler
X_style = scaler.transform(X_style)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_style,
    y_style,
    test_size=0.2,
    random_state=42,
    stratify=y_style
)

style_model = SVC(
    kernel='rbf',
    C=50,
    gamma='scale',
    probability=True,
    random_state=42
)

style_model.fit(X_train_s, y_train_s)

pred_s = style_model.predict(X_test_s)
style_accuracy = accuracy_score(y_test_s, pred_s)

print(f"Style Model Accuracy (SVM): {style_accuracy*100:.2f}%")

joblib.dump(style_model, "models/style_model.pkl")
print("Style model saved.")

print("\nTraining completed successfully.")