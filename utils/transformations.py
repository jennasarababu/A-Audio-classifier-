import librosa
import soundfile as sf

def create_nightcore(input_path, output_path):
    y, sr = librosa.load(input_path)

    y_fast = librosa.effects.time_stretch(y, rate=1.2)
    y_shift = librosa.effects.pitch_shift(y_fast, sr=sr, n_steps=3)

    sf.write(output_path, y_shift, sr)