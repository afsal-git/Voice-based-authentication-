import os
import numpy as np
from pydub import AudioSegment
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tempfile
import shutil

# Convert MP3 to temporary WAV file
def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

# Extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Load dataset of speaker folders
def load_dataset(data_folder):
    X = []
    y = []

    for speaker in os.listdir(data_folder):
        speaker_path = os.path.join(data_folder, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for file in os.listdir(speaker_path):
            if file.endswith(".mp3"):
                mp3_path = os.path.join(speaker_path, file)
                wav_path = convert_mp3_to_wav(mp3_path)
                features = extract_features(wav_path)
                os.remove(wav_path)  # Clean up

                if features is not None:
                    X.append(features)
                    y.append(speaker)

    return np.array(X), np.array(y)

# Train SVM model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return clf

# Authenticate a voice sample
def authenticate(model, mp3_file):
    wav_path = convert_mp3_to_wav(mp3_file)
    features = extract_features(wav_path)
    os.remove(wav_path)

    if features is None:
        return "Unknown", 0.0

    prediction = model.predict([features])
    confidence = model.predict_proba([features]).max()
    return prediction[0], confidence

# === MAIN PROGRAM ===
dataset_path = "/content/voice_data"  # This should be the parent folder of speaker subfolders
print("🔄 Loading and processing dataset...")
X, y = load_dataset(dataset_path)

if len(X) == 0:
    print("❌ No valid data found. Please check your dataset path and audio files.")
else:
    print("🎯 Training model...")
    model = train_model(X, y)

    # Example authentication
    test_sample = "/content/voice_data/afsal/afsal11.mp3"
    if os.path.exists(test_sample):
        user, confidence = authenticate(model, test_sample)
        print(f"🔐 Authenticated as: {user} (confidence: {confidence:.2f})")
    else:
        print("❗ Test voice file not found. Please provide a valid path.")
