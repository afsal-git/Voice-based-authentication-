import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import librosa
from sklearn.mixture import GaussianMixture

app = Flask(__name__)

# Configuration for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store user voices
USER_VOICES = {}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract audio features (MFCC)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Save the speaker model to disk
def save_model():
    with open('speaker_model.pkl', 'wb') as f:
        pickle.dump(USER_VOICES, f)

# Load the speaker model
def load_model():
    global USER_VOICES
    if os.path.exists('speaker_model.pkl'):
        with open('speaker_model.pkl', 'rb') as f:
            USER_VOICES = pickle.load(f)

# Train a new model using stored voices
def train_model():
    X = []
    y = []
    for user, audio_paths in USER_VOICES.items():
        for path in audio_paths:
            features = extract_features(path)
            X.append(features)
            y.append(user)

    X = np.array(X)
    y = np.array(y)

    # Use Gaussian Mixture Model for speaker recognition
    model = GaussianMixture(n_components=len(USER_VOICES))
    model.fit(X)
    
    # Save model for authentication
    with open('gmm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load the trained GMM model
def load_gmm_model():
    if os.path.exists('gmm_model.pkl'):
        with open('gmm_model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None

# Route to register new user's voice
@app.route('/register', methods=['POST'])
def register_voice():
    user_name = request.form['username']
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    if user_name not in USER_VOICES:
        USER_VOICES[user_name] = []
    
    USER_VOICES[user_name].append(file_path)
    
    # Re-train the model after adding new voice
    train_model()
    save_model()
    
    return jsonify({"message": f"Voice registered for {user_name}!"})

# Route to authenticate user based on voice
@app.route('/authenticate', methods=['POST'])
def authenticate_voice():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load the trained GMM model
    gmm_model = load_gmm_model()
    if gmm_model is None:
        return jsonify({"error": "Model not trained yet."}), 400

    # Extract features of the uploaded file
    features = extract_features(file_path)

    # Predict the speaker from the GMM model
    predicted_speaker = gmm_model.predict([features])

    return jsonify({"message": f"Authenticated as: {predicted_speaker[0]}"})

if __name__ == '__main__':
    # Load the existing model (if any) when the server starts
    load_model()
    app.run(debug=True)
