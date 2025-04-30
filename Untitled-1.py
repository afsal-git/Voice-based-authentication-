
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Simulated speaker recognition function
def recognize_speaker(audio_path):
    # Here you'd load your ML model and analyze the audio
    # Replace this with real logic using speech recognition, voice embeddings, etc.
    return "John Doe"

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files['audio']
    file_path = os.path.join('uploads', audio.filename)
    os.makedirs('uploads', exist_ok=True)
    audio.save(file_path)

    speaker_name = recognize_speaker(file_path)

    # Optionally, delete the file after processing
    os.remove(file_path)

    return jsonify({"speaker": speaker_name})

if __name__ == '__main__':
    app.run(debug=True)
