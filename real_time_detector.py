# real_time_detector.py (Final Copy-Paste Ready Version)

import sounddevice as sd
import librosa
import numpy as np
import sqlite3
import tensorflow.keras.models as keras_models
from alert_handler import send_alert_sms

# Extract MFCC features from real-time audio
def extract_realtime_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Fetch emergency contact number from database using email
def get_emergency_contact(email):
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute("SELECT emergency_contact FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Real-time detection using all models
def real_time_detect_all_models(cnn1d_model, cnn2d_model, lstm_model, emergency_contact):
    duration = 3  # seconds
    sample_rate = 22050

    print("\n🎙 Real-time scream detection using CNN1D + CNN2D + LSTM started. Press Ctrl+C to stop.\n")

    try:
        while True:
            print("⏺ Listening...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()

            audio = audio.flatten()
            features = extract_realtime_features(audio, sr=sample_rate)

            input_1d = features.reshape(1, 40, 1)
            input_2d = features.reshape(1, 10, 4, 1)

            pred1 = cnn1d_model.predict(input_1d)
            pred2 = cnn2d_model.predict(input_2d)
            pred3 = lstm_model.predict(input_1d)

            avg_pred = (pred1 + pred2 + pred3) / 3
            final_label = np.argmax(avg_pred)

            if final_label == 1:
                print("🚨 Scream detected! Sending alert...")
                send_alert_sms(emergency_contact)
            else:
                print("✅ No scream detected.")

    except KeyboardInterrupt:
        print("\n🛑 Detection stopped.")

# Entry function to start real-time detection for given email

def start_realtime_detection(email):
    emergency_contact = get_emergency_contact(email)
    if not emergency_contact:
        print("❌ Emergency contact not found for:", email)
        return

    cnn1d_model = keras_models.load_model("models/cnn1d_model.h5")
    cnn2d_model = keras_models.load_model("models/cnn2d_model.h5")
    lstm_model = keras_models.load_model("models/lstm_model.h5")

    real_time_detect_all_models(cnn1d_model, cnn2d_model, lstm_model, emergency_contact)
