import os
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr
import time
import tkinter as tk
from tkinter import messagebox

# Load your pre-trained emotion recognition model
model = tf.keras.models.load_model(r'D:\audiospeech (2)\audiospeech\emotion_recognition_model.keras')

# Define your labels
labels = ['happy', 'angry', 'neutral']  # Update this based on your trained labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Function to extract features for emotion prediction
def extract_features(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# Function to predict emotion
def predict_emotion(audio_data):
    features = extract_features(audio_data)
    predicted_emotion = np.argmax(model.predict(features), axis=-1)
    emotion_label = label_encoder.inverse_transform(predicted_emotion)
    description = get_emotion_description(emotion_label[0])
    return emotion_label[0], description

# Function to describe emotion
def get_emotion_description(emotion):
    descriptions = {
        'happy': "The speaker sounds joyful, positive, and in a good mood.",
        'angry': "The speaker sounds frustrated, aggressive, or annoyed.",
        'neutral': "The speaker has a calm and even-toned demeanor."
    }
    return descriptions.get(emotion, "Emotion not recognized.")

# Function for speech recognition
def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    audio_data = audio_data.flatten()  # Flatten for speech recognition
    audio_data_instance = sr.AudioData(audio_data.tobytes(), 44100, 1)  # Create AudioData instance
    try:
        text = recognizer.recognize_google(audio_data_instance)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Function to record audio
def record_audio():
    print("Recording audio...")
    audio_data = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype='float32')  # Record 5 seconds
    sd.wait()  # Wait until recording is finished
    sf.write('recorded_audio.wav', audio_data, 44100)
    print("Audio saved as 'recorded_audio.wav'")
    return audio_data

# Function to process audio and update GUI
def process_audio():
    audio_data = record_audio()
    
    # Predict emotion
    emotion, description = predict_emotion(audio_data.flatten())  # Flatten audio data for feature extraction
    result_var.set(f"Predicted Emotion: {emotion}\nDescription: {description}")

    # Recognize speech
    spoken_text = speech_to_text(audio_data.flatten())
    result_var.set(f"{result_var.get()}\nRecognized Speech: {spoken_text}")

# GUI setup
app = tk.Tk()
app.title("Emotion and Speech Recognition")
app.geometry("400x300")

result_var = tk.StringVar()
result_label = tk.Label(app, textvariable=result_var, wraplength=350)
result_label.pack(pady=20)

record_button = tk.Button(app, text="Record Audio", command=process_audio)
record_button.pack(pady=10)

exit_button = tk.Button(app, text="Exit", command=app.quit)
exit_button.pack(pady=10)

app.mainloop()
