import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define dataset paths
actor_01_path = r'C:\Users\Dell\Downloads\audiospeech\Ravdess\audio_speech_actors_01-24\Actor_01'
ieo_path = r'C:\Users\Dell\Downloads\Savee'  # Update this to the correct path

savee_path = r'C:\Users\Dell\Downloads\audiospeech\Savee'
tess_path=r'C:\Users\Dell\Downloads\audiospeech\Tess'
# Function to get all files from folders
def get_audio_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

# Load files from each dataset
actor_files = get_audio_files(actor_01_path)
ieo_files = get_audio_files(ieo_path)
savee_files = get_audio_files(savee_path)

# Combine file paths and corresponding labels (manually assign labels)
file_paths = actor_files + ieo_files + savee_files
labels = ['happy'] * len(actor_files) + ['angry'] * len(ieo_files) + ['neutral'] * len(savee_files)  # Adjust according to your dataset's labels

# Extract features
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Extract features from audio files
features = [extract_features(file) for file in file_paths]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels_one_hot), test_size=0.2, random_state=42)
# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

# Save the trained model
model.save('emotion_recognition_model.keras')

# Real-time emotion recognition (similar to earlier example)
