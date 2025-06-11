from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import io
import soundfile as sf
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and label encoder
model = tf.keras.models.load_model("cnn_lstm_music_classifier.h5")

# Load or define your labels (change based on your training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("classes.npy", allow_pickle=True)
 # Save this during training

def extract_mel_spectrogram(waveform, sr=16000, n_mels=128):
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def pad_spectrogram(spec, target_shape=(128, 128)):
    spec = tf.convert_to_tensor(spec, dtype=tf.float32)
    spec = tf.image.resize_with_pad(spec[..., tf.newaxis], *target_shape)
    return tf.squeeze(spec).numpy()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and decode audio
    audio_bytes = await file.read()
    waveform, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Preprocessing
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    mel_spec = extract_mel_spectrogram(waveform, sr=sr)
    padded_spec = pad_spectrogram(mel_spec)
    
    # Model expects shape (1, 128, 128)
    input_tensor = np.expand_dims(padded_spec, axis=(0, -1))  # shape: (1, 128, 128, 1)
    
    prediction = model.predict(input_tensor)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return {"prediction": predicted_label}
