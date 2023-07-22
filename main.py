from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import statistics
import tensorflow_io as tfio
import math


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def getStress(yHat,n):
  py_arr = yHat.tolist()
  scores = [80,75,15,15,60,25,80,15,80,80,75,25,15,60]
  tot_val = 0
  for it in py_arr:
    i = 0
    val = 0
    for t in it:
      val += math.log(scores[i])*(t/100)
      i += 1 
    tot_val += math.exp(val*100)      
  return int(tot_val/n)

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([30000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


@app.get("/")
def get_root():
    return {"message": "Hello World"}


model_path = Path("assets/emotion_classifier.h5")


@app.post("/guess_emotion")
async def guessEmotion(file: UploadFile = File(...)):

    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    emotion_list = ['OAF_angry',
                    'YAF_disgust',
                    'YAF_happy',
                    'YAF_neutral',
                    'YAF_sad',
                    'YAF_pleasant_surprised',
                    'YAF_angry',
                    'OAF_happy',
                    'OAF_Fear',
                    'YAF_fear',
                    'OAF_disgust',
                    'OAF_Pleasant_surprise',
                    'OAF_neutral',
                    'OAF_Sad']
    mp3 = file.filename
    wav = load_mp3_16k_mono(mp3)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        wav, wav, sequence_length=30000, sequence_stride=30000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    if os.path.isfile(model_path):
        new_model = load_model(model_path)
        yhat1 = new_model.predict(audio_slices)
        yhat = np.argmax(yhat1, axis=1)
        emo_in = statistics.mode(yhat)
        return {"overall_emotion": emotion_list[emo_in], "stress_level": getStress(yhat1, yhat1.shape[0])}
    else:
        return ({"model_loaded": False, "error": "Model file not found"})
