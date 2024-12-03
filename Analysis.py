'''Load trained LSTM model and perform sentiment analysis on Reddit Sports data
Visualize the results'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Model import preprocess_data
import pandas as pd
import keras

model = keras.models.load_model("initial_model.keras")
data = pd.read_csv("reddit_data_nhl.csv", names=["team","id","timestamp","comment"])
cleaned_data = preprocess_data(data)
data["predicted sentiment"] = model.predict(cleaned_data["comment"])