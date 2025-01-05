'''Load trained LSTM model and perform sentiment analysis on Reddit Sports data
Visualize the results'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Model import preprocess_data
import pandas as pd
import matplotlib.pyplot as plt
import keras
import pickle
import numpy as np

model = keras.saving.load_model("glove_model.keras")

data = pd.read_csv("reddit_data.csv")
cleaned_data = preprocess_data(data)

from_disk = pickle.load(open("tv_layer.pkl", "rb"))
vectorizer = keras.layers.TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(cleaned_data["comment"])
vectorizer.set_weights(from_disk['weights'])

model_data = np.array(vectorizer(np.array([[s] for s in cleaned_data["comment"]])))

data["predicted sentiment"] = model.predict(model_data)
data.to_csv("Sentiment-Data.csv",index=False)
plt.hist(data["predicted sentiment"], bins=40)
plt.xlabel("Predicted Sentiment Score")
plt.ylabel("Number of Messages")
plt.title("Sentiment Distribution across Sports Subreddits")
plt.show()