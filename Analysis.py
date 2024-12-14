'''Load trained LSTM model and perform sentiment analysis on Reddit Sports data
Visualize the results'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Model import preprocess_data
import pandas as pd
import matplotlib.pyplot as plt
import keras
import pickle

model = keras.saving.load_model("glove_model.keras")
with open('logistic-classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
print(model.summary())

'''data = pd.read_csv("reddit_data.csv")
cleaned_data = preprocess_data(data)
data["predicted sentiment"] = model.predict(cleaned_data["comment"])
plt.hist(data["predicted sentiment"], bins=40)
plt.xlabel("Predicted Sentiment Score")
plt.ylabel("Number of Messages")
plt.title("Sentiment Distribution across Sports Subreddits")
plt.show()'''