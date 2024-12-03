'''Load trained LSTM model and perform sentiment analysis on Reddit Sports data
Visualize the results'''
import keras

model = keras.models.load_model("initial_model.keras")