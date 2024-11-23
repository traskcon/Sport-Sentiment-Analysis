import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import keras


## DATA PRE-PROCESSING FUNCTIONS

def remove_tags(string):
    result = re.sub('','',str(string))          #remove HTML tags
    result = re.sub('http\S+','',result)   #remove URLs
    result = re.sub(r"@\w+", "", result) #Remove handles
    result = re.sub(r'\W+', ' ',result)    #remove non-alphanumeric characters 
    result = result.lower() #Make text lowercase
    return result

def lemmatize_text(text, w_tokenizer=nltk.tokenize.WhitespaceTokenizer(), lemmatizer=nltk.stem.WordNetLemmatizer()):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

def preprocess_data(data):
    #NLP Data Pre-processing Pipeline:
    # Remove HTML tags, URLs, etc.
    data['comment'] = data['comment'].apply(lambda cw : remove_tags(cw))
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    data['comment'] = data['comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Lemmatize data
    data['comment'] = data.comment.apply(lemmatize_text)
    return data

def data_summary(cleaned_data):
    s = 0.0
    max_length = 0
    for i in cleaned_data['comment']:
        word_list = i.split()
        s = s + len(word_list)
        if len(word_list) > max_length:
            max_length = len(word_list)
    print("Average length of each comment : ",s/cleaned_data.shape[0])
    print("Max comment length: ", max_length)
    pos = 0
    neg = 0
    for i in range(cleaned_data.shape[0]):
        if cleaned_data.iloc[i]['sentiment'] == 4:
            pos += 1
        elif cleaned_data.iloc[i]["sentiment"] == 0:
            neg += 1
    print("Percentage of reviews with positive sentiment is "+str(pos/cleaned_data.shape[0]*100)+"%")
    print("Percentage of reviews with negative sentiment is "+str(neg/cleaned_data.shape[0]*100)+"%")


## MODEL FUNCTIONS
def build_model(train_comments):
    # Model hyperparameters
    embedding_dim = 100
    max_length = 100
    vocab_size = 3000
    # Create tokenizer layer
    encoder = keras.layers.TextVectorization(max_tokens=vocab_size,
                                            standardize=None,
                                            output_sequence_length=max_length)
    encoder.adapt(train_comments)
    print(np.array(encoder.get_vocabulary())[:20])
    # model initialization
    #TODO: Add input layer to the sequential model
    model = keras.Sequential([
        encoder,
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # compile model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

sentiment_data = pd.read_csv("training.1600000.processed.noemoticon.csv", names=["sentiment","id","date","query","user","comment"])
cleaned_data = preprocess_data(sentiment_data)
#data_summary(cleaned_data)
comments = cleaned_data["comment"].values
labels = cleaned_data["sentiment"].values
train_comments, test_comments, train_labels, test_labels = train_test_split(comments, labels, stratify = labels)
model = build_model(train_comments)
print(model.summary())