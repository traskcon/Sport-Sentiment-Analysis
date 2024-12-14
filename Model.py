'''Build LSTM model in Keras and train on Sentiment140 dataset
Training sentiment scores are 0 (negative), 2 (neutral), 4 (positive)
As a result, LSTM acts as regressor, not classifier'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import keras
from keras_preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression


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
def build_model(embedding_matrix, embedding_dim, num_tokens):
    # Create Embedding Layer
    embedding_layer = keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
    )
    embedding_layer.build((1,))
    embedding_layer.set_weights([embedding_matrix])
    # model initialization
    model = keras.Sequential([
        keras.layers.Input((None,)),
        embedding_layer,
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation=None)
    ])
    # compile model
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mse'])
    return model

def embedding_for_vocab(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))
    embeddings_index = {}
    hits, misses = 0, 0
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            embeddings_index[word] = np.array(vector, dtype=np.float32)[:embedding_dim]
    for word, i in word_index.items():     
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_vocab[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Embedded {} words ({} misses)".format(hits, misses))
    return embedding_matrix_vocab

if __name__ == "main":
    sentiment_data = pd.read_csv("training.1600000.processed.noemoticon.csv", names=["sentiment","id","date","query","user","comment"])
    cleaned_data = preprocess_data(sentiment_data)
    #cleaned_data.to_csv("clean_training_data.csv", index=False)
    #print(data_summary(cleaned_data))
    comments = cleaned_data["comment"].values
    labels = cleaned_data["sentiment"].values
    train_comments, test_comments, train_labels, test_labels = train_test_split(comments, labels, stratify = labels)

    vectorizer = keras.layers.TextVectorization(max_tokens=100000, output_sequence_length=50)
    # Only consider the top 100,000 words, set sequence length to 50 (length of longest comment)
    vectorizer.adapt(comments)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    embedding_dim = 50
    embedding_matrix_vocab = embedding_for_vocab('./glove.twitter.27B.'+str(embedding_dim)+'d.txt', word_index,
    embedding_dim)

    model = build_model(embedding_matrix_vocab, embedding_dim, len(word_index)+1)
    print(model.summary())

    x_train = np.array(vectorizer(np.array([[s] for s in train_comments])))
    x_test = np.array(vectorizer(np.array([[s] for s in test_comments])))
    print(x_train.shape)

    model.fit(x_train, train_labels, epochs=3)

    np.save("x_train", x_train)
    np.save("x_test", x_test)
    np.save("train-labels", train_labels)
    np.save("test-labels",test_labels)
    np.save("embedding-matrix",embedding_matrix_vocab)
    model.save("glove_model.keras")