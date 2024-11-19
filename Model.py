import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
import pandas as pd
import re
#import keras

def remove_tags(string):
    result = re.sub('','',str(string))          #remove HTML tags
    result = re.sub('https://.*','',result)   #remove URLs
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

nhl_data = pd.read_csv("reddit_data_nhl.csv", names=["team","id","date","comment"])
cleaned_data = preprocess_data(nhl_data)
print(cleaned_data.head(10))