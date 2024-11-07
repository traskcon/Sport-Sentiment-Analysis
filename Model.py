import nltk
import string
import re
import keras

# Key NLP Pipeline Steps (not yet in order):
# 1. Tokenize the text
tokens = nltk.word_tokenize("text")

# 2. Make text lowercase (case-insensitive)
lowercased_tokens = [token.lower() for token in tokens]

# 3. Remove punctuation
filtered_tokens = [token for token in tokens if token not in string.punctuation]

# 4. Remove stopwords
stopwords = nltk.corpus.stopwords.words("english")
filtered_tokes = [token for token in tokens if token.lower() not in stopwords]

# 5. Remove URLs
pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
cleaned_text = re.sub(pattern, "", "text")

# 6. Remove handles
pattern = r"@[^]+"
cleaned_text = re.sub(pattern, "", "text")