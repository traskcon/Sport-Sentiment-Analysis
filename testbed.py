import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle
import keras

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
train_labels = np.load("train-labels.npy")
test_labels = np.load("test-labels.npy")
embedding_matrix_vocab = np.load("embedding-matrix.npy")

'''embedded_train = np.zeros((1200000,50,50))
embedded_test = np.zeros((x_test.shape[0],50,50))
for i in tqdm(range(x_train.shape[0])):
    for j in range(x_train.shape[1]):
        embedded_train[i,:,j] = embedding_matrix_vocab[x_train[i,j]]

for i in tqdm(range(x_test.shape[0])):
    for j in range(x_test.shape[1]):
        embedded_test[i,:,j] = embedding_matrix_vocab[x_test[i,j]]

# Logistic Regression needs (n_samples, n_features)
lr_train = np.reshape(embedded_train, (1200000,2500))
lr_test = np.reshape(embedded_test, (x_test.shape[0],2500))
print("Data ready")
clf = LogisticRegression(random_state=0, solver="sag", verbose=True, tol=0.001, max_iter=20).fit(lr_train, train_labels)
with open('logistic-classifier.pkl','wb') as f:
    pickle.dump(clf,f)
print(clf.score(lr_test, test_labels))'''

model = keras.saving.load_model("glove_model.keras")
preds = model.predict(x_test)
# Round predictions to calculate accuracy values
test_preds = np.where(preds > 2, 4, 0).flatten()
print(test_preds.shape)
print(test_labels.shape)
print("LSTM Model Accuracy: {:.4f}".format(np.count_nonzero(test_preds == test_labels)/len(test_labels)))