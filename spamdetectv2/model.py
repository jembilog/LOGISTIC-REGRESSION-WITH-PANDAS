import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spam_dataset.csv")

messages = df["Message"]
Y = df["Spam"].values.reshape(-1,1)

tokens = []
for sentence in messages:
    sentence = sentence.lower()
    words = sentence.split()
    tokens.append(words)

vocab = []
for sentence in tokens:
    for word in sentence:
        if word not in vocab:
            vocab.append(word)

X = []
for sentence in tokens:
    vector = []
    for word in vocab:
        if word in sentence:
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)

X = np.array(X)

np.random.seed(42)

indices = np.random.permutation(len(X))

split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
Y_train = Y[train_indices]

X_test = X[test_indices]
Y_test = Y[test_indices]

print("Training size:", X_train.shape)
print("Testing size:",X_test.shape)

n_features = X_train.shape[1]
w = np.zeros((n_features, 1))
b = 0
learning_rate = 0.1
epochs = 5000

n = len(X)

def sigmoid(z):

    return 1/(1+np.exp(-z))

for epoch in range(epochs):
    z = np.dot(X_train, w) + b
    Y_pred = sigmoid(z)

    dw = (1/n) * np.dot(X_train.T, (Y_pred - Y_train))
    db = (1/n) * np.sum(Y_pred - Y_train)

    w -= learning_rate * dw
    b -= learning_rate * db

#test

z_test = np.dot(X_test, w) + b
probabilities = sigmoid(z_test)
predictions = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predictions == Y_test)
print()
print("Test Accuracy:",
      accuracy*100,
      "%")