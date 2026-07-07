import numpy as np
import pandas as pd

df = pd.read_csv("spam_dataset.csv")

print("DATASET")
print(df)

messages = df["Message"]
y = df["Spam"].values.reshape(-1,1)

print("\nLabels")
print(y)

#tokenization
tokens = []

for sentence in messages:
    sentence = sentence.lower()
    words = sentence.split()
    tokens.append(words)

for t in tokens:
    print(t)

vocab = []
for sentence in tokens:
    for word in sentence:
        if word not in vocab:
            vocab.append(word)
print(vocab)

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
print("\nFEATURE MATRIX")
print(X)
print("\nShape:", X.shape)
print("\n==============================")

for i in range(len(messages)):

    print(messages[i])

    print(X[i])

    print("------------------------------")



n_features = X.shape[1]
w = np.zeros((n_features, 1))
b = 0
learning_rate = 0.1
epochs = 3000
n = len(X)


def sigmoid(z):
    return 1/(1+np.exp(-z))

#training
loss_history = []

for epoch in range(epochs):
    z = np.dot(X, w) + b
    Y_pred = sigmoid(z)

    loss = -np.mean(
        y*np.log(Y_pred+1e-9)+
        (1-y)*np.log(1-Y_pred+1e-9)
    )

    loss_history.append(loss)

    dw = (1/n) * np.dot(X.T, (Y_pred - y))
    db = (1/n) * np.sum(Y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch%300==0:
        print(f"Epoch {epoch} Loss {loss:.4f}")

def predict(message):
    message = message.lower()

    words = message.split()
    vector = []

    for word in vocab:
        if word in words:
            vector.append(1)
        else:
            vector.append(0)
    vector = np.array(vector).reshape(1,-1)

    z = np.dot(vector, w) + b
    probability = sigmoid(z)

    prediction = 1 if probability >= 0.5 else 0

    return probability[0][0], prediction

prob,pred = predict("Meeting at money")

print()

print("Probability:",prob)

if pred==1:

    print("Prediction : SPAM")

else:

    print("Prediction : NOT SPAM")
