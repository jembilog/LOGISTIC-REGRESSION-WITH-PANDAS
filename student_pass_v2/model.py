import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_pass_features.csv")

X = df[["Hours", "Sleep", "Attendance"]].values
Y = df["Pass"].values.reshape(-1, 1)  # Reshape Y to a column vector
#why y
# Y represents the target variable (pass/fail), and reshaping it to a column vector ensures it's compatible with matrix operations in the logistic regression implementation.
#why not x
# X represents the feature matrix, and it doesn't need to be reshaped to a column vector as it's already in the correct format for matrix operations.

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

n_shape  = X.shape[1]  # Number of features
w = np.zeros((n_shape, 1))  # Initialize weights as a column vector
b = 0  # Initialize bias as a scalar

learning_rate = 0.1
epochs = 5000
n = len(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

loss_history = []

for epoch in range(epochs):
    # forward pass
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    # loss (log loss)
    loss = -np.mean(
        Y * np.log(y_pred + 1e-9) +
        (1 - Y) * np.log(1 - y_pred + 1e-9)
    )
    loss_history.append(loss)

    # gradients
    dw = np.dot(X.T, (y_pred - Y)) / n
    db = np.mean(y_pred - Y)

    # update weights
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}") 

def predict(hours, sleep, attendane):
    sample = np.array([[hours, sleep, attendane]])

    sample = (sample - X_mean) / X_std  # Normalize the sample using the same mean and std as training data
    # z = np.dot(sample, w) + b
    prob = sigmoid(np.dot(sample, w) + b)
    pred = 1 if prob >= 0.5 else 0
    return pred, prob[0][0]
    #or
    # z = np.dot(sample, w) + b
    # prob = sigmoid(z)
    # pred = 1 if prob >= 0.5 else 0
    # return pred, prob[0][0]

test_pred, test_prob = predict(7, 6, 90)

print("\n--- TEST ---")
print("Probability:", test_prob)

if test_pred == 1:
    print("PASS")
else:
    print("FAIL")