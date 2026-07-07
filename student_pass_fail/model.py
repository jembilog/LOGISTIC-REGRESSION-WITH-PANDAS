import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_data.csv")

X = df["Hours"].values
y = df["Pass"].values

# reshape X to column vector
X = X.reshape(-1, 1)#what does it mean to reshape X to a column vector?
#Reshaping X to a column vector means changing its shape from a one-dimensional array (which is typically represented as a row vector) to a two-dimensional array with one column and multiple rows. In this case, `X` originally contains the number of hours studied for each student, and by reshaping it to a column vector, we ensure that each value of `X` is treated as an individual observation in a format that is compatible with matrix operations used in machine learning algorithms.
X_mean = np.mean(X)
X_std = np.std(X)

X = (X - X_mean) / X_std

w = 0
b = 0

learning_rate = 0.1
epochs = 1000
n = len(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

loss_history = []

for i in range(epochs):

    # forward pass
    z = w * X + b
    y_pred = sigmoid(z)

    # loss (log loss) -> explain why we add 1e-9 to avoid log(0)
    # Adding 1e-9 prevents taking the logarithm of zero, which would result in negative infinity
    loss = -np.mean(
        y * np.log(y_pred + 1e-9) +
        (1 - y) * np.log(1 - y_pred + 1e-9)
    )

    loss_history.append(loss)

    # gradients
    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    # update weights
    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

def predict(x):
    x = (x - X_mean) / X_std
    z = w * x + b
    prob = sigmoid(z)
    return prob, 1 if prob >= 0.5 else 0

# test example
test_hours = 7
prob, pred = predict(test_hours)

print("\nTest Input:", test_hours)
print("Probability:", prob)
print("Prediction:", "Pass" if pred == 1 else "Fail")

plt.plot(loss_history)
plt.title("Loss Curve (Logistic Regression)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# create smooth X range (IMPORTANT)
x_line = np.linspace(X.min(), X.max(), 200)

# compute sigmoid on that
y_line = sigmoid(w * x_line + b)

plt.scatter(X, y, label="Data")
plt.plot(x_line, y_line, color="red", label="Sigmoid Curve")

plt.title("Logistic Regression S-Curve")
plt.legend()
plt.show()