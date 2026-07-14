import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions,zero_division=0))
new_patient = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [140],
    "BloodPressure": [80],
    "SkinThickness": [30],
    "Insulin": [120],
    "BMI": [42.5],
    "DiabetesPedigreeFunction": [0.85],
    "Age": [55]
})
new_patient = scaler.transform(new_patient)
prediction = model.predict(new_patient)
print("\nPrediction")
if prediction[0] == 1:
    print("Diabetic")
else:
    print("Not Diabetic")
