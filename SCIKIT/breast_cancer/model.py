import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

X = df.drop("Target", axis=1)
y = df["Target"]

X_train ,X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))
print("\nROC-AUC Score")
print(roc_auc_score(y_test, probabilities[:, 1]))

new_patient = pd.DataFrame(
    [X.iloc[0]],
    columns=X.columns
)

new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print("\nPrediction")

if prediction[0] == 1:
    print("Benign Tumor")
else:
    print("Malignant Tumor")

print(f"Probability of Benign: {probability[0][1]:.2%}")
print(f"Probability of Malignant: {probability[0][0]:.2%}")
