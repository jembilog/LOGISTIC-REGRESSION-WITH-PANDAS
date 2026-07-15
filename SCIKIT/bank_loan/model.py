import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("loan_approval.csv")
print(df)
X = df.drop("Approved", axis=1)
y = df["Approved"]

categorical_features = [
    "Employment",
    "Education",
    "MaritalStatus",
    "ExistingLoan"
]
numerical_features = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount"
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))

new_applicant = pd.DataFrame({
    "Age": [35],
    "Income": [70000],
    "CreditScore": [740],
    "Employment": ["Employed"],
    "Education": ["Bachelor"],
    "MaritalStatus": ["Married"],
    "ExistingLoan": ["No"],
    "LoanAmount": [180000]
})
prediction = model.predict(new_applicant)
print("\nPrediction")

if prediction[0] == "Yes":
    print("Loan Approved")
else:
    print("Loan Rejected")
