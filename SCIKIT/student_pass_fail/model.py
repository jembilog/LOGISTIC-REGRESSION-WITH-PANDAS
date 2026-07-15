import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student_pass.csv")
numerical_cols = [
    "StudyHours",
    "Attendance",
    "SleepHours",
    "AssignmentsCompleted",
    "PreviousGrade"
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols = [
    "Participation",
    "InternetAccess"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna(subset=["Result"])

par_encoder = LabelEncoder()
internet_encoder = LabelEncoder()
result_encoder = LabelEncoder()

df["Participation"] = par_encoder.fit_transform(df["Participation"])
df["InternetAccess"] = internet_encoder.fit_transform(df["InternetAccess"])
df["Result"] = result_encoder.fit_transform(df["Result"])
print("\nEncoded Labels")
print("Participation:", par_encoder.classes_)
print("Internet Access:", internet_encoder.classes_)
print("Result:", result_encoder.classes_)

X = df.drop("Result", axis=1)
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
model =  LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))

new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [90],
    "SleepHours": [7],
    "AssignmentsCompleted": [9],
    "PreviousGrade": [87],
    "Participation": ["High"],
    "InternetAccess": ["Yes"]
})

new_student["Participation"] = par_encoder.transform(new_student["Participation"])
new_student["InternetAccess"] = internet_encoder.transform(new_student["InternetAccess"])


prediction = model.predict(new_student)

print("\nPrediction")

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")
