import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv("titanic.csv")

X= df.drop(columns=["Survived", "PassengerId" , "Name", "Ticket", "Cabin"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

numeric_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_features = ["Sex", "Embarked"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first",sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))
print(log_loss(y_test,predictions))
probabilities = model.predict_proba(X_test)
print(probabilities[:,1])
