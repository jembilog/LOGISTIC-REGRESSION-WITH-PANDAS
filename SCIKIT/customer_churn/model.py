import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("customer_churn.csv")

gender_encoder = LabelEncoder()
contract_encoder = LabelEncoder()
internet_encoder = LabelEncoder()
churn_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["ContractType"]= contract_encoder.fit_transform(df["ContractType"])
df["InternetService"] = internet_encoder.fit_transform(df["InternetService"])
df["Churn"] = churn_encoder.fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nAccuracy")
print(accuracy_score(y_test, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report")
print(classification_report(y_test, predictions))

new_customer = pd.DataFrame({
    "Age":[34],
    "Gender":["Male"],
    "Tenure":[8],
    "MonthlyCharges":[68.5],
    "ContractType":["Month-to-month"],
    "InternetService":["Fiber"],
    "SupportCalls":[3]
})
new_customer["Gender"] = gender_encoder.transform(new_customer["Gender"])
new_customer["ContractType"] = contract_encoder.transform(new_customer["ContractType"])
new_customer["InternetService"] = internet_encoder.transform(new_customer["InternetService"])
new_customer = scaler.transform(new_customer)
prediction = model.predict(new_customer)
print("\nPrediction")
if prediction[0] == 1:
    print("Customer will CHURN")
else:
    print("Customer will STAY")
