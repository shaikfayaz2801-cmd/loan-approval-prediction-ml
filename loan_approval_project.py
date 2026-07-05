import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
df = pd.read_csv("loan_approval_decision_tree_dataset.csv")

print("\n========== DATASET ==========")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())
emp_encoder = LabelEncoder()
edu_encoder = LabelEncoder()
loan_encoder = LabelEncoder()

df["Employment_Status"] = emp_encoder.fit_transform(
    df["Employment_Status"]
)

df["Education_Level"] = edu_encoder.fit_transform(
    df["Education_Level"]
)

df["Loan_Status"] = loan_encoder.fit_transform(
    df["Loan_Status"]
)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n========== RESULTS ==========")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.savefig("decision_tree.png")
print("\nDecision Tree saved as decision_tree.png")

print("\n========== LOAN PREDICTION ==========")

choice = input(
    "\nDo you want to predict a new loan? (yes/no): "
).lower()

if choice == "yes":

    income = float(input("Annual Income: "))
    credit_score = int(input("Credit Score: "))
    loan_amount = float(input("Loan Amount: "))

    employment = input(
        "Employment Status: "
    )

    education = input(
        "Education Level: "
    )

    debt = float(input("Existing Debt: "))

    employment = emp_encoder.transform(
        [employment]
    )[0]

    education = edu_encoder.transform(
        [education]
    )[0]

    new_data = [[
        income,
        credit_score,
        loan_amount,
        employment,
        education,
        debt
    ]]

    prediction = model.predict(new_data)

    result = loan_encoder.inverse_transform(
        prediction
    )

    print("\nPrediction:", result[0])

print("\nProject Execution Completed Successfully")