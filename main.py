import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Sale.csv")

print("Original Data:", df.shape)

# Remove index
df = df.drop(columns=["index"], errors="ignore")

# -----------------------------------------
# Convert Order ID to numeric
# -----------------------------------------

df["Order_ID"] = (
    df["Order ID"]
    .astype(str)
    .str.replace("-", "", regex=False)
    .astype("int64")
)

# -----------------------------------------
# Target
# Cancelled = 0
# Shipped = 1
# -----------------------------------------

df["target"] = (
    df["Status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne("cancelled")
    .astype(int)
)

print("\nStatus values:")
print(df["Status"].value_counts())

print("\nTarget values:")
print(df["target"].value_counts())

# -----------------------------------------
# Feature
# -----------------------------------------

X = df[["Order_ID"]]
y = df["target"]

# -----------------------------------------
# Train Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# Random Forest
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------------------
# Accuracy
# -----------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy: {:.2f}%".format(
    accuracy * 100
))

# -----------------------------------------
# Save model
# -----------------------------------------

joblib.dump(model, "order_cancel_model.pkl")

print("\norder_cancel_model.pkl created successfully!")