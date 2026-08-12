import pandas as pd
import joblib

# Load trained model
model = joblib.load("order_cancel_model.pkl")

# Order IDs to predict
order_ids = [
    "405-8078784-5731545",
    "171-9198151-1101146",
    "404-0687676-7273146",
    "403-9615377-8133951",
    "407-1069790-7240320"
]

# Convert Order ID to numeric
order_numbers = [
    int(order_id.replace("-", ""))
    for order_id in order_ids
]

# Create input
X = pd.DataFrame({
    "Order_ID": order_numbers
})

# Prediction
prediction = model.predict(X)

# Only output
print(prediction.astype(int).tolist())