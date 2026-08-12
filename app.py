import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("order_cancel_model.pkl")

# Load dataset only for Order IDs
df = pd.read_csv("Sale.csv")

# Page settings
st.set_page_config(
    page_title="E-Commerce Order Prediction",
    page_icon="🛒",
    layout="centered"
)

# Title
st.title("🛒 E-Commerce Order Prediction")
st.write("Select an Order ID from the dataset.")

st.divider()

# Get all Order IDs
order_ids = df["Order ID"].astype(str).tolist()

# Select full Order ID
selected_order = st.selectbox(
    "📦 Select Order ID",
    order_ids
)

# Show selected ID
st.info(f"Selected Order ID: {selected_order}")

# Predict
if st.button(
    "🔮 Predict Order",
    use_container_width=True
):

    # Convert Order ID to number
    order_number = int(
        selected_order.replace("-", "")
    )

    # Model input
    input_data = pd.DataFrame({
        "Order_ID": [order_number]
    })

    # Prediction
    prediction = int(
        model.predict(input_data)[0]
    )

    st.divider()

    st.subheader("Prediction Result")

    st.write(
        f"**Order ID:** `{selected_order}`"
    )

    if prediction == 1:

        st.success("✅ ORDER SHIPPED")

        st.metric(
            "Prediction",
            "1"
        )

    else:

        st.error("❌ ORDER CANCELLED")

        st.metric(
            "Prediction",
            "0"
        )