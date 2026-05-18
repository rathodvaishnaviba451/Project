import joblib
import numpy as np
import streamlit as st


# ===============================
# LOAD MODEL
# ===============================

model = joblib.load(
    "saved_model.pkl"
)


# ===============================
# UI
# ===============================

st.title(
    "Credit Card Fraud Detection"
)

st.write(
    "Enter transaction values"
)


inputs = []


for i in range(30):

    value = st.number_input(
        f"Feature {i+1}",
        value=0.0
    )

    inputs.append(
        value
    )


# ===============================
# PREDICTION
# ===============================

if st.button(
    "Predict"
):

    sample = np.array(
        inputs
    ).reshape(1, -1)

    prediction = model.predict(
        sample
    )[0]

    probability = model.predict_proba(
        sample
    )[0][1]


    if prediction == 1:

        st.error(
            f"Fraudulent Transaction ({probability:.2%})"
        )

    else:

        st.success(
            f"Legitimate Transaction ({1-probability:.2%})"
        )