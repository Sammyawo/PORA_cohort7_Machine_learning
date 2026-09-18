import requests
import streamlit as st

API_URL = "http://127.0.0.1:8055/api/predict"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="centered"
)

st.title("  Customer Churn Predictor")
st.write(
    "Fill in a customer's details and check whether the model predicts"
    "they'll **stay** or **leave**. This calls the FastAPI backend running "
    "at `/api/predict`."
)

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=515)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=8, max_value=100, value=42)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=15, value=2)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
            )

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    payload = {
        "credit_score": int(credit_score),
        "geography": geography,
        "age": int(age),
        "gender": gender,
        "tenure": int(tenure),
        "balance": float(balance),
        "num_of_products": int(num_of_products),
        "has_credit_card": 1 if has_credit_card == "Yes" else 0,
        "is_active_member": 1 if is_active_member == "Yes" else 0, 
        "estimated_salary": float(estimated_salary)
    }    

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        churn_status = result["churn_status"]
        probability = result["churn_probability"]

        st.divider()
        if churn_status == "leave":
            st.error(f"Prediction: customer is likely to **leave**")
        else:
            st.success(f"Prediction: customer is likely to **stay**")

        if probability is not None:
            st.metric("Predicted churn probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the API at"
            f"`{API_URL}`. Make sure it's running: `uvicorn api:app --reload`"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
