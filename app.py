import streamlit as st
import pandas as pd
import joblib
import importlib.util

# --- Page Functions ---


def prediction_page():
    st.title("Body Fat Prediction App")
    st.write("This app predicts body fat percentage based on user inputs.")

    # Load the trained model
    model_path = "models/bodyfat_model.pkl"
    model = joblib.load(model_path)

    st.header("Input Features")
    features = {
        "Density": st.number_input("Density (g/cm³)", min_value=0.9, max_value=1.2, value=1.0),
        "Age": st.number_input("Age (years)", min_value=1, max_value=120, value=25),
        "Weight": st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0),
        "Height": st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0),
        "Neck": st.number_input("Neck Circumference (cm)", min_value=10.0, max_value=60.0, value=40.0),
        "Chest": st.number_input("Chest Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0),
        "Abdomen": st.number_input("Abdomen Circumference (cm)", min_value=50.0, max_value=200.0, value=90.0),
        "Hip": st.number_input("Hip Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0),
        "Thigh": st.number_input("Thigh Circumference (cm)", min_value=30.0, max_value=100.0, value=60.0),
        "Knee": st.number_input("Knee Circumference (cm)", min_value=20.0, max_value=70.0, value=40.0),
        "Ankle": st.number_input("Ankle Circumference (cm)", min_value=10.0, max_value=50.0, value=25.0),
        "Biceps": st.number_input("Biceps Circumference (cm)", min_value=20.0, max_value=70.0, value=35.0),
        "Forearm": st.number_input("Forearm Circumference (cm)", min_value=15.0, max_value=50.0, value=30.0),
        "Wrist": st.number_input("Wrist Circumference (cm)", min_value=10.0, max_value=30.0, value=15.0),
    }
    input_data = pd.DataFrame([features])
    if st.button("Predict Body Fat Percentage"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Body Fat Percentage: {prediction[0]:.2f}%")


def eda_page():
    eda_path = "eda.py"
    spec = importlib.util.spec_from_file_location("eda", eda_path)
    if spec and spec.loader:
        eda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eda)
    else:
        st.error(
            "Could not load the EDA page. Please check the file path and try again.")


def about_page():
    about_path = "devnev.py"
    spec = importlib.util.spec_from_file_location("devnev", about_path)
    if spec and spec.loader:
        about = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(about)
    else:
        st.error(
            "Could not load the About page. Please check the file path and try again.")

# --- Multipage Dictionary ---
PAGES = {
    "Prediction": prediction_page,
    "EDA": eda_page,
    "About DevNev": about_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Render Selected Page ---
page_func = PAGES[selection]
page_func()
