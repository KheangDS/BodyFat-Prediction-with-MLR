import streamlit as st
import pandas as pd
import joblib
import importlib.util
import os

# ---------- Utility functions ----------
def water_correction_factor(temp_c):
    """Return approximate water density (kg/L) from temp in C."""
    return 0.99987 + (-0.00007 * temp_c) + (-0.000002 * (temp_c ** 2))

def compute_density(WA, WW, LV, temp_c):
    """Compute density from underwater weighing formula."""
    cf = water_correction_factor(temp_c)
    denom = ((WA - WW) / cf) - LV
    if denom <= 0:
        raise ValueError("Computed denominator <= 0. Check WA, WW, LV inputs.")
    density = WA / denom
    return density

def load_density_model(path="model/density_model.pkl"):
    """Load density-only model saved as a dict {'model':..., 'feature_names': [...] }."""
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    # backward compatibility if user saved just sklearn model
    if isinstance(payload, dict) and "model" in payload:
        return payload
    else:
        # saved raw model (older). Wrap it
        return {"model": payload, "feature_names": ["Density"]}

# --- Page Functions ---
# def prediction_page():
#     st.title("Body Fat Prediction App")
#     st.write("This app predicts body fat percentage based on user inputs.")

#     # Load the trained model
#     model_path = "models/bodyfat_model.pkl"
#     model = joblib.load(model_path)

#     st.header("Input Features")

#     # --- User inputs (NO direct Density input) ---
#     age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
#     weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
#     height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)

#     neck = st.number_input("Neck Circumference (cm)", min_value=10.0, max_value=60.0, value=40.0)
#     chest = st.number_input("Chest Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0)
#     abdomen = st.number_input("Abdomen Circumference (cm)", min_value=50.0, max_value=200.0, value=90.0)
#     hip = st.number_input("Hip Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0)
#     thigh = st.number_input("Thigh Circumference (cm)", min_value=30.0, max_value=100.0, value=60.0)
#     knee = st.number_input("Knee Circumference (cm)", min_value=20.0, max_value=70.0, value=40.0)
#     ankle = st.number_input("Ankle Circumference (cm)", min_value=10.0, max_value=50.0, value=25.0)
#     biceps = st.number_input("Biceps Circumference (cm)", min_value=20.0, max_value=70.0, value=35.0)
#     forearm = st.number_input("Forearm Circumference (cm)", min_value=15.0, max_value=50.0, value=30.0)
#     wrist = st.number_input("Wrist Circumference (cm)", min_value=10.0, max_value=30.0, value=15.0)

#     # --- Compute Density from Weight and Height ---
#     # Replace this formula with your real density formula if needed
#     weight_g = weight * 1000
#     density = weight_g / height ** 3  # example: BMI-style formula

#     st.markdown(f"**Computed Density (from weight & height):** `{density:.4f}`")

#     # --- Build input DataFrame for the model ---
#     features = {
#         "Density": density,   # use computed density
#         "Age": age,
#         "Weight": weight,
#         "Height": height,
#         "Neck": neck,
#         "Chest": chest,
#         "Abdomen": abdomen,
#         "Hip": hip,
#         "Thigh": thigh,
#         "Knee": knee,
#         "Ankle": ankle,
#         "Biceps": biceps,
#         "Forearm": forearm,
#         "Wrist": wrist,
#     }

#     input_data = pd.DataFrame([features])

#     if st.button("Predict Body Fat Percentage"):
#         prediction = model.predict(input_data)
#         st.success(f"Predicted Body Fat Percentage: {prediction[0]:.2f}%")

def prediction_page():
    st.title("Body Fat Prediction App")
    st.write("Predict body fat percentage. Preferred method: Underwater inputs to compute density.")

    # Try to load density-only model first
    density_model_payload = load_density_model("model/density_model.pkl")

    # If not available, try legacy model path
    legacy_path = "models/bodyfat_model.pkl"
    legacy_payload = None
    if not density_model_payload and os.path.exists(legacy_path):
        legacy_payload = joblib.load(legacy_path)

    st.header("Method A — Underwater Weighing Inputs (recommended)")

    with st.form("underwater_form"):
        WA = st.number_input("Weight in air (WA) — kg", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        WW = st.number_input("Weight in water (WW) — kg", min_value=0.0, max_value=300.0, value=3.2, step=0.01)
        temp_c = st.number_input("Water temperature (°C)", min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        LV = st.number_input("Residual lung volume (LV) — liters", min_value=0.0, max_value=6.0, value=1.2, step=0.01)
        submit = st.form_submit_button("Compute Density & Predict Body Fat")

    if submit:
        try:
            density = compute_density(WA, WW, LV, temp_c)
            st.markdown(f"**Computed Body Density:** `{density:.4f}` kg/L")

            # Prepare input for model
            input_df = pd.DataFrame([{"Density": density}])

            if density_model_payload:
                model = density_model_payload["model"]
                # predict
                pred = model.predict(input_df[ density_model_payload.get("feature_names", ["Density"]) ])
                st.success(f"Predicted Body Fat Percentage : {pred[0]:.2f}%")
            elif legacy_payload:
                # If legacy model expects many features, we try to use only Density if possible
                try:
                    # If legacy_payload is dict with model, or direct model
                    if isinstance(legacy_payload, dict) and "model" in legacy_payload:
                        model = legacy_payload["model"]
                        feature_names = legacy_payload.get("feature_names", ["Density"])
                    else:
                        model = legacy_payload
                        # if sklearn exposes feature_names_in_
                        feature_names = getattr(model, "feature_names_in_", ["Density"])
                    # If model expects only Density, predict. Else warn.
                    if len(feature_names) == 1 and feature_names[0].lower().startswith("density"):
                        pred = model.predict(input_df[feature_names])
                        st.success(f"Predicted Body Fat Percentage (legacy model): {pred[0]:.2f}%")
                    else:
                        st.warning(
                            "A legacy model exists but it expects more than Density. "
                            "Either train/save a density-only model (recommended) or provide full feature values."
                        )
                except Exception as e:
                    st.error(f"Could not use legacy model: {e}")
            else:
                st.info(
                    "No density-only model found. Run the training script 'train_density_model.py' to create it "
                    "or save a model at 'models/bodyfat_density_model.pkl'."
                )

        except Exception as e:
            st.error(f"Error computing density/prediction: {e}")


    st.markdown("---")
    st.header("Method B — Legacy Manual Inputs (optional)")
    st.write("If you have the older input form (weight, height, circumference measurements), you can use this section. NOTE: this pathway uses a different density calculation (BMI-like) — not recommended if underwater inputs are available.")

    # --- Legacy UI (your original inputs) ---
    with st.expander("Show legacy input form"):
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
        neck = st.number_input("Neck Circumference (cm)", min_value=10.0, max_value=60.0, value=40.0)
        chest = st.number_input("Chest Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0)
        abdomen = st.number_input("Abdomen Circumference (cm)", min_value=50.0, max_value=200.0, value=90.0)
        hip = st.number_input("Hip Circumference (cm)", min_value=50.0, max_value=200.0, value=100.0)
        thigh = st.number_input("Thigh Circumference (cm)", min_value=30.0, max_value=100.0, value=60.0)
        knee = st.number_input("Knee Circumference (cm)", min_value=20.0, max_value=70.0, value=40.0)
        ankle = st.number_input("Ankle Circumference (cm)", min_value=10.0, max_value=50.0, value=25.0)
        biceps = st.number_input("Biceps Circumference (cm)", min_value=20.0, max_value=70.0, value=35.0)
        forearm = st.number_input("Forearm Circumference (cm)", min_value=15.0, max_value=50.0, value=30.0)
        wrist = st.number_input("Wrist Circumference (cm)", min_value=10.0, max_value=30.0, value=15.0)

        if st.button("Predict (legacy)"):
            # Compute a simple density-like proxy from weight & height (legacy)
            weight_g = weight * 1000
            density_proxy = weight_g / (height ** 3)
            st.markdown(f"**Density proxy (legacy):** `{density_proxy:.6f}`")

            input_features = {
                "Density": density_proxy,
                "Age": age,
                "Weight": weight,
                "Height": height,
                "Neck": neck,
                "Chest": chest,
                "Abdomen": abdomen,
                "Hip": hip,
                "Thigh": thigh,
                "Knee": knee,
                "Ankle": ankle,
                "Biceps": biceps,
                "Forearm": forearm,
                "Wrist": wrist,
            }
            input_df = pd.DataFrame([input_features])

            # Try to use legacy model
            if os.path.exists(legacy_path):
                payload = joblib.load(legacy_path)
                if isinstance(payload, dict) and "model" in payload:
                    model = payload["model"]
                    f_names = payload.get("feature_names", input_df.columns.tolist())
                else:
                    model = payload
                    f_names = getattr(model, "feature_names_in_", input_df.columns.tolist())

                # Check features presence
                missing = [f for f in f_names if f not in input_df.columns]
                if missing:
                    st.error(f"Model expects features that are missing from the legacy input: {missing}")
                else:
                    try:
                        pred = model.predict(input_df[f_names])
                        st.success(f"Predicted Body Fat Percentage: {pred[0]:.2f}%")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            else:
                st.info("Legacy model file not found at 'model/density_model.pkl'.")

# --- EDA and About pages (unchanged loading approach) ---
def eda_page():
    eda_path = "eda.py"
    spec = importlib.util.spec_from_file_location("eda", eda_path)
    if spec and spec.loader:
        eda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eda)
    else:
        st.error(
            "Could not load the EDA page. Please check the file path and try again."
        )

def about_page():
    about_path = "devnev.py"
    spec = importlib.util.spec_from_file_location("devnev", about_path)
    if spec and spec.loader:
        about = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(about)
    else:
        st.error(
            "Could not load the About page. Please check the file path and try again."
        )


# --- Multipage Dictionary ---
PAGES = {
    "Prediction": prediction_page,
    "EDA": eda_page,
    "About DevNev": about_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Render Selected Page ---
page_func = PAGES[selection]
page_func()
