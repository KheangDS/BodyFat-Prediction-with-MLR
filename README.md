# Body Fat Prediction with Multiple Linear Regression (MLR)

A Streamlit web app that predicts **body fat percentage** from body measurements using a **Multiple Linear Regression** model trained on the classic Body Fat dataset.

## Demo (Streamlit)
Run locally and open the Streamlit UI in your browser.

## Project structure
- `app.py` — Streamlit multipage app (Prediction / EDA / About)
- `eda.py` — EDA page (interactive charts)
- `devnev.py` — About page (team section)
- `notebook/` — model training & testing notebooks
- `images/` — images used by the About page
- `requirements.txt` — Python dependencies

> Note: `app.py` expects a trained model at `models/bodyfat_model_02.pkl` and `eda.py` expects a dataset at `data/bodyfat.csv`.

## Features
- Predict body fat % from anthropometric inputs (age, weight, height, and circumferences)
- Interactive EDA: preview, summary statistics, histograms, correlation heatmap, and scatter plots
- Simple multi-page navigation in Streamlit

## Dataset
Source: Kaggle — Body Fat Prediction Dataset.

## Installation
```bash
# (optional) create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Usage
1. Open the **Prediction** page.
2. Enter the measurements.
3. Click **Predict Body Fat Percentage**.

## Notes / Troubleshooting
- If you see an error about missing files, add:
  - `models/bodyfat_model_02.pkl`
  - `data/bodyfat.csv`
- Ensure your model was trained with the same feature names used in `app.py`:
  `Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist`.

## License
Add a license for your preferred usage (e.g., MIT).