# Heart Disease Prediction Streamlit App

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Required files

Keep these files in the project folder:

- `streamlit_app.py`
- `heart_disease_cleaned.csv`
- `output/model_metrics.json`

The app checks for missing files and shows a clear error instead of failing silently.
