import json

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLEAN_DATA_PATH = "heart_disease_cleaned.csv"
METRICS_PATH = "output/model_metrics.json"

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ENGINEERED_COLUMNS = [
    "age_group",
    "cholesterol_risk",
    "bp_risk",
    "heart_rate_gap",
    "oldpeak_high",
    "risk_count",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ENGINEERED_COLUMNS


SEX_OPTIONS = {"Female": 0, "Male": 1}
YES_NO_OPTIONS = {"No": 0, "Yes": 1}
CHEST_PAIN_OPTIONS = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
RESTING_ECG_OPTIONS = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
SLOPE_OPTIONS = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3,
}
VESSEL_OPTIONS = {
    "0 vessels": 0,
    "1 vessel": 1,
    "2 vessels": 2,
    "3 vessels": 3,
}
THAL_OPTIONS = {
    "Normal": 3,
    "Fixed defect": 6,
    "Reversible defect": 7,
}


def add_features(df):
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 45, 55, 65, 120],
        labels=["young", "middle", "senior", "older"],
    )
    df["cholesterol_risk"] = (df["chol"] >= 240).astype(int)
    df["bp_risk"] = (df["trestbps"] >= 140).astype(int)
    df["heart_rate_gap"] = 220 - df["age"] - df["thalach"]
    df["oldpeak_high"] = (df["oldpeak"] >= 2).astype(int)
    df["risk_count"] = (
        df["cholesterol_risk"] + df["bp_risk"] + df["oldpeak_high"] + df["exang"]
    )
    return df


def build_model():
    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "heart_rate_gap",
        "risk_count",
    ]
    categorical_features = [column for column in FEATURE_COLUMNS if column not in numeric_features]

    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_steps, numeric_features),
            ("cat", categorical_steps, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )


@st.cache_resource
def load_model():
    df = pd.read_csv(CLEAN_DATA_PATH)
    model = build_model()
    model.fit(df[FEATURE_COLUMNS], df["target"])
    return model


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def make_input_row(values):
    return pd.DataFrame([values])


def predict_rows(model, model_df):
    model_df = add_features(model_df)
    predictions = model.predict(model_df)

    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = model.predict_proba(model_df)[:, 1]
    else:
        probabilities = [None] * len(model_df)

    return predictions, probabilities


def readable_to_model_data(readable_df):
    rows = []
    for _, row in readable_df.iterrows():
        rows.append(
            {
                "age": int(row["Age"]),
                "sex": SEX_OPTIONS[row["Sex"]],
                "cp": CHEST_PAIN_OPTIONS[row["Chest Pain Type"]],
                "trestbps": int(row["Resting Blood Pressure"]),
                "chol": int(row["Cholesterol"]),
                "fbs": YES_NO_OPTIONS[row["Fasting Blood Sugar Over 120 mg/dl"]],
                "restecg": RESTING_ECG_OPTIONS[row["Resting ECG Result"]],
                "thalach": int(row["Maximum Heart Rate"]),
                "exang": YES_NO_OPTIONS[row["Exercise Induced Angina"]],
                "oldpeak": float(row["Oldpeak"]),
                "slope": SLOPE_OPTIONS[row["ST Slope"]],
                "ca": VESSEL_OPTIONS[row["Major Vessels"]],
                "thal": THAL_OPTIONS[row["Thalassemia Result"]],
            }
        )
    return pd.DataFrame(rows)


def default_patient_table():
    return pd.DataFrame(
        [
            {
                "Age": 55,
                "Sex": "Male",
                "Chest Pain Type": "Asymptomatic",
                "Resting Blood Pressure": 130,
                "Cholesterol": 240,
                "Fasting Blood Sugar Over 120 mg/dl": "No",
                "Resting ECG Result": "Normal",
                "Maximum Heart Rate": 150,
                "Exercise Induced Angina": "No",
                "Oldpeak": 1.0,
                "ST Slope": "Flat",
                "Major Vessels": "0 vessels",
                "Thalassemia Result": "Normal",
            }
        ]
    )


st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #54606c;
        font-size: 16px;
        margin-bottom: 18px;
    }
    .result-box {
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 22px;
        background: #f8fafc;
    }
    .risk-high {
        border-left: 6px solid #d92d20;
    }
    .risk-low {
        border-left: 6px solid #07803a;
    }
    .small-note {
        color: #667085;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()
metrics = load_metrics()
best_model_name = metrics["best_model"]
best_metrics = metrics["engineered"][best_model_name]

st.markdown('<div class="main-title">Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter patient data at the top, then view the prediction below.</div>',
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Best Model", best_model_name)
metric_col2.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
metric_col3.metric("Recall", f"{best_metrics['recall']:.2%}")
metric_col4.metric("ROC-AUC", f"{best_metrics['roc_auc']:.2%}")

with st.sidebar:
    st.header("Project Info")
    st.write("AI final exam heart disease prediction app.")
    st.write("Best model:", best_model_name)
    st.write("Accuracy:", f"{best_metrics['accuracy']:.2%}")
    st.caption("Educational project only. Not for real diagnosis.")

st.divider()

st.subheader("Single Patient Input")
st.write("Fill this form first. All choices use readable medical labels.")

with st.form("patient_form"):
    basic_col, heart_col, medical_col = st.columns(3)

    with basic_col:
        st.markdown("**Basic Info**")
        age = st.slider("Age", min_value=20, max_value=100, value=55)
        sex_label = st.radio("Sex", list(SEX_OPTIONS.keys()), horizontal=True)
        fbs_label = st.radio("Fasting Blood Sugar Over 120 mg/dl", list(YES_NO_OPTIONS.keys()), horizontal=True)

    with heart_col:
        st.markdown("**Heart Test Values**")
        chest_pain_label = st.selectbox("Chest Pain Type", list(CHEST_PAIN_OPTIONS.keys()))
        trestbps = st.slider("Resting Blood Pressure", 80, 220, 130)
        chol = st.slider("Cholesterol", 100, 600, 240)
        thalach = st.slider("Maximum Heart Rate", 60, 230, 150)

    with medical_col:
        st.markdown("**Other Medical Values**")
        resting_ecg_label = st.selectbox("Resting ECG Result", list(RESTING_ECG_OPTIONS.keys()))
        exang_label = st.radio("Exercise Induced Angina", list(YES_NO_OPTIONS.keys()), horizontal=True)
        oldpeak = st.slider("Oldpeak", 0.0, 7.0, 1.0, 0.1)
        slope_label = st.selectbox("ST Slope", list(SLOPE_OPTIONS.keys()))
        vessel_label = st.selectbox("Major Vessels Seen in Fluoroscopy", list(VESSEL_OPTIONS.keys()))
        thal_label = st.selectbox("Thalassemia Result", list(THAL_OPTIONS.keys()))

    predict_clicked = st.form_submit_button("Predict This Patient", width="stretch")

input_values = {
    "age": age,
    "sex": SEX_OPTIONS[sex_label],
    "cp": CHEST_PAIN_OPTIONS[chest_pain_label],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": YES_NO_OPTIONS[fbs_label],
    "restecg": RESTING_ECG_OPTIONS[resting_ecg_label],
    "thalach": thalach,
    "exang": YES_NO_OPTIONS[exang_label],
    "oldpeak": oldpeak,
    "slope": SLOPE_OPTIONS[slope_label],
    "ca": VESSEL_OPTIONS[vessel_label],
    "thal": THAL_OPTIONS[thal_label],
}

patient_df = make_input_row(input_values)
patient_with_features = add_features(patient_df)

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Prediction Result")

    if predict_clicked:
        prediction = model.predict(patient_with_features)[0]

        if hasattr(model.named_steps["model"], "predict_proba"):
            probability = model.predict_proba(patient_with_features)[0][1]
            probability_text = f"{probability:.2%}"
        else:
            probability_text = "Not available"

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box risk-high">
                    <h3>Heart disease risk detected</h3>
                    <p><b>Prediction probability:</b> {probability_text}</p>
                    <p>The model predicts that this patient may have heart disease risk.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box risk-low">
                    <h3>No heart disease risk detected</h3>
                    <p><b>Prediction probability:</b> {probability_text}</p>
                    <p>The model predicts that this patient is less likely to have heart disease.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Complete the input form above and click Predict This Patient.")

    st.caption("This app is only for an educational exam project. It is not a real medical diagnosis tool.")

with right:
    st.subheader("Patient Summary")
    readable_summary = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex_label,
                "Chest Pain": chest_pain_label,
                "Resting BP": trestbps,
                "Cholesterol": chol,
                "Max Heart Rate": thalach,
                "Exercise Angina": exang_label,
                "ST Slope": slope_label,
                "Vessels": vessel_label,
                "Thalassemia": thal_label,
            }
        ]
    )
    st.dataframe(readable_summary, width="stretch", hide_index=True)

    with st.expander("Technical values sent to the model"):
        display_df = patient_df.rename(
            columns={
                "sex": "sex_code",
                "cp": "chest_pain_code",
                "trestbps": "resting_bp",
                "fbs": "fasting_sugar_code",
                "restecg": "resting_ecg_code",
                "thalach": "max_heart_rate",
                "exang": "exercise_angina",
                "slope": "slope_code",
                "ca": "vessel_code",
                "thal": "thal_code",
            }
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

st.divider()

st.subheader("Multiple Patient Data Input")
st.write("Add or edit rows in this table, then predict all rows together.")

editable_df = st.data_editor(
    default_patient_table(),
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "Age": st.column_config.NumberColumn("Age", min_value=20, max_value=100, step=1),
        "Sex": st.column_config.SelectboxColumn("Sex", options=list(SEX_OPTIONS.keys())),
        "Chest Pain Type": st.column_config.SelectboxColumn(
            "Chest Pain Type", options=list(CHEST_PAIN_OPTIONS.keys())
        ),
        "Resting Blood Pressure": st.column_config.NumberColumn(
            "Resting Blood Pressure", min_value=80, max_value=220, step=1
        ),
        "Cholesterol": st.column_config.NumberColumn("Cholesterol", min_value=100, max_value=600, step=1),
        "Fasting Blood Sugar Over 120 mg/dl": st.column_config.SelectboxColumn(
            "Fasting Blood Sugar Over 120 mg/dl", options=list(YES_NO_OPTIONS.keys())
        ),
        "Resting ECG Result": st.column_config.SelectboxColumn(
            "Resting ECG Result", options=list(RESTING_ECG_OPTIONS.keys())
        ),
        "Maximum Heart Rate": st.column_config.NumberColumn(
            "Maximum Heart Rate", min_value=60, max_value=230, step=1
        ),
        "Exercise Induced Angina": st.column_config.SelectboxColumn(
            "Exercise Induced Angina", options=list(YES_NO_OPTIONS.keys())
        ),
        "Oldpeak": st.column_config.NumberColumn("Oldpeak", min_value=0.0, max_value=7.0, step=0.1),
        "ST Slope": st.column_config.SelectboxColumn("ST Slope", options=list(SLOPE_OPTIONS.keys())),
        "Major Vessels": st.column_config.SelectboxColumn(
            "Major Vessels", options=list(VESSEL_OPTIONS.keys())
        ),
        "Thalassemia Result": st.column_config.SelectboxColumn(
            "Thalassemia Result", options=list(THAL_OPTIONS.keys())
        ),
    },
)

if st.button("Predict All Entered Patients", width="stretch"):
    try:
        batch_model_df = readable_to_model_data(editable_df)
        batch_predictions, batch_probabilities = predict_rows(model, batch_model_df)

        result_df = editable_df.copy()
        result_df["Prediction"] = [
            "Heart disease risk detected" if value == 1 else "No heart disease risk detected"
            for value in batch_predictions
        ]
        result_df["Disease Probability"] = [
            "Not available" if value is None else f"{value:.2%}" for value in batch_probabilities
        ]

        st.success("Batch prediction completed.")
        st.dataframe(result_df, width="stretch", hide_index=True)
        st.download_button(
            "Download Prediction Results",
            result_df.to_csv(index=False).encode("utf-8"),
            "heart_disease_predictions.csv",
            "text/csv",
            width="stretch",
        )
    except Exception as error:
        st.error(f"Please complete all patient values before predicting. Details: {error}")

st.divider()

with st.expander("Model explanations and comparison"):
    st.write("Logistic Regression: uses feature weights to calculate disease chance.")
    st.write("Random Forest: combines many decision trees and uses their votes.")
    st.write("SVM: finds the best boundary between disease and no disease patients.")

    model_rows = []
    for model_name, row in metrics["engineered"].items():
        model_rows.append(
            {
                "Model": model_name,
                "Accuracy": row["accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1-score": row["f1"],
                "ROC-AUC": row["roc_auc"],
            }
        )
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

st.markdown(
    '<p class="small-note">Tip: In medical prediction, recall is very important because false negatives can be dangerous.</p>',
    unsafe_allow_html=True,
)
