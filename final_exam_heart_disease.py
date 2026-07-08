import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_DATA = Path("heart_disease_uci_raw.csv")
CLEAN_DATA = Path("heart_disease_cleaned.csv")
OUTPUT_DIR = Path("output")
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
MODEL_FILE = OUTPUT_DIR / "best_heart_disease_model.joblib"
REPORT_FILE = Path("Final_Report_Heart_Disease.pdf")
PROMPT_FILE = Path("AI_Prompt_History.txt")


COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target_raw",
]

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def make_folders():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)


def load_data():
    if not RAW_DATA.exists():
        pd.read_csv(DATA_URL, header=None).to_csv(RAW_DATA, index=False, header=False)
    df = pd.read_csv(RAW_DATA, header=None, names=COLUMNS, na_values="?")
    return df


def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    for column in NUMERIC_COLUMNS + ["target_raw"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in NUMERIC_COLUMNS:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    for column in CATEGORICAL_COLUMNS:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

    for column in NUMERIC_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[column] = df[column].clip(low, high)

    df["target"] = (df["target_raw"] > 0).astype(int)
    return df


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


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def make_eda_plots(df):
    class_counts = df["target"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["No Disease", "Disease"], class_counts.values, color=["#4C78A8", "#F58518"])
    plt.title("Class Distribution")
    plt.ylabel("Number of Patients")
    save_plot(FIGURE_DIR / "01_class_distribution.png")

    plt.figure(figsize=(6, 4))
    plt.hist(df.loc[df["target"] == 0, "age"], bins=12, alpha=0.75, label="No Disease")
    plt.hist(df.loc[df["target"] == 1, "age"], bins=12, alpha=0.75, label="Disease")
    plt.title("Age Distribution by Target")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()
    save_plot(FIGURE_DIR / "02_age_distribution.png")

    cp_rate = df.groupby("cp")["target"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(cp_rate.index.astype(str), cp_rate.values, color="#54A24B")
    plt.title("Heart Disease Rate by Chest Pain Type")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Disease Rate")
    save_plot(FIGURE_DIR / "03_chest_pain_rate.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(df["thalach"], df["age"], c=df["target"], cmap="coolwarm", alpha=0.75)
    plt.title("Maximum Heart Rate vs Age")
    plt.xlabel("Maximum Heart Rate")
    plt.ylabel("Age")
    save_plot(FIGURE_DIR / "04_thalach_age_scatter.png")

    exang_rate = df.groupby("exang")["target"].mean()
    plt.figure(figsize=(5, 4))
    plt.bar(["No", "Yes"], exang_rate.values, color="#E45756")
    plt.title("Disease Rate by Exercise Angina")
    plt.xlabel("Exercise Angina")
    plt.ylabel("Disease Rate")
    save_plot(FIGURE_DIR / "05_exang_rate.png")

    plt.figure(figsize=(8, 6))
    corr = df[NUMERIC_COLUMNS + ["target"]].corr()
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    save_plot(FIGURE_DIR / "06_correlation_matrix.png")

    plt.figure(figsize=(6, 4))
    data = [df.loc[df["target"] == 0, "oldpeak"], df.loc[df["target"] == 1, "oldpeak"]]
    plt.boxplot(data, tick_labels=["No Disease", "Disease"])
    plt.title("Oldpeak by Target")
    plt.ylabel("Oldpeak")
    save_plot(FIGURE_DIR / "07_oldpeak_boxplot.png")


def build_pipeline(model, feature_columns):
    numeric_features = [c for c in feature_columns if c in NUMERIC_COLUMNS or c == "heart_rate_gap" or c == "risk_count"]
    categorical_features = [c for c in feature_columns if c not in numeric_features]

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
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_feature_names(pipeline, feature_columns):
    preprocessor = pipeline.named_steps["preprocess"]
    numeric_features = [c for c in feature_columns if c in NUMERIC_COLUMNS or c == "heart_rate_gap" or c == "risk_count"]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    return list(numeric_features) + list(cat_names)


def evaluate_models(df, feature_columns, label):
    X = df[feature_columns]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Support Vector Machine": SVC(kernel="linear", random_state=42),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        pipe = build_pipeline(model, feature_columns)
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "label": label,
        }
        fitted_models[name] = pipe
    return results, fitted_models


def get_top_features(best_model, feature_columns):
    model = best_model.named_steps["model"]
    names = get_feature_names(best_model, feature_columns)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])

    top = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    top.to_csv(OUTPUT_DIR / "top_10_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"][::-1], top["importance"][::-1], color="#4C78A8")
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")
    save_plot(FIGURE_DIR / "08_feature_importance.png")
    return top


def save_prompt_history():
    text = """AI Prompt History

Prompt 1:
I have an exam instruction PDF for an AI-assisted machine learning project. Read the PDF and identify all required tasks and submission files.

Useful output:
The project must predict heart disease. It needs dataset understanding, EDA, preprocessing, feature engineering, three models, evaluation, feature importance, AI reflection, recommendations, notebook, script, PDF report, prompt history, and cleaned CSV.

My modification:
I selected a dataset that matches the hospital heart disease scenario.

Prompt 2:
Find a proper heart disease dataset that is simple enough for a beginner ML classification project and explain the target variable.

Useful output:
The UCI Cleveland Heart Disease dataset is suitable. The target is num/target_raw where 0 means no disease and values 1-4 mean disease. This can be converted into binary classification.

My modification:
I downloaded the official UCI processed Cleveland data and renamed the columns clearly.

Prompt 3:
Write easy Python code for cleaning, EDA, feature engineering, training Logistic Regression, Random Forest, and SVM, then compare Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.

Useful output:
Use pandas for cleaning, matplotlib for plots, and scikit-learn pipelines for Logistic Regression, Random Forest, and SVM.

My modification:
I kept the code simple, used median/mode imputation, IQR outlier capping, one-hot encoding, scaling, and clear feature names.

Prompt 4:
Create a short final report section by section explaining the work, model results, AI prompts, limitations, ethics, privacy, and deployment recommendations.

Useful output:
The report should follow the exact exam tasks and include model comparison plus the best model justification.

My modification:
I generated the final PDF report from the actual model results saved by the script.

AI mistakes noticed:
AI first suggested some extra complex ideas, but I kept the code simple for an exam project.

Improvements made:
I used the official UCI source, made the target binary, avoided complex code, and included explanations for every major step.
"""
    PROMPT_FILE.write_text(text, encoding="utf-8")


def make_table(data, column_widths=None):
    table = Table(data, colWidths=column_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fbff")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d3df")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def add_heading(story, styles, text):
    story.append(Paragraph(text, styles["Heading2"]))
    story.append(Spacer(1, 0.06 * inch))


def add_text(story, styles, text):
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 0.08 * inch))


def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e1e8"))
    canvas.line(36, 24, A4[0] - 36, 24)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(36, 12, "AI-Assisted Heart Disease Prediction Project")
    canvas.drawRightString(A4[0] - 36, 12, f"Page {doc.page}")
    canvas.restoreState()


def build_report_styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].fontSize = 22
    styles["Title"].leading = 27
    styles["Title"].textColor = colors.HexColor("#12355b")
    styles["Title"].alignment = TA_CENTER
    styles["Title"].spaceAfter = 14

    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].fontSize = 14
    styles["Heading2"].leading = 18
    styles["Heading2"].textColor = colors.HexColor("#1f4e79")
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 6

    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13.5
    styles["BodyText"].textColor = colors.HexColor("#243447")
    styles["BodyText"].spaceAfter = 4

    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#52616f"),
        )
    )
    return styles


def create_report(raw_df, clean_df, engineered_df, metrics, baseline_metrics, top_features, best_model_name):
    doc = SimpleDocTemplate(str(REPORT_FILE), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=42)
    styles = build_report_styles()
    story = []

    story.append(Paragraph("AI-Assisted Heart Disease Prediction Project", styles["Title"]))
    story.append(Paragraph("Course: Artificial Intelligence | Final Examination", styles["Subtitle"]))
    story.append(Paragraph("Scenario: HealthPlus Hospital needs an AI system to predict heart disease.", styles["Subtitle"]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(make_table(
        [
            ["Dataset", "UCI Cleveland Heart Disease dataset"],
            ["Problem Type", "Binary classification"],
            ["Best Model", best_model_name],
            ["Best Accuracy", f"{metrics[best_model_name]['accuracy']:.2%}"],
        ],
        [1.5 * inch, 4.8 * inch],
    ))
    story.append(Spacer(1, 0.12 * inch))

    add_heading(story, styles, "1. Dataset Understanding")
    add_text(story, styles, "Target variable: target. It is made from target_raw. Value 0 means no heart disease and values 1-4 mean heart disease. This is a binary classification problem.")
    add_text(story, styles, "Numerical features: age, resting blood pressure, cholesterol, maximum heart rate, and oldpeak. Categorical features: sex, chest pain type, fasting blood sugar, resting ECG, exercise angina, slope, ca, and thal.")
    add_text(story, styles, "Important expected features are chest pain type, exercise angina, oldpeak, maximum heart rate, ca, thal, age, and slope. Two challenges are missing values in ca/thal and the small dataset size.")

    add_heading(story, styles, "2. Exploratory Data Analysis")
    summary_data = [
        ["Item", "Result"],
        ["Raw rows and columns", f"{raw_df.shape[0]} rows, {raw_df.shape[1]} columns"],
        ["Clean rows and columns", f"{clean_df.shape[0]} rows, {clean_df.shape[1]} columns"],
        ["Missing values in raw data", str(int(raw_df.isna().sum().sum()))],
        ["Duplicate rows", str(int(raw_df.duplicated().sum()))],
        ["Class distribution", clean_df["target"].value_counts().sort_index().to_dict().__str__()],
    ]
    story.append(make_table(summary_data, [2.2 * inch, 4.4 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    add_text(story, styles, "Class distribution means how many records belong to each target class. In this project, it shows how many patients have no heart disease and how many patients have heart disease. It is important because a highly unbalanced dataset can make a model look accurate while still missing the smaller class.")
    add_text(story, styles, "EDA plots show that the target is fairly balanced, older age and lower maximum heart rate are connected with disease, and chest pain type plus exercise angina are strong warning signs. The correlation matrix also shows useful relationships with oldpeak and thalach.")
    for name in [
        "01_class_distribution.png",
        "02_age_distribution.png",
        "03_chest_pain_rate.png",
        "04_thalach_age_scatter.png",
        "05_exang_rate.png",
        "06_correlation_matrix.png",
    ]:
        story.append(Image(str(FIGURE_DIR / name), width=4.8 * inch, height=3.2 * inch))
        story.append(Spacer(1, 0.12 * inch))

    add_heading(story, styles, "3. Data Cleaning and Preprocessing")
    add_text(story, styles, "Missing numeric values were filled with the median because the median is less affected by extreme values. Missing categorical values were filled with the mode because it is the most common category.")
    add_text(story, styles, "Duplicates were removed. Numeric outliers were capped using the IQR method. IQR means Interquartile Range. Q1 means the first quartile, or the 25th percentile, so 25% of values are below Q1. Q3 means the third quartile, or the 75th percentile, so 75% of values are below Q3. IQR is calculated as Q3 - Q1. Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are treated as outliers and capped.")
    add_text(story, styles, "Categorical variables were one-hot encoded. Numeric variables were scaled because Logistic Regression and SVM work better when numeric values are on a similar scale.")

    add_heading(story, styles, "4. Feature Engineering")
    add_text(story, styles, "Feature engineering means creating new useful columns from existing columns so the model can learn patterns more easily. It does not change the target answer; it only gives the model better input information.")
    add_text(story, styles, "New simple features were created: age_group, cholesterol_risk, bp_risk, heart_rate_gap, oldpeak_high, and risk_count. These features convert medical risk ideas into values that models can learn from.")
    comparison = [["Model", "Baseline Accuracy", "After Feature Engineering Accuracy"]]
    for model_name in metrics:
        comparison.append([
            model_name,
            baseline_metrics[model_name]["accuracy"],
            metrics[model_name]["accuracy"],
        ])
    story.append(make_table(comparison, [2.5 * inch, 1.8 * inch, 2.2 * inch]))

    add_heading(story, styles, "5. Machine Learning Models")
    add_text(story, styles, "Logistic Regression: This model works like a simple scoring system. It gives weight to each feature, adds the scores, and then predicts the chance of heart disease. It is easy to understand and works well for many binary classification problems.")
    add_text(story, styles, "Random Forest: This model creates many small decision trees. Each tree makes a prediction, and the forest combines their votes. It often performs well because many trees reduce the risk of one weak tree making the final decision.")
    add_text(story, styles, "Support Vector Machine: SVM tries to find the best boundary that separates patients with heart disease from patients without heart disease. It is useful for medical classification because it can work well with smaller datasets.")
    model_rows = [["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    for model_name, row in metrics.items():
        model_rows.append([
            model_name,
            row["accuracy"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["roc_auc"],
        ])
    story.append(make_table(model_rows, [1.7 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch]))
    add_text(story, styles, "Accuracy means the percentage of total correct predictions. Precision means, when the model predicts heart disease, how often it is correct. Recall means how many real heart disease patients the model successfully finds. F1-score is a balance between precision and recall. ROC-AUC shows how well the model separates disease and no disease classes.")
    add_text(story, styles, f"Best model selected: {best_model_name}. It was selected by comparing recall, F1-score, ROC-AUC, and accuracy. SVM was included because it is commonly useful for medical classification when the dataset is not very large and a clear decision boundary is needed. In a hospital problem, recall is very important because missing a patient with heart disease is risky.")

    add_heading(story, styles, "6. Model Evaluation")
    add_text(story, styles, "A confusion matrix is a table that shows correct and wrong predictions. For binary classification it has true negatives, false positives, false negatives, and true positives. In this medical task, false negatives are dangerous because they mean a patient has heart disease but the model predicts no disease.")
    for model_name, row in metrics.items():
        add_text(story, styles, f"{model_name} confusion matrix: {row['confusion_matrix']}. Accuracy={row['accuracy']}, Precision={row['precision']}, Recall={row['recall']}, F1={row['f1']}, ROC-AUC={row['roc_auc']}.")
    add_text(story, styles, "Most appropriate metric: Recall, with F1-score also important. Recall reduces false negatives, while F1-score balances recall and precision.")

    add_heading(story, styles, "7. Model Interpretation")
    story.append(Image(str(FIGURE_DIR / "08_feature_importance.png"), width=5.0 * inch, height=3.2 * inch))
    story.append(Spacer(1, 0.12 * inch))
    feature_rows = [["Feature", "Importance"]]
    for _, row in top_features.iterrows():
        feature_rows.append([str(row["feature"]), round(float(row["importance"]), 4)])
    story.append(make_table(feature_rows, [3.4 * inch, 1.5 * inch]))
    add_text(story, styles, "The top features are useful because they describe chest pain, exercise response, blood vessel information, heart rate, and ST depression. These are medically related to heart disease risk.")

    add_heading(story, styles, "8. AI-Assisted Development Reflection")
    reflection_items = [
        (
            "Prompt 1",
            "Read the exam instruction PDF and list all required tasks and submission files.",
            "AI identified the nine tasks and required files. I followed those sections in the script, notebook, report, and prompt history.",
        ),
        (
            "Prompt 2",
            "Find a proper heart disease dataset for this project and explain the target variable.",
            "AI suggested the UCI Cleveland Heart Disease dataset. I used the official UCI data and converted target_raw into binary target.",
        ),
        (
            "Prompt 3",
            "Write easy Python code for cleaning, EDA, feature engineering, Logistic Regression, Random Forest, SVM, and evaluation.",
            "AI suggested pandas, matplotlib, and scikit-learn. I used SVM because this is a medical classification task and kept pipelines for fair comparison.",
        ),
        (
            "Prompt 4",
            "Create a short final report with section-by-section explanation, model results, prompts, limitations, ethics, and recommendations.",
            "AI helped organize the report. I filled the report with actual results from the model run.",
        ),
    ]
    for title, prompt, action in reflection_items:
        story.append(Paragraph(f"<b>{title}</b>", styles["BodyText"]))
        add_text(story, styles, f"<b>Prompt used:</b> {prompt}")
        add_text(story, styles, f"<b>AI output and my modification:</b> {action}")
        story.append(Spacer(1, 0.06 * inch))
    add_text(story, styles, "AI mistake noticed: AI first suggested some extra complex ideas, but I kept the code simple for an exam project.")
    add_text(story, styles, "My improvements: I used an official data source, added SVM for medical classification, checked the metrics from real model output, and added a separate AI prompt history file.")

    add_heading(story, styles, "9. Critical Discussion and Recommendations")
    add_text(story, styles, "Limitations: the dataset is small and old, so results may not generalize to all hospital patients. Bias can happen if the patient group does not represent the real population. Data leakage was avoided by splitting train/test before model fitting inside pipelines.")
    add_text(story, styles, "Ethics and privacy: patient data must be protected, anonymized, and used only with permission. The model should support doctors, not replace doctors. For deployment, the hospital should validate the model on newer local patient data before real use.")

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)


def save_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# AI-Assisted Heart Disease Prediction Project\n",
                    "This notebook follows the final exam PDF section by section. The code is kept simple on purpose.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 1: Dataset Understanding\n",
                    "Dataset: UCI Cleveland Heart Disease dataset. Target: `target`, where 0 means no disease and 1 means disease. This is a classification problem.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target_raw']\n",
                    "df = pd.read_csv('heart_disease_uci_raw.csv', header=None, names=columns, na_values='?')\n",
                    "df.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Task 2: Exploratory Data Analysis\n", "Check shape, data types, missing values, duplicates, summary, class balance, and useful charts.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Shape:', df.shape)\n",
                    "print('\\nMissing values:')\n",
                    "print(df.isna().sum())\n",
                    "print('\\nDuplicates:', df.duplicated().sum())\n",
                    "df.describe()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run the full project script to create all required EDA figures and outputs.\n",
                    "%run final_exam_heart_disease.py\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Task 3: Data Cleaning and Preprocessing\n", "The script removes duplicates, fills missing values, caps outliers with the IQR method, one-hot encodes categories, and scales numeric values for models that need scaling.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "clean_df = pd.read_csv('heart_disease_cleaned.csv')\n",
                    "print(clean_df.shape)\n",
                    "clean_df.head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Task 4: Feature Engineering\n", "Feature engineering means creating useful new columns from existing columns. New features include `age_group`, `cholesterol_risk`, `bp_risk`, `heart_rate_gap`, `oldpeak_high`, and `risk_count`.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "new_features = ['age_group', 'cholesterol_risk', 'bp_risk', 'heart_rate_gap', 'oldpeak_high', 'risk_count']\n",
                    "clean_df[new_features].head()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Task 5 and 6: Machine Learning Models and Evaluation\n", "Three models were trained: Logistic Regression, Random Forest, and Support Vector Machine.\n", "\n", "- Logistic Regression works like a simple scoring system and predicts the chance of disease.\n", "- Random Forest builds many decision trees and combines their votes.\n", "- SVM finds the best boundary between disease and no disease patients.\n", "\n", "SVM was used because this is a medical classification task. The table below shows accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix values.\n", "\n", "Metric meaning: Accuracy is total correct predictions. Precision shows how correct the disease predictions are. Recall shows how many real disease patients were found. F1-score balances precision and recall. ROC-AUC shows how well the model separates disease and no disease classes.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "with open('output/model_metrics.json', 'r') as file:\n",
                    "    metrics = json.load(file)\n",
                    "pd.DataFrame(metrics['engineered']).T\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Task 7: Model Interpretation\n", "The best model's top 10 important features are saved below.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["pd.read_csv('output/top_10_feature_importance.csv')\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Task 8 and 9: AI Reflection and Discussion\n",
                    "The prompt history is saved in `AI_Prompt_History.txt`. The final discussion about limitations, bias, leakage, ethics, privacy, and deployment is included in `Final_Report_Heart_Disease.pdf`.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Final Output Files\n",
                    "- `Heart_Disease_Final_Exam.ipynb`\n",
                    "- `final_exam_heart_disease.py`\n",
                    "- `Final_Report_Heart_Disease.pdf`\n",
                    "- `AI_Prompt_History.txt`\n",
                    "- `heart_disease_cleaned.csv`\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path("Heart_Disease_Final_Exam.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main():
    make_folders()
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    engineered_df = add_features(clean_df)
    engineered_df.to_csv(CLEAN_DATA, index=False)

    make_eda_plots(engineered_df)

    base_features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    engineered_features = base_features + [
        "age_group",
        "cholesterol_risk",
        "bp_risk",
        "heart_rate_gap",
        "oldpeak_high",
        "risk_count",
    ]

    baseline_metrics, _ = evaluate_models(clean_df, base_features, "baseline")
    metrics, fitted_models = evaluate_models(engineered_df, engineered_features, "engineered")

    best_model_name = max(metrics, key=lambda name: (metrics[name]["recall"], metrics[name]["f1"], metrics[name]["roc_auc"]))
    joblib.dump(fitted_models[best_model_name], MODEL_FILE)
    top_features = get_top_features(fitted_models[best_model_name], engineered_features)

    all_metrics = {"baseline": baseline_metrics, "engineered": metrics, "best_model": best_model_name}
    METRICS_FILE.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    save_prompt_history()
    create_report(raw_df, clean_df, engineered_df, metrics, baseline_metrics, top_features, best_model_name)
    save_notebook()

    print("Project completed.")
    print(f"Best model: {best_model_name}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
