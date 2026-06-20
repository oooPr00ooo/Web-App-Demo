from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "artifacts" / "models"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"

MODEL_ORDER = [
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "VotingClassifier",
]

MODEL_LABELS = {
    "DecisionTreeClassifier": "Decision Tree",
    "KNeighborsClassifier": "K-NN",
    "GaussianNB": "Naive Bayes",
    "RandomForestClassifier": "Random Forest",
    "AdaBoostClassifier": "AdaBoost",
    "GradientBoostingClassifier": "Gradient Boosting",
    "XGBClassifier": "XGBoost",
    "VotingClassifier": "Ensemble (Soft Voting)",
}

FEATURE_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

EXAMPLE_PATIENTS = {
    "Example 1 (No Heart Disease)": {
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
        "actual_target": 0,
    },
    "Example 2 (Heart Disease)": {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        "actual_target": 1,
    },
}


def scale_continuous(value: float, minimum: float, maximum: float) -> float:
    clipped = min(max(value, minimum), maximum)
    return (clipped - minimum) / (maximum - minimum)


def encode_patient_input(patient_input: dict[str, float | int]) -> pd.DataFrame:
    encoded = {
        "age": scale_continuous(float(patient_input["age"]), 29, 77),
        "trestbps": scale_continuous(float(patient_input["trestbps"]), 94, 200),
        "chol": scale_continuous(float(patient_input["chol"]), 126, 564),
        "thalach": scale_continuous(float(patient_input["thalach"]), 71, 202),
        "oldpeak": scale_continuous(float(patient_input["oldpeak"]), 0.0, 6.2),
        "sex": float(patient_input["sex"]),
        "cp": (int(patient_input["cp"]) - 1) / 3,
        "fbs": float(patient_input["fbs"]),
        "restecg": int(patient_input["restecg"]) / 2,
        "exang": float(patient_input["exang"]),
        "slope": (int(patient_input["slope"]) - 1) / 2,
        "ca": int(patient_input["ca"]) / 3,
        "thal": {3: 0.0, 6: 0.5, 7: 1.0}[int(patient_input["thal"])],
    }
    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)


@st.cache_resource
def load_models(_version: tuple[float, ...]) -> dict[str, object]:
    return {
        model_name: joblib.load(MODELS_DIR / f"{model_name}.joblib")
        for model_name in MODEL_ORDER
    }


@st.cache_data
def load_reports(_version: tuple[float, ...]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for model_name in MODEL_ORDER:
        report_path = REPORTS_DIR / f"{model_name}_metrics.json"
        if report_path.exists():
            reports[model_name] = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            reports[model_name] = {}
    return reports


def get_artifact_version() -> tuple[float, ...]:
    stamps: list[float] = []
    for model_name in MODEL_ORDER:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        report_path = REPORTS_DIR / f"{model_name}_metrics.json"
        stamps.append(model_path.stat().st_mtime if model_path.exists() else 0.0)
        stamps.append(report_path.stat().st_mtime if report_path.exists() else 0.0)
    return tuple(stamps)


def predict_all_models(
    patient_df: pd.DataFrame,
    patient_input: dict[str, float | int],
    actual_target: int | None,
) -> pd.DataFrame:
    artifact_version = get_artifact_version()
    models = load_models(artifact_version)
    reports = load_reports(artifact_version)
    rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        model = models[model_name]
        probability = float(model.predict_proba(patient_df)[0][1])
        predicted_target = int(probability >= 0.5)
        confidence = probability if predicted_target == 1 else 1 - probability

        predicted_label = "Heart Disease" if predicted_target == 1 else "No Heart Disease"
        correctness = None if actual_target is None else predicted_target == actual_target

        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "prediction": predicted_label,
                "prediction_target": predicted_target,
                "confidence": confidence,
                "heart_disease_probability": probability,
                "test_accuracy": reports[model_name].get("test_metrics", {}).get("accuracy"),
                "test_f1": reports[model_name].get("test_metrics", {}).get("f1"),
                "is_correct": correctness,
                "correctness_label": "Correct" if correctness is True else "Wrong",
            }
        )

    return pd.DataFrame(rows)


def build_prediction_signature(
    artifact_version: tuple[float, ...],
    patient_input: dict[str, float | int],
    actual_target: int | None,
) -> tuple:
    ordered_values = tuple(patient_input[column] for column in FEATURE_COLUMNS)
    return artifact_version + ordered_values + (actual_target,)


def render_chart(results_df: pd.DataFrame) -> None:
    chart_df = results_df.copy()
    chart_df["confidence_percent"] = (chart_df["confidence"] * 100).round(1)
    chart_df["bar_text"] = chart_df["confidence_percent"].astype(str) + "%"
    chart_df["inside_text"] = chart_df["prediction"].map(
        {
            "No Heart Disease": "✅ No Heart Disease",
            "Heart Disease": "🫀 Heart Disease",
        }
    )

    label_order = [MODEL_LABELS[m] for m in MODEL_ORDER]
    fig = px.bar(
        chart_df,
        x="model_label",
        y="confidence",
        color="correctness_label",
        text="bar_text",
        category_orders={"model_label": label_order},
        color_discrete_map={
            "Correct": "#2e8540",
            "Wrong": "#c92f4a",
        },
        hover_data={
            "model": True,
            "model_label": False,
            "prediction": True,
            "correctness_label": True,
            "confidence_percent": True,
            "heart_disease_probability": ":.3f",
            "test_accuracy": ":.3f",
            "test_f1": ":.3f",
            "confidence": False,
            "inside_text": False,
        },
    )
    fig.update_traces(
        textposition="outside",
        marker_line_color="#111111",
        marker_line_width=1.5,
        cliponaxis=False,
    )
    for row in chart_df.itertuples(index=False):
        fig.add_annotation(
            x=row.model_label,
            y=max(row.confidence * 0.52, 0.12),
            text=row.inside_text,
            showarrow=False,
            textangle=90,
            font=dict(size=11, color="white"),
        )
    fig.update_layout(
        title="Model Predictions",
        xaxis_title="Model",
        yaxis_title="Prediction Confidence",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=14, color="#111111"),
        margin=dict(l=10, r=10, t=52, b=10),
    )
    fig.update_xaxes(tickangle=-28)
    st.plotly_chart(fig, use_container_width=True)


def render_model_name_list() -> None:
    st.markdown(
        """
        <div class="model-name-list">
            <div>DecisionTreeClassifier</div>
            <div>AdaBoostClassifier</div>
            <div>RandomForestClassifier</div>
            <div>GradientBoostingClassifier</div>
            <div>XGBClassifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        /* Make the Streamlit page use the same large, left-aligned canvas as the target image */
        .block-container {
            padding-top: 0.9rem;
            padding-left: 1.75rem;
            padding-right: 1.75rem;
            padding-bottom: 1.5rem;
            max-width: none;
        }
        .stApp {
            background: #ffffff;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .hero-title {
            color: #ff1200;
            font-size: 4.15rem;
            font-weight: 600;
            line-height: 0.95;
            letter-spacing: -0.035em;
            margin: 0 0 1.45rem 0;
        }

        /* Stable styling via st.container(key=...). This avoids :has() selecting a wrong ancestor. */
        .st-key-left_panel_shell {
            background: linear-gradient(180deg, #292929 0%, #252525 100%);
            border: 2px solid #111111;
            border-radius: 0;
            padding: 0 0 0.9rem 0;
            overflow: hidden;
        }
        .st-key-left_panel_shell > div {
            gap: 0 !important;
        }
        .left-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #f5f5f5;
            font-size: 1.15rem;
            font-weight: 800;
            padding: 0.72rem 1rem;
            border-bottom: 1px solid #414141;
            background: #292929;
        }
        .left-panel-header .caret {
            font-size: 0.92rem;
            color: #f5f5f5;
        }
        .st-key-form_row_1,
        .st-key-form_row_2,
        .st-key-form_row_3,
        .st-key-form_row_4 {
            border: 1px solid #444444;
            border-radius: 8px;
            margin: 0.72rem 0.72rem 0 0.72rem;
            padding: 0.72rem 0.85rem 0.55rem 0.85rem;
            background: #292929;
        }
        .st-key-form_action_row {
            margin: 0.72rem 0.72rem 0 0.72rem;
        }
        .st-key-example_card,
        .st-key-predict_card {
            border: 1px solid #444444;
            border-radius: 8px;
            background: #292929;
            padding: 0.72rem 0.85rem 0.62rem 0.85rem;
            min-height: 6.6rem;
        }

        .st-key-right_panel {
            background: #ffffff;
            border: 4px solid #202020;
            border-radius: 0;
            padding: 0 0.35rem 0.8rem 0rem;
            margin-top: 5.65rem;
        }
        .chart-tag {
            display: inline-block;
            background: #2b2b2b;
            color: #ffffff;
            border-radius: 0 0 8px 0;
            padding: 0.48rem 0.85rem;
            font-weight: 800;
            font-size: 1.08rem;
            margin: 0 0 0.1rem 0;
        }
        .st-key-right_panel .stPlotlyChart {
            padding: 0 0.2rem;
        }
        .model-name-list {
            font-size: 1.9rem;
            line-height: 1.05;
            color: #111111;
            margin-top: 0.8rem;
            margin-left: 2.7rem;
            font-weight: 500;
        }

        /* Widgets */
        div[data-testid="stWidgetLabel"] label,
        .stSelectbox label,
        .stNumberInput label {
            color: #f2f2f2 !important;
            font-weight: 800 !important;
            opacity: 1 !important;
            display: block !important;
            visibility: visible !important;
            font-size: 1.02rem !important;
            line-height: 1.25 !important;
            margin-bottom: 0.35rem !important;
        }
        div[data-baseweb="select"] > div,
        .stNumberInput input {
            background: #444444 !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
            min-height: 3.05rem !important;
        }
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] input {
            font-size: 1.06rem !important;
            color: #ffffff !important;
        }
        .stButton button {
            background: #595959;
            color: white;
            border: none;
            border-radius: 8px;
            min-height: 3.55rem;
            font-size: 1.12rem;
            font-weight: 800;
            box-shadow: none;
        }
        .stButton button:hover {
            background: #666666;
            color: white;
            border: none;
        }
        .st-key-predict_card .stButton {
            margin-top: 1.28rem;
        }

        @media (max-width: 1100px) {
            .hero-title {
                font-size: 3rem;
            }
            .st-key-right_panel {
                margin-top: 1rem;
            }
            .model-name-list {
                font-size: 1.25rem;
                margin-left: 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    st.markdown('<div class="hero-title">Web App Demo</div>', unsafe_allow_html=True)

    example_names = list(EXAMPLE_PATIENTS.keys())
    if "example_patient" not in st.session_state:
        st.session_state["example_patient"] = example_names[0]
    selected_example = st.session_state["example_patient"]
    defaults = EXAMPLE_PATIENTS[selected_example]

    artifact_version = get_artifact_version()

    left_col, right_col = st.columns([1.02, 1.0], gap="large")

    with left_col:
        with st.container(key="left_panel_shell"):
            st.markdown(
                '<div class="left-panel-header"><span>&#9997; Enter Patient Features</span><span class="caret">&#9660;</span></div>',
                unsafe_allow_html=True,
            )

            with st.container(key="form_row_1"):
                c1, c2, c3, c4 = st.columns(4)
                age = c1.number_input("age (years)", min_value=1, max_value=120, value=int(defaults["age"]))
                sex = c2.selectbox("sex (0=female, 1=male)", [0, 1], index=int(defaults["sex"]))
                cp = c3.selectbox("cp (chest pain type 1..4)", [1, 2, 3, 4], index=int(defaults["cp"]) - 1)
                trestbps = c4.number_input("trestbps (resting BP mmHg)", min_value=60, max_value=250, value=int(defaults["trestbps"]))

            with st.container(key="form_row_2"):
                c1, c2, c3, c4 = st.columns(4)
                chol = c1.number_input("chol (serum cholesterol mg/dl)", min_value=100, max_value=700, value=int(defaults["chol"]))
                fbs = c2.selectbox("fbs (>120 mg/dl? 1/0)", [0, 1], index=int(defaults["fbs"]))
                restecg = c3.selectbox("restecg (0..2)", [0, 1, 2], index=int(defaults["restecg"]))
                thalach = c4.number_input("thalach (max heart rate)", min_value=50, max_value=250, value=int(defaults["thalach"]))

            with st.container(key="form_row_3"):
                c1, c2, c3, c4 = st.columns(4)
                exang = c1.selectbox("exang (exercise angina 1/0)", [0, 1], index=int(defaults["exang"]))
                oldpeak = c2.number_input("oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=float(defaults["oldpeak"]), step=0.1)
                slope = c3.selectbox("slope (1..3)", [1, 2, 3], index=int(defaults["slope"]) - 1)
                ca = c4.selectbox("ca (major vessels 0..3)", [0, 1, 2, 3], index=int(defaults["ca"]))

            with st.container(key="form_row_4"):
                thal = st.selectbox("thal (3=normal, 6=fixed, 7=reversible)", [3, 6, 7], index=[3, 6, 7].index(int(defaults["thal"])))

            with st.container(key="form_action_row"):
                bottom_left, bottom_right = st.columns([1.0, 1.05], gap="small")
                with bottom_left:
                    with st.container(key="example_card"):
                        st.selectbox("Select Example Patient", example_names, key="example_patient")
                with bottom_right:
                    with st.container(key="predict_card"):
                        predict_clicked = st.button("🔍 Predict", use_container_width=True)

    actual_target = int(EXAMPLE_PATIENTS[st.session_state["example_patient"]]["actual_target"])
    patient_input = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    current_signature = build_prediction_signature(artifact_version, patient_input, actual_target)
    saved_signature = st.session_state.get("prediction_signature")

    if predict_clicked or saved_signature != current_signature or "prediction_results" not in st.session_state:
        patient_df = encode_patient_input(patient_input)
        st.session_state["prediction_results"] = predict_all_models(patient_df, patient_input, actual_target)
        st.session_state["actual_target"] = actual_target
        st.session_state["prediction_signature"] = current_signature

    with right_col:
        results_df = st.session_state.get("prediction_results")
        with st.container(key="right_panel"):
            st.markdown('<div class="chart-tag"><svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;margin-right:6px"><line x1="2" y1="16" x2="16" y2="16" stroke="white" stroke-width="1.5" stroke-linecap="round"/><line x1="2" y1="16" x2="2" y2="2" stroke="white" stroke-width="1.5" stroke-linecap="round"/><circle cx="5" cy="12" r="1.5" fill="white"/><circle cx="8" cy="8" r="1.5" fill="white"/><circle cx="11" cy="10" r="1.5" fill="white"/><circle cx="14" cy="5" r="1.5" fill="white"/><line x1="3.5" y1="13.5" x2="13.5" y2="4.5" stroke="white" stroke-width="1" stroke-linecap="round" stroke-dasharray="2 2"/></svg> Model Predictions Overview</div>', unsafe_allow_html=True)
            render_chart(results_df)
        render_model_name_list()


if __name__ == "__main__":
    main()
