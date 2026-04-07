# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
st.set_page_config(
    page_title="Cancer Diagnosis Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "The_Cancer_data_1500_V2.csv")
TARGET_COLUMN = "Diagnosis"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    X_raw = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Encode non-numeric columns if present.
    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    return {
        "X_columns": X.columns.tolist(),
        "X_template": X,
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "lr_acc": lr_acc,
        "rf_acc": rf_acc,
        "lr_precision": lr_precision,
        "rf_precision": rf_precision,
        "lr_recall": lr_recall,
        "rf_recall": rf_recall,
        "lr_f1": lr_f1,
        "rf_f1": rf_f1,
        "y_test": y_test,
        "rf_pred": rf_pred,
        "lr_pred": lr_pred,
    }


def build_input_row(X_template: pd.DataFrame) -> pd.DataFrame:
    input_values = {}
    
    cols = st.columns(3)
    for idx, col in enumerate(X_template.columns):
        series = X_template[col]
        col_min = float(series.min())
        col_max = float(series.max())
        col_mean = float(series.mean())

        with cols[idx % 3]:
            if col.lower() == "age":
                input_values[col] = int(
                    st.number_input(
                        f"**{col}**",
                        min_value=int(col_min),
                        max_value=int(col_max),
                        value=int(round(col_mean)),
                        step=1,
                        key=f"input_{col}",
                    )
                )
            elif col.lower() == "gender":
                default_val = int(round(col_mean))
                gender_label = st.selectbox(
                    "**Gender**",
                    options=["Female", "Male"],
                    index=default_val,
                    key=f"input_{col}",
                )
                input_values[col] = 1 if gender_label == "Male" else 0
            elif col.lower() == "smoking":
                default_val = int(round(col_mean))
                smoking_label = st.selectbox(
                    "**Smoking**",
                    options=["No", "Yes"],
                    index=default_val,
                    key=f"input_{col}",
                )
                input_values[col] = 1 if smoking_label == "Yes" else 0
            elif set(np.unique(series)).issubset({0, 1}):
                default_val = int(round(col_mean))
                input_values[col] = st.selectbox(
                    f"**{col}**",
                    options=[0, 1],
                    index=default_val,
                    key=f"input_{col}",
                )
            else:
                step = max((col_max - col_min) / 100.0, 0.01)
                input_values[col] = st.number_input(
                    f"**{col}**",
                    min_value=col_min,
                    max_value=col_max,
                    value=col_mean,
                    step=step,
                    key=f"input_{col}",
                )

    return pd.DataFrame([input_values])


def display_model_metrics(artifacts):
    """Display model performance metrics in a beautiful format."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Logistic Regression")
        met1, met2, met3, met4 = st.columns(4)
        with met1:
            st.metric("Accuracy", f"{artifacts['lr_acc']:.2%}")
        with met2:
            st.metric("Precision", f"{artifacts['lr_precision']:.2%}")
        with met3:
            st.metric("Recall", f"{artifacts['lr_recall']:.2%}")
        with met4:
            st.metric("F1 Score", f"{artifacts['lr_f1']:.2%}")
    
    with col2:
        st.markdown("#### Random Forest")
        met1, met2, met3, met4 = st.columns(4)
        with met1:
            st.metric("Accuracy", f"{artifacts['rf_acc']:.2%}")
        with met2:
            st.metric("Precision", f"{artifacts['rf_precision']:.2%}")
        with met3:
            st.metric("Recall", f"{artifacts['rf_recall']:.2%}")
        with met4:
            st.metric("F1 Score", f"{artifacts['rf_f1']:.2%}")


def display_cancer_info():
    st.markdown("### Cancer Information")
    st.markdown(
        "Cancer is a group of diseases where abnormal cells grow uncontrollably and can spread to other parts of the body. "
        "Early detection improves treatment outcomes, which is why regular screening and medical checkups matter."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**Common warning signs**\n\n"
            "- Unusual lumps or swelling\n"
            "- Unexplained weight loss\n"
            "- Persistent pain or fatigue"
        )
    with col2:
        st.markdown(
            "**Common risk factors**\n\n"
            "- Smoking or tobacco use\n"
            "- Family history\n"
            "- Poor diet and inactivity"
        )
    with col3:
        st.markdown(
            "**Healthy habits**\n\n"
            "- Avoid tobacco\n"
            "- Stay active\n"
            "- Go for regular screenings"
        )

    st.info(
        "This app is for educational and prediction support only. It does not replace professional medical advice, diagnosis, or treatment."
    )


def display_agewise_cancer_rate(df: pd.DataFrame):
    st.markdown("### Cancer Rate by Age")
    st.caption("The dataset does not include calendar year data, so this chart shows cancer rate grouped by age.")

    age_rate = (
        df.groupby("Age")[TARGET_COLUMN]
        .mean()
        .reset_index()
        .sort_values("Age")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(age_rate["Age"], age_rate[TARGET_COLUMN] * 100, marker="o", linewidth=2, color="#667eea")
    ax.fill_between(age_rate["Age"], age_rate[TARGET_COLUMN] * 100, color="#667eea", alpha=0.12)
    ax.set_xlabel("Age", fontweight="bold")
    ax.set_ylabel("Cancer Rate (%)", fontweight="bold")
    ax.set_title("Cancer Rate by Age", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    st.pyplot(fig, use_container_width=True)


def main() -> None:
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            height: 50px;
            font-size: 1.1rem;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #667eea; font-size: 3rem;">CANCER DIAGNOSIS PREDICTOR</h1>
        <p style="color: #666; font-size: 1.1rem;">AI-Powered Medical Diagnosis System</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Dataset not found at '{DATA_PATH}'. Place the CSV in the same folder as this app.")
        st.stop()

    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' is missing in dataset.")
        st.stop()

    # Train models
    artifacts = train_models(df)

    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        model_name = st.radio(
            "Select Model:",
            ["Random Forest", "Logistic Regression"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("## Model Performance")
        display_model_metrics(artifacts)

    # Main content
    tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Info", "Model Comparison"])

    with tab1:
        st.markdown("### Enter Patient Medical Data")
        st.markdown("*Use the form below to input patient measurements*")
        
        input_df = build_input_row(artifacts["X_template"])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("Predict Diagnosis", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner("Analyzing patient data..."):
                if model_name == "Random Forest":
                    pred = artifacts["rf"].predict(input_df)[0]
                    pred_proba = artifacts["rf"].predict_proba(input_df)[0]
                    model_used = "Random Forest"
                else:
                    scaled_input = artifacts["scaler"].transform(input_df)
                    pred = artifacts["lr"].predict(scaled_input)[0]
                    pred_proba = artifacts["lr"].predict_proba(scaled_input)[0]
                    model_used = "Logistic Regression"

                # Display results
                st.markdown("---")
                
                if int(pred) == 1:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown("### WARNING")
                    with col2:
                        st.markdown(f"### **Cancer Positive** - Further Investigation Recommended")
                    confidence = pred_proba[1] * 100
                    st.info(f"**Model:** {model_used}\n**Confidence:** {confidence:.1f}%")
                else:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown("### NORMAL")
                    with col2:
                        st.markdown(f"### **Cancer Negative** - Normal Result")
                    confidence = pred_proba[0] * 100
                    st.success(f"**Model:** {model_used}\n**Confidence:** {confidence:.1f}%")
                
                # Confidence visualization
                st.markdown("#### Prediction Confidence")
                prob_df = pd.DataFrame({
                    "Diagnosis": ["Negative", "Positive"],
                    "Probability": [pred_proba[0] * 100, pred_proba[1] * 100]
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#2ecc71' if prob_df.iloc[i]["Diagnosis"] == "Negative" else '#e74c3c' 
                         for i in range(len(prob_df))]
                bars = ax.barh(prob_df["Diagnosis"], prob_df["Probability"], color=colors)
                ax.set_xlabel("Probability (%)", fontsize=12, fontweight='bold')
                ax.set_xlim(0, 100)
                
                # Add percentage labels
                for i, (bar, prob) in enumerate(zip(bars, prob_df["Probability"])):
                    ax.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold')
                
                ax.set_facecolor('#f8f9fa')
                fig.patch.set_facecolor('#f8f9fa')
                st.pyplot(fig, use_container_width=True)

    with tab2:
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            cancer_positive = (df[TARGET_COLUMN] == 1).sum()
            st.metric("Cancer Positive", cancer_positive)
        with col4:
            cancer_negative = (df[TARGET_COLUMN] == 0).sum()
            st.metric("Cancer Negative", cancer_negative)
        
        st.markdown("---")
        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
        display_agewise_cancer_rate(df)

        st.markdown("---")
        display_cancer_info()

    with tab3:
        st.markdown("### Model Performance Comparison")
        
        metrics_comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Random Forest": [
                artifacts['rf_acc'],
                artifacts['rf_precision'],
                artifacts['rf_recall'],
                artifacts['rf_f1']
            ],
            "Logistic Regression": [
                artifacts['lr_acc'],
                artifacts['lr_precision'],
                artifacts['lr_recall'],
                artifacts['lr_f1']
            ]
        })
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metrics_comparison))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, metrics_comparison['Random Forest'], width, 
                       label='Random Forest', color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, metrics_comparison['Logistic Regression'], width,
                       label='Logistic Regression', color='#764ba2', alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_comparison['Metric'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.dataframe(metrics_comparison.set_index("Metric"), use_container_width=True)


if __name__ == "__main__":
    main()
