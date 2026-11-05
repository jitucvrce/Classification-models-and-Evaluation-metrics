import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)

# Set page configuration
st.set_page_config(page_title="ML Classifier Benchmark Dashboard", layout="wide")

st.title("📊 Multi-Model Machine Learning Classification Dashboard")
st.caption("Fulfilling BITS Virtual Lab Assignment Constraints | Dataset: 14 Features, 600 Instances")

# 1. Dataset Generation & Loading Pipeline
@st.cache_data
def generate_dataset():
    # 14 features (>= 12), 600 instances (>= 500)
    X, y = make_classification(
        n_samples=600, 
        n_features=14, 
        n_informative=10, 
        n_classes=2, 
        random_state=42
    )
    feature_names = [f"feature_{i+1}" for i in range(14)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

df = generate_dataset()

# Sidebar Layout for Interactive Inputs
st.sidebar.header("⚙️ Configuration Panels")
test_size = st.sidebar.slider("Test Split Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random Seed State", value=42, step=1)

# Preprocessing splits
X = df.drop(columns=['target']).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "🚀 Classifier Evaluation", "📈 Performance Matrix"])

with tab1:
    st.subheader("Dataset Parameters Checklist Verification")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Instances (Rows)", df.shape[0], delta="Compliant (>=500)")
    col2.metric("Feature Parameters (Columns)", df.shape[1] - 1, delta="Compliant (>=12)")
    col3.metric("Target Output Classes", df['target'].nunique(), delta="Binary Classification")
    
    st.markdown("### Preview of Generated Test Data Baseline (`test_data.csv`)")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download buffer for submission requirement verification
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download test_data.csv for GitHub Repository",
        data=csv_data,
        file_name="test_data.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Model Selection & Parameter Control Setup")
    
    # Let users choose the model dynamically to inspect individual details
    selected_model_name = st.selectbox(
        "Select Model to Inspect Parameters",
        ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier", "Naive Bayes Classifier", "Support Vector Machine"]
    )
    
    # Dictionary to initialize actual models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes Classifier": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=random_state)
    }
    
    # Train specific model
    active_model = models[selected_model_name]
    active_model.fit(X_train, y_train)
    
    y_pred = active_model.predict(X_test)
    y_prob = active_model.predict_proba(X_test)[:, 1] if hasattr(active_model, "predict_proba") else y_pred
    
    # Calculate Individual Metrics
    m_acc = accuracy_score(y_test, y_pred)
    m_auc = roc_auc_score(y_test, y_prob)
    m_prec = precision_score(y_test, y_pred)
    m_rec = recall_score(y_test, y_pred)
    m_f1 = f1_score(y_test, y_pred)
    m_mcc = matthews_corrcoef(y_test, y_pred)
    
    # Performance cards for the selected model
    st.markdown(f"#### Performance Scores for **{selected_model_name}**")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Accuracy Score", f"{m_acc:.4f}")
    m_col2.metric("AUC Score", f"{m_auc:.4f}")
    m_col3.metric("Precision Score", f"{m_prec:.4f}")
    
    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("Recall Score", f"{m_rec:.4f}")
    m_col5.metric("F1 Score", f"{m_f1:.4f}")
    m_col6.metric("MCC Score", f"{m_mcc:.4f}")

with tab3:
    st.subheader("Unified Benchmarking Stack Matrix (All 5 Classifiers)")
    st.markdown("Running all models simultaneously to extract the required documentation submission table:")
    
    results_list = []
    
    for name, model_obj in models.items():
        # Train and extract
        model_obj.fit(X_train, y_train)
        preds = model_obj.predict(X_test)
        probs = model_obj.predict_proba(X_test)[:, 1] if hasattr(model_obj, "predict_proba") else preds
        
        results_list.append({
            "Model Classifier": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "AUC Score": round(roc_auc_score(y_test, probs), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1 Score": round(f1_score(y_test, preds), 4),
            "MCC Score": round(matthews_corrcoef(y_test, preds), 4)
        })
        
    results_df = pd.DataFrame(results_list)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizing performance comparison chart
    st.markdown("### Metric Comparison Chart")
    plot_df = results_df.set_index("Model Classifier")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(plot_df, annot=True, cmap="Blues", fmt=".4f", cbar=True, ax=ax)
    plt.title("Classifier Performance Heatmap Across Evaluation Criteria")
    st.pyplot(fig)
