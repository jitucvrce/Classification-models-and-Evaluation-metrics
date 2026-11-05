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

# 1. Set page configuration layout
st.set_page_config(page_title="BITS V-Lab Assignment Pipeline", layout="wide")

st.title("📊 Benchmarking Suite for All 5 ML Classification Models")
st.caption("Fulfilling BITS Virtual Lab Evaluation Assignment Framework | Dataset Parameters: 14 Features, 600 Instances")

# 2. Compliant Synthetic Dataset Generation Pipeline
@st.cache_data
def generate_dataset():
    # Constructing a dataset with 14 features (>= 12) and 600 instances (>= 500)
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

# 3. Sidebar Layout Controls
st.sidebar.header("⚙️ Model Train-Test Splits Settings")
test_size = st.sidebar.slider("Test Split Size Proportion", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random Seed Allocation Integer", value=42, step=1)

# Preprocessing tabular configurations splits
X = df.drop(columns=['target']).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# 4. Tab Layout Architecture
tab1, tab2, tab3 = st.tabs(["📋 Tabular Data Verification", "🔍 Specific Classifier Focus", "🏆 Unified Benchmarking Matrix"])

# Dictionary configuration encapsulating all 5 required models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes Classifier": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=random_state)
}

# TAB 1: Preview Data Dataframe Check
with tab1:
    st.subheader("Dataset Constraints Check Verification")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Instances Row Count", df.shape[0], delta="Compliant (>=500)")
    col2.metric("Total Extracted Columns (Features)", df.shape[1] - 1, delta="Compliant (>=12)")
    col3.metric("Target Output Sub-classes", df['target'].nunique(), delta="Binary Schema")
    
    st.markdown("### Top 10 Data Rows Preview for `test_data.csv` verification:")
    st.dataframe(df.head(10), use_container_width=True)
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download generated 'test_data.csv' file for repository commit",
        data=csv_bytes,
        file_name="test_data.csv",
        mime="text/csv"
----
    )

# TAB 2: Specific Inspector Breakdown view
with tab2:
    st.subheader("Interactive Metrics Inspector per Classifier Model")
    selected_model_name = st.selectbox("Choose a model algorithm to focus metrics metrics summary:", list(models.keys()))
    
    active_model = models[selected_model_name]
    active_model.fit(X_train, y_train)
    
    preds = active_model.predict(X_test)
    probs = active_model.predict_proba(X_test)[:, 1] if hasattr(active_model, "predict_proba") else preds
    
    # Isolation scorecard display columns
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Accuracy Score", f"{accuracy_score(y_test, preds):.4f}")
    m_col2.metric("AUC Score", f"{roc_auc_score(y_test, probs):.4f}")
    m_col3.metric("Precision", f"{precision_score(y_test, preds):.4f}")
    
    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("Recall Rate", f"{recall_score(y_test, preds):.4f}")
    m_col5.metric("F1 Metric Score", f"{f1_score(y_test, preds):.4f}")
    m_col6.metric("Matthews Correlation (MCC)", f"{matthews_corrcoef(y_test, preds):.4f}")

# TAB 3: Main Benchmark Pipeline Matrix
with tab3:
    st.subheader("Comparative Benchmark Performance Matrix View")
    st.write("Below matrix displays all evaluations computed directly under active test constraints:")
    
    results_list = []
    
    for name, model_obj in models.items():
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        y_prob = model_obj.predict_proba(X_test)[:, 1] if hasattr(model_obj, "predict_proba") else y_pred
        
        results_list.append({
            "Classification Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC Score": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "MCC Score": round(matthews_corrcoef(y_test, y_pred), 4)
        })
        
    results_df = pd.DataFrame(results_list)
    st.dataframe(results_df, use_container_width=True)
    
    # Render direct visual comparison heatmaps
    st.markdown("### Comparative Performance Heatmap Matrix View")
    plot_df = results_df.set_index("Classification Model")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(plot_df, annot=True, cmap="YlGnBu", fmt=".4f", cbar=True, ax=ax)
    plt.title("Visual Multi-Class Benchmark Grid Profile Breakdown")
    st.pyplot(fig)
