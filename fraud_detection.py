import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def preprocess_data(data, handle_missing=False, remove_duplicates=False, scale_data=True):
    if handle_missing:
        data = data.dropna()
    
    if remove_duplicates:
        data = data.drop_duplicates()
    
    if scale_data:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def apply_pca(data, n_components=2):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data[numeric_columns])
    return pca_result, pca.explained_variance_ratio_

def split_features_labels(data, target_column='Class'):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if y.dtypes != 'int' and y.dtypes != 'object':
        y = (y > 0.5).astype(int)
    return X, y

def get_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

def get_param_grid():
    return {
        "Logistic Regression": {'C': [0.1, 1, 10]},
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        "XGBoost": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
    }

def evaluate_model(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred)
    }

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def main():
    st.title("💳 Credit Card Fraud Detection App")

    uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

    if uploaded_file:
        data = load_data(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        st.write(f"Initial number of rows: {data.shape[0]}")

        st.subheader("🧹 Data Preprocessing")

        handle_missing = st.checkbox("Remove missing values")
        remove_duplicates = st.checkbox("Remove duplicate rows")

        data = preprocess_data(data, handle_missing, remove_duplicates)

        st.write(f"Number of rows after preprocessing: {data.shape[0]}")

        if st.checkbox("Apply PCA (2 Components)"):
            pca_result, variance_ratio = apply_pca(data)
            st.write(f"Explained Variance Ratio: {variance_ratio}")

        # Splitting features and labels
        if 'Class' not in data.columns:
            st.error("Dataset must contain a 'Class' column for fraud labels.")
            return

        X, y = split_features_labels(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("🧠 Model Training")

        models = get_models()
        selected_model_name = st.selectbox("Choose a model", list(models.keys()))

        model = models[selected_model_name]

        if st.checkbox("Perform Grid Search (Hyperparameter Tuning)"):
            param_grid = get_param_grid()
            grid_search = GridSearchCV(model, param_grid[selected_model_name], cv=5)
            try:
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f"Best Parameters: {grid_search.best_params_}")
            except Exception as e:
                st.error(f"Grid search failed: {e}")
                return

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        metrics = evaluate_model(y_test, y_pred)
        st.subheader("📊 Model Performance")
        for k, v in metrics.items():
            st.write(f"{k}: {v:.2f}")

        st.subheader("🔍 Confusion Matrix")
        plot_confusion_matrix(y_test, y_pred)

        # Fraud amount (if exists)
        if 'Amount' in data.columns:
            total_fraud_amount = data[data['Class'] == 1]['Amount'].sum()
            st.write(f"💰 Total Fraudulent Amount: ${total_fraud_amount:.2f}")

        # Real-time Prediction
        st.subheader("🧪 Real-time Fraud Prediction")

        num_rows = st.number_input("Select number of rows to check", min_value=1, max_value=len(data), value=1)

        selected_rows = data.head(num_rows)
        st.dataframe(selected_rows)

        for idx, row in selected_rows.iterrows():
            st.markdown(f"#### Row {idx + 1}")
            input_data = row[X.columns].to_dict()  # Use only features
            input_df = pd.DataFrame([input_data])

            if st.button(f"Predict Fraud for Row {idx + 1}"):
                try:
                    prediction = model.predict(input_df)[0]
                    result = "🚨 Fraudulent" if prediction == 1 else "✅ Legitimate"
                    st.success(f"Prediction: {result}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
