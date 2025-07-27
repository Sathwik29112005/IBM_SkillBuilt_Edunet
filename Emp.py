# Import Libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df
df = load_data()

# Encode Categorical Features
def encode_data(df):
    df_copy = df.copy()
    label_encoders = {}
    for col in df_copy.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
        label_encoders[col] = le
    return df_copy, label_encoders
df_encoded, label_encoders = encode_data(df)

# Prepare features and target
X = df_encoded.drop("income", axis=1)
y = df_encoded["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train or load model
def train_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, "model.pkl")
    return clf
def get_model():
    try:
        model = joblib.load("model.pkl")
    except:
        model = train_model(X_train, y_train)
    return model
model = get_model()

# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar input
with st.sidebar:
    st.header("Input Employee Details")
    input_data = {}
    for col in X.columns:
        if col in df.select_dtypes(include='object').columns:
            input_data[col] = st.selectbox(f"{col}", df[col].unique())
        elif df[col].dtype in [np.int64, np.float64]:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            default = int(df[col].median())
            input_data[col] = st.slider(f"{col}", min_val, max_val, default)

# Create DataFrame for input
input_df = pd.DataFrame([input_data])

# Encode input using training label encoders
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"Invalid input for {col}: value not seen in training data")
            st.stop()

# Ensure columns match training set
input_df = input_df[X.columns]

# Display input
st.subheader("Entered Input")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction")
    if prediction == 1:
        st.success("Predicted Income: >50K")
    else:
        st.success("Predicted Income: ≤50K")

# Correlation Matrix
st.subheader("Feature Correlation Matrix")
corr = df_encoded.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Model Evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importances
st.subheader("Feature Importances")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
st.pyplot(fig)