import pandas as pd
def load_data():
    df = pd.read_csv("adult 3.csv")
    return df
# Preprocessing function using pandas
def clean_data(df):
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Encoding categorical features
def encode_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df
# Splitting the dataset
def split_data(df):
    X = df.drop("income", axis=1)
    y = df["income"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
# Training the model
def train_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, "model.pkl")
    return clf
def get_model(X_train, y_train):
    try:
        model = joblib.load("model.pkl")
    except:
        model = train_model(X_train, y_train)
    return model

import matplotlib.pyplot as plt
import seaborn as sns
def plot_income_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    income_counts = df['income'].value_counts()
    labels = income_counts.index
    sizes = income_counts.values
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
    ax.axis('equal')
    return fig

import streamlit as st
import numpy as np
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("Employee Salary Prediction App")
st.markdown("This app predicts whether an employee earns more than 50K based on provided features.")
# Load and clean data
df = load_data()
df = clean_data(df)
# Visualize
# Encode data
st.markdown("---")
df_encoded = encode_data(df)
st.subheader("Feature Correlation Matrix")
corr = df_encoded.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
# Encode & Split
X_train, X_test, y_train, y_test = split_data(df_encoded)
model = get_model(X_train, y_train)
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text(classification_report(y_test, y_pred))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Feature Importances")
importances = model.feature_importances_
features = X_train.columns
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
st.pyplot(fig)

# User input form
st.subheader("Enter Details for Prediction")
input_data = []
for col in X_train.columns:
    val = st.number_input(f"{col}", step=1.0)
    input_data.append(val)

# Prediction
if st.button("Predict Income"):
    result = model.predict([input_data])[0]
    if result == 1:
        st.success("Predicted Income: >50K")
    else:
        st.success("Predicted Income: <=50K")