import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Clinic & Candidate Analysis App", layout="wide")

# Load data
@st.cache_data
def load_data():
    appointments = pd.read_csv("expanded_clinic_appointments.csv")
    candidates = pd.read_csv("expanded_job_candidates.csv")
    # Load patient medication data
    medications = pd.read_csv("expanded_medication_schedules.csv")
    return appointments, candidates, medication schedules

appointments, candidates, medications = load_data()

st.title("🌐 HealthConnect App")


# Main tabs for navigation
tab1, tab2, tab3 = st.tabs(["Clinic Appointments", "Job Candidates", "Patient Medication Schedules"])

with tab1:
    st.header("Clinic Appointments Overview")
    if st.checkbox("Show raw appointment data"):
        st.write(appointments.head())

    if 'status' in appointments.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=appointments, x='status', ax=ax)
        ax.set_title("Appointment Status Distribution")
        st.pyplot(fig)

with tab2:
    st.header("Job Candidate Data Overview")
    if st.checkbox("Show raw candidate data"):
        st.write(candidates.head())

    # Basic prediction demo (if label and text features are available)
    if 'hired' in candidates.columns and 'resume' in candidates.columns:
        st.subheader("Basic Hiring Prediction Model")
        candidates = candidates.dropna(subset=['resume', 'hired'])
        X = candidates['resume']
        y = candidates['hired']

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of Logistic Regression: {acc:.2f}")

        user_input = st.text_area("Paste a candidate resume below:")
        if st.button("Predict Hiring Outcome"):
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]
            st.success(f"Prediction: {'Hired' if prediction else 'Not Hired'}")

with tab3:
    st.header("Patient Medication Scedules")
    if st.checkbox("Show raw medication data"):
        st.write(medications.head())

    # Basic summary or visualization
    if not medications.empty:
        st.subheader("Medication Count by Medication Name")
        med_counts = medications['medication_name'].value_counts()
        st.bar_chart(med_counts)

        # Optional: Show frequency by start or end date
        if 'start_date' in medications.columns:
            medications['start_date'] = pd.to_datetime(medications['start_date'])
            st.subheader("Medications Started Over Time")
            start_counts = medications['start_date'].dt.to_period('M').value_counts().sort_index()
            st.line_chart(start_counts)

