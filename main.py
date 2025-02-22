import streamlit as st
import pandas as pd
import plotly.express as px

# Load the CSV file
@st.cache_data
def load_data():
    file_path = "BRCA.csv"  # Ensure the file is in the same directory
    df = pd.read_csv(file_path)

    # Convert dates
    df["Date_of_Surgery"] = pd.to_datetime(df["Date_of_Surgery"], errors='coerce')
    df["Date_of_Last_Visit"] = pd.to_datetime(df["Date_of_Last_Visit"], errors='coerce')

    return df

df = load_data()

# Title
st.title("BRCA Data Visualization")

# Sidebar for user interaction
st.sidebar.header("Filter Options")
selected_tumor_stage = st.sidebar.multiselect(
    "Select Tumor Stage:",
    options=df["Tumour_Stage"].dropna().unique(),
    default=df["Tumour_Stage"].dropna().unique()
)

filtered_df = df[df["Tumour_Stage"].isin(selected_tumor_stage)]

# 1. Age Distribution
st.subheader("Age Distribution")
fig_age = px.histogram(df, x="Age", nbins=20, marginal="box", title="Age Distribution", color_discrete_sequence=["teal"])
st.plotly_chart(fig_age)

# 2. Tumor Stage Distribution
st.subheader("Tumor Stage Distribution")
fig_tumor = px.bar(df["Tumour_Stage"].value_counts(), title="Tumor Stage Distribution", color_discrete_sequence=["orange"])
st.plotly_chart(fig_tumor)

# 3. Surgery Type Distribution
st.subheader("Surgery Type Distribution")
fig_surgery = px.bar(df["Surgery_type"].value_counts(), title="Surgery Type Distribution", color_discrete_sequence=["blue"])
st.plotly_chart(fig_surgery)

# 4. Protein Levels by Tumor Stage
st.subheader("Protein Levels by Tumor Stage")
fig_protein = px.box(filtered_df, x="Tumour_Stage", y="Protein1", title="Protein1 Levels by Tumor Stage", color="Tumour_Stage")
st.plotly_chart(fig_protein)

# 5. Patient Status Breakdown
st.subheader("Patient Status Breakdown")
fig_status = px.pie(df, names="Patient_Status", title="Patient Status Breakdown", color_discrete_sequence=["lightblue", "salmon"])
st.plotly_chart(fig_status)

# Run the app using: `streamlit run app.py`
