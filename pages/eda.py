import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA page content only (no navigation/sidebar)
st.title("Body Fat Data EDA")
st.write("Explore the body fat dataset with interactive visualizations and summary statistics.")

# Load data
csv_path = "data/bodyfat.csv"
df = pd.read_csv(csv_path)

st.header("Data Preview")
st.dataframe(df.head())

st.header("Summary Statistics")
st.write(df.describe())

st.header("Histograms")
selected_col = st.selectbox(
    "Select a column to view its histogram:", df.columns)
fig, ax = plt.subplots()
ax.hist(df[selected_col], bins=20, color='skyblue', edgecolor='black')
ax.set_title(f"Histogram of {selected_col}")
st.pyplot(fig)

st.header("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.header("Pairplot")
st.write("Pairplot of selected features (first 5 columns):")
fig3 = sns.pairplot(df[df.columns[:5]])
st.pyplot(fig3)
