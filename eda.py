import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
fig = px.histogram(df, x=selected_col, nbins=50,
                   title=f"Histogram of {selected_col}")
st.plotly_chart(fig)

st.header("Correlation Heatmap")
corr = df.corr()
fig2 = px.imshow(corr, text_auto=True,
                 color_continuous_scale='RdBu', title="Correlation Heatmap")
fig2.update_layout(width=700, height=700)
st.plotly_chart(fig2)

st.header("Pairplot")
st.write("Pairplot of selected features (first 5 columns):")
fig3 = go.Figure(
    data=go.Splom(
        dimensions=[
            dict(label=col, values=df[col]) for col in df.columns[:5]
        ],
        diagonal=dict(visible=False),
        marker=dict(
            size=5,
            color='blue',           # Set marker color
            opacity=0.7,           # Set transparency
            line=dict(             # Add border to markers
                width=0.5,
                color='darkblue'
            )
        )
    )
)
fig3.update_layout(
    dragmode='select',
    width=700,
    height=700,
    hovermode='closest')
st.plotly_chart(fig3)

st.header("Pairplot of 2 Selected Features")
feature_x = st.selectbox("Select X-axis feature:", df.columns[1: ])
feature_y = st.selectbox("Select Y-axis feature:", df.columns[ 1 if feature_x != df.columns[1] else 2:])
fig4 = px.scatter(df, x=feature_x, y=feature_y,  marginal_y="violin",
                  marginal_x="box",
                  title=f"Scatter Plot of {feature_x} vs {feature_y}")
st.plotly_chart(fig4)
