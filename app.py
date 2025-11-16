import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import matplotlib.pyplot as plt


st.set_page_config(page_title="Iris Data Visualization", layout="wide")

st.title("Iris Dataset Visualization App")


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df, iris

df, iris = load_data()

st.sidebar.header("Filter Options")

species_selected = st.sidebar.multiselect(
    "Select Species",
    df["species"].unique(),
    default=df["species"].unique()
)

x_axis = st.sidebar.selectbox(
    "X-axis",
    options=df.columns[:-1],
    index=0
)

y_axis = st.sidebar.selectbox(
    "Y-axis",
    options=df.columns[:-1],
    index=1
)

filtered_df = df[df["species"].isin(species_selected)]

st.subheader("Dataset Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", len(df))
col2.metric("Filtered Rows", len(filtered_df))
col3.metric("Unique Species", df["species"].nunique())

st.subheader("🌿 Scatter Plot")
fig_scatter = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color="species",
    title=f"{x_axis} vs {y_axis}"
)
st.plotly_chart(fig_scatter, use_container_width=True)


col_left, col_right = st.columns(2)

with col_left:
    
    numeric_col = st.sidebar.selectbox(
        "Select column for histogram",
        options=df.columns[:-1],
        index=2
    )

    fig_hist = px.histogram(
        filtered_df,
        x=numeric_col,
        color="species",
        title=f"Distribution of {numeric_col}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.markdown("### Pie Chart (Species Distribution)")
    species_counts = df["species"].value_counts()
    species_labels = species_counts.index.tolist()

    fig, ax = plt.subplots()
    ax.pie(
        species_counts,
        labels=species_labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)


if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)
