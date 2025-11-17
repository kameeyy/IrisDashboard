import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris Data Visualization", layout="wide")

# Simple CSS to polish the look
st.markdown(
    """
    <style>
    .header {
        font-size:20px;
        margin-bottom: -10px;
    }
    .metric-box {
        border-radius: 8px;
        padding: 8px;
    }
    .plotly-graph-div {
        border-radius:10px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df

df = load_data()


st.sidebar.header("Filter & Plot Options")

species_selected = st.sidebar.multiselect(
    "Select Species",
    options=df["species"].unique(),
    default=list(df["species"].unique())
)

# Axis selectors
x_axis = st.sidebar.selectbox("X-axis", options=df.columns[:-1], index=0)
y_axis = st.sidebar.selectbox("Y-axis", options=df.columns[:-1], index=1)


numeric_col = st.sidebar.selectbox("Histogram column", options=df.columns[:-1], index=2)


show_pairplot = st.sidebar.checkbox("Show Pairplot (Seaborn)", value=False)
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=False)


filtered_df = df[df["species"].isin(species_selected)]


st.title("Iris Dataset Dashboard")
st.markdown(
    "Interactive dashboard to explore the classic **Iris** dataset. "
    "Use the sidebar to filter species and choose features to visualize."
)
st.markdown("---")


col1, col2, col3 = st.columns([1,1,1])
col1.metric("Total rows", len(df))
col2.metric("Filtered rows", len(filtered_df))
col3.metric("Unique species", df["species"].nunique())

 
tab_scatter, tab_distribution, tab_overview = st.tabs(["Scatter", "Distribution", "Overview"])

# 1) Scatter tab
with tab_scatter:
    st.subheader(f"Scatter Plot — {x_axis} vs {y_axis}")
    @st.cache_data
    def get_scatter(data, x, y):
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color="species",
            title=f"{x} vs {y}",
            labels={x: x, y: y},
            marginal_x="violin",
            marginal_y="violin",
            hover_data=data.columns.tolist()
        )
        fig.update_layout(legend_title_text="Species")
        return fig

    fig_scatter = get_scatter(filtered_df, x_axis, y_axis)
    st.plotly_chart(fig_scatter, use_container_width=True)

# 2) Distribution tab
with tab_distribution:
    left, right = st.columns(2)

    with left:
        st.subheader(f"Histogram — {numeric_col}")
        @st.cache_data
        def get_hist(data, col):
            fig = px.histogram(
                data,
                x=col,
                color="species",
                barmode="overlay",
                nbins=20,
                title=f"Distribution of {col}"
            )
            fig.update_layout(legend_title_text="Species")
            return fig

        fig_hist = get_hist(filtered_df, numeric_col)
        st.plotly_chart(fig_hist, use_container_width=True)

       

    with right:
        st.subheader("Pie Chart — Species Distribution (Filtered)")
        species_counts = filtered_df["species"].value_counts()
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=species_counts.index.tolist(),
                    values=species_counts.values,
                    hoverinfo="label+percent",
                    textinfo="label+percent"
                )
            ]
        )
        fig_pie.update_traces(hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

# 3) Overview tab
with tab_overview:
    st.subheader("Dataset Overview & Advanced Plots")
    st.markdown("Use these to inspect relationships and correlations across all features.")

    if show_heatmap:
        st.markdown("**Correlation Heatmap**")
        @st.cache_data
        def get_corr_heatmap(data):
            corr = data.iloc[:, :-1].corr()
            fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
            return fig

        fig_corr = get_corr_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)

    if show_pairplot:
        st.markdown("**Pairplot (This may take a little time)**")
        # Seaborn pairplot can't be returned as a plotly figure; use matplotlib backend
        @st.cache_data
        def get_pairplot(data):
            sns.set(style="ticks")
            pair_grid = sns.pairplot(data, hue="species", corner=False, markers=["o", "s", "D"])
            return pair_grid

        try:
            pair_grid = get_pairplot(df)
            st.pyplot(pair_grid)
        except Exception as e:
            st.error("Pairplot could not be generated. If running on Streamlit Cloud, it may be slow or resource-limited.")
            st.exception(e)

    
    with st.expander("Filtered data"):
        st.dataframe(filtered_df.reset_index(drop=True))


