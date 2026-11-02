import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

st.set_page_config(page_title="Workout Recommendation System", page_icon="💪", layout="centered")

st.title("💪 Workout Recommendation System")
st.write("Get workout suggestions using **K-Nearest Neighbors (KNN)** and **K-Means clustering**.")
st.caption("Educational demo: recommendations are based on a small sample dataset, not medical advice.")

@st.cache_data
def load_data():
    data = pd.read_csv("workouts.csv")
    return data

try:
    data = load_data()
except Exception:
    st.error("Could not find workouts.csv. Keep it in the same folder as app.py.")
    st.stop()

categorical_features = ["goal", "level", "equipment"]
numeric_features = ["duration_min"]
features = categorical_features + numeric_features

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", StandardScaler(), numeric_features),
    ]
)

X = preprocessor.fit_transform(data[features])

# K-Means groups workouts with similar input profiles.
cluster_count = min(4, len(data))
kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
data["cluster"] = kmeans.fit_predict(X)

# KNN finds workouts nearest to the user's preferences.
knn = NearestNeighbors(n_neighbors=min(5, len(data)), metric="euclidean")
knn.fit(X)

st.subheader("Enter your preferences")
with st.form("preferences"):
    goal = st.selectbox("Fitness goal", sorted(data["goal"].unique()))
    level = st.selectbox("Fitness level", ["Beginner", "Intermediate", "Advanced"])
    duration = st.slider("Available time (minutes)", min_value=10, max_value=60, value=30, step=5)
    equipment = st.selectbox("Available equipment", sorted(data["equipment"].unique()))
    submitted = st.form_submit_button("Recommend workouts")

if submitted:
    user_input = pd.DataFrame([{
        "goal": goal,
        "level": level,
        "duration_min": duration,
        "equipment": equipment,
    }], columns=features)

    user_vector = preprocessor.transform(user_input)
    distances, indices = knn.kneighbors(user_vector)
    recommendations = data.iloc[indices[0]].copy()
    recommendations["similarity_distance"] = distances[0]
    recommendations = recommendations.sort_values("similarity_distance")

    # Show the cluster assigned to the user's preference profile.
    user_cluster = int(kmeans.predict(user_vector)[0])
    st.success(f"Found recommendations! Your preference profile belongs to cluster {user_cluster + 1}.")
    st.subheader("Recommended workouts")
    display_cols = ["workout", "goal", "level", "duration_min", "equipment", "intensity", "estimated_calories"]
    for _, item in recommendations.head(5).iterrows():
        with st.container(border=True):
            st.markdown(f"### {item['workout']}")
            st.write(f"**Goal:** {item['goal']}  |  **Level:** {item['level']}")
            st.write(f"**Duration:** {int(item['duration_min'])} minutes  |  **Equipment:** {item['equipment']}")
            st.write(f"**Intensity:** {item['intensity']}  |  **Estimated calories:** {int(item['estimated_calories'])}")
    st.info("Choose a comfortable intensity, warm up first, and stop if you feel pain or unwell. Calorie estimates are approximate.")

with st.expander("About the dataset and algorithms"):
    st.write(f"The included sample dataset contains {len(data)} workout records.")
    st.write("**KNN:** compares your preferences with workout records and returns the closest matches.")
    st.write("**K-Means:** groups workout records with similar feature patterns into clusters.")
    st.write("**Preprocessing:** one-hot encodes categories and scales duration before distance-based modeling.")
    st.dataframe(data.drop(columns=["cluster"]), width="stretch")
