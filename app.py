import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ---------------- #

tmdb = pd.read_csv("data/tmdb_5000_movies.csv")

tmdb = tmdb[['title', 'overview']].dropna()

# ---------------- TF-IDF ---------------- #

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(tmdb['overview'])

# ---------------- COSINE SIMILARITY ---------------- #

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------- TITLE INDICES ---------------- #

indices = pd.Series(tmdb.index, index=tmdb['title']).drop_duplicates()

# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie_title):

    if movie_title not in indices:
        return ["Movie not found"]

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return tmdb['title'].iloc[movie_indices].tolist()

# ---------------- STREAMLIT UI ---------------- #

st.title("🎬 Movie Recommendation System")

movie_list = tmdb['title'].values

movie_name = st.selectbox(
    "Select a movie",
    movie_list
)

if st.button("Recommend"):

    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies:")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")