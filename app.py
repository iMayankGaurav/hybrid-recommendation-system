import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#API DATA FETCHING

API_KEY = ""


def fetch_poster(movie_title):

    try:

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"

        response = requests.get(url, timeout=10)

        data = response.json()

        if data['results']:

            poster_path = data['results'][0].get('poster_path')

            if poster_path:

                full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

                return full_path

    except Exception as e:

        print("Error fetching poster:", e)

    return None

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

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):

        poster = fetch_poster(movie)

        with cols[idx]:

            st.text(movie)

            if poster:
                st.image(poster)
            else:
                st.write("Poster not available")