import streamlit as st 
import pandas as pd
import ast 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch

# 🔑 Your SerpAPI Key (keep secret when deploying publicly)
SERPAPI_API_KEY = "fde446646e13a9f99114b9883112b2d9366509e7bc9720516d7f186559d1d6e2"

# ----- SerpAPI Poster Fetcher -----
@st.cache_data(show_spinner=False)
def get_poster_from_serpapi(movie_title):
    try:
        params = {
            "q": f"{movie_title} movie poster",
            "tbm": "isch",
            "api_key": SERPAPI_API_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        images = results.get("images_results", [])
        return images[0]["original"] if images else None
    except Exception:
        return None

# ----- Load & Clean Data -----
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    def extract_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres]
        except:
            return []

    df['genres'] = df['genres'].apply(extract_genres)
    df = df[['original_title', 'genres', 'overview']].dropna().reset_index(drop=True)
    return df

movies = load_data()

# ----- Build Similarity Matrix -----
@st.cache_resource
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_to_index = pd.Series(movies.index, index=movies['original_title'].str.lower())
    return cosine_sim, title_to_index

cosine_sim, title_to_index = compute_similarity(movies)

# ----- Recommender Functions -----
def recommend_by_genres(selected_genres, top_n=5):
    filtered = movies[movies['genres'].apply(lambda g: any(genre in g for genre in selected_genres))]
    return filtered[['original_title', 'genres']].head(top_n)

def recommend_similar_movies(movie_title, top_n=5):
    title = movie_title.lower()
    if title not in title_to_index:
        return f"❌ Movie '{movie_title}' not found in dataset."
    
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return movies[['original_title', 'genres']].iloc[sim_indices]

# ----- Streamlit UI -----
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.markdown("Select genres or enter a movie title to get recommendations.")

# Genre Selection
all_genres = sorted({genre for sublist in movies['genres'] for genre in sublist})
selected_genres = st.multiselect("📂 Select Genres:", all_genres)

# Movie Title Input
movie_title = st.text_input("🎥 Enter a Movie Title (optional):")

# Recommend Button
if st.button("Recommend"):
    if selected_genres:
        with st.spinner("🔍 Finding genre-based recommendations..."):
            st.subheader("🎯 Genre-Based Recommendations")
            genre_results = recommend_by_genres(selected_genres)
            for _, row in genre_results.iterrows():
                poster_url = get_poster_from_serpapi(row['original_title'])
                if poster_url:
                    st.image(poster_url, width=150)
                else:
                    st.write("🖼️ Poster not found.")
                st.markdown(f"**{row['original_title']}** — {', '.join(row['genres'])}")
    
    if movie_title.strip():
        with st.spinner("🤖 Searching for similar movies..."):
            st.subheader("🤖 Content-Based Recommendations")
            similar = recommend_similar_movies(movie_title)
            if isinstance(similar, str):
                st.warning(similar)
            else:
                for _, row in similar.iterrows():
                    poster_url = get_poster_from_serpapi(row['original_title'])
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.write("🖼️ Poster not found.")
                    st.markdown(f"**{row['original_title']}** — {', '.join(row['genres'])}")

    if not selected_genres and not movie_title.strip():
        st.info("⚠️ Please select at least a genre or enter a movie title.")
