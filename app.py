import streamlit as st
import pickle as pk
import requests as rq
import streamlit.components.v1 as components

# Function to fetch movie posters
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=29704a4bf8366980a1aa2001dbb33a04&language=en-US".format(movie_id)
    data = rq.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Load movies and similarity matrices from pickle files
movies = pk.load(open("movies_list.pkl",'rb'))
similarity = pk.load(open("similarity.pkl",'rb'))
movies_list = movies['title'].values

# Streamlit UI components
st.header("Movies Recommendation System")
selected_movie = st.selectbox("Select a movie:", movies_list)

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    recommended_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# Display recommendations when button is clicked
if st.button("Show Recommendation"):
    movie_names, movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for col, movie_name, movie_poster in zip(cols, movie_names, movie_posters):
        with col:
            st.text(movie_name)
            st.image(movie_poster)
