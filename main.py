import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset
movies = pd.read_csv('dataset.csv')

# Display dataset summary statistics and info
print(movies.describe())
print(movies.info())
print(movies.isnull().sum())
print(movies.columns)

# Filter the DataFrame to include only the specified columns
movies = movies[['id', 'title', 'overview', 'genre']]

# Handle missing values
movies['overview'] = movies['overview'].fillna('')
movies['genre'] = movies['genre'].fillna('')

# Create a new column 'tags' by combining 'overview' and 'genre'
movies['tags'] = movies['overview'] + ' ' + movies['genre']

# Drop the 'overview' and 'genre' columns
new_data = movies.drop(columns=['overview', 'genre'])

# Initialize the CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')

# Fit and transform the 'tags' column
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()

# Compute the cosine similarity matrix
similarity = cosine_similarity(vector)

# Print the first 5 rows of the new_data DataFrame
print(new_data.head())

# Print the shape of the similarity matrix to verify it
print(similarity.shape)

# Function to recommend movies
def recommend(movie_title, num_recommendations=5):
    if movie_title not in new_data['title'].values:
        return f"Movie '{movie_title}' not found in the dataset."
    
    # Get the index of the movie that matches the title
    idx = new_data[new_data['title'] == movie_title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar movies
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top most similar movies
    return new_data['title'].iloc[movie_indices].tolist()

# Example usage
print(recommend('Iron Man', 5))

# Save the DataFrame and similarity matrix to pickle files
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Load the DataFrame and similarity matrix from pickle files
loaded_movies_list = pickle.load(open('movies_list.pkl', 'rb'))
loaded_similarity = pickle.load(open('similarity.pkl', 'rb'))

# Verify loaded data
print(loaded_movies_list.head())
print(loaded_similarity.shape)
