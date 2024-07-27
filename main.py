import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_excel(r'C:\songs_records.excel.xlsx')

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN values in the 'Genre' column with an empty string
data['Genre'] = data['Genre'].fillna('')

# Construct the TF-IDF matrix based on the 'Genre' column
tfidf_matrix = tfidf.fit_transform(data['Genre'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend songs based on genre
def recommend_songs_based_on_genre(song_name, cosine_sim=cosine_sim):
    # Convert input song name to lowercase
    song_name_lower = song_name.lower()
    
    # Find the index of the song with the given name
    idx = data[data['SONG NAME'].str.lower() == song_name_lower].index
    if len(idx) == 0:
        print("Sorry, the song is not found in the database.")
        return
    
    # Get the index of the first match
    idx = idx[0]
    
    # Get the pairwise similarity scores of all songs with that song
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar songs
    sim_scores = sim_scores[1:11]

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar songs
    return data.iloc[song_indices]

# Ask the user for input
song_name = input("Enter the name of the song: ")

# Recommend songs similar to the input song
recommendations = recommend_songs_based_on_genre(song_name)
if recommendations is not None:
    print("\nRecommended Songs:")
    print(recommendations[['SONG NAME', 'artist name', 'Genre']])
