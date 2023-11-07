import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

app = Flask(__name__)

movies_data = pd.read_csv('movies.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']

    # Rest of your recommendation logic (from movie_name to recommendations)
    # movie_name = input('Enter your favourite movie name: ')

    list_of_all_titles = movies_data['title'].str.lower().tolist()

    find_close_match = difflib.get_close_matches(movie_name.lower(), list_of_all_titles)

    if len(find_close_match) > 0:
        close_match = find_close_match[0]

        m = movies_data.title.str.lower()
        index_of_the_movie = movies_data[m == close_match]['index'].values[0]


        selected_features = ['genres','keywords','tagline','cast','director']
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
    
        combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectors)
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        print('Movies suggested for you:')

        i = 1
        recommendations = []

        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            if i < 30:
                print(i, '.', title_from_index)
                recommendations.append((i, title_from_index))
                i += 1
    else:
        print('No close match found for the given movie name.')

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
