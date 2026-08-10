import numpy as np
import pandas as pd
from numpy.linalg import norm
from copy import deepcopy

def cosine_similarity_movie(matrix):
    norms = norm(matrix, axis = 0)
    """ axis = 0 is walking down each column
        norm is taking the sqrt of the sum of the square with each item in the column"""
    norms[norms == 0] = 1e-10
    # use 1e-10 = 0.0000000001 replace 0 to make sure the data is worked
    normalised = matrix/norms
    similarity_matrix = np.dot(normalised.T, normalised)
    # (n-movie * n-users) @ (n-users * n-movie) = (n-movie * n-movie)
    return similarity_matrix

def cosine_similarity_user(matrix):
    norms = norm(matrix, axis = 1)
    norms[norms == 0] = 1e-10
    normalised = matrix/norms[:, np.newaxis]
    # comparing to the users, not the movie.
    similarity_matrix = np.dot(normalised, normalised.T)
    return similarity_matrix

#cleaning movie df
movie_df = pd.read_csv("data/movies.csv")
genres_split = movie_df['genres'].str.get_dummies(sep='|')
extracted = movie_df['title'].str.extract(r'^(?P<clean_title>.*?)\s*\((?P<year>\d{4})\)\s*$')
movie_df['clean_title'] = extracted['clean_title']
movie_df['year'] = extracted['year'].astype('Int64') 
movie = pd.concat([movie_df['movieId'], genres_split])
movie.fillna(movie.fillna(0))
movies = movie.fillna(0)

#cleaning ratings df
ratings_df = pd.read_csv("data/ratings.csv")
ratings = ratings_df.pivot(index = 'userId', columns='movieId', values = 'rating')
ratings = ratings.fillna(0)
counts = (ratings > 0).sum()
ratings = ratings.loc[:, counts >= 5]

#convert both movies and ratings df into numpy
movie_arr = movies.to_numpy()
ratings_arr = ratings.to_numpy()

#get the consine simiarlity
sim_matrix = cosine_similarity_movie(ratings_arr)
