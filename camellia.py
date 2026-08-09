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

"""
def evaluate(R, seed = 42, test_frac = 0.2):
    rng = np.random.default_rng(seed)

    users, items = R.nonzero()
    pairs = np.column_stack([users, items])

    idx = rng.permutation(len(pairs))
    n_test = int(test_frac * len(pairs))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    R_train = deepcopy(R)
    test_set=[]
    for k in test_idx:
        u, i = pairs[k]
        test_set.append((u,i, (R_train[u, i])))

        R_train[u,i] = 0

    empty_users = (R_train.sum(axis=1) == 0).sum()
    empty_items = (R_train.sum(axis=0) == 0).sum()

    print(f"held out {len(test_set)} ratings | empty users: {empty_users}, empty items: {empty_items}")

    return R_train, test_set

def matrix_factorisation_bias(R, K=20, lr=0.01, reg=0.05, epochs=30, seed=42):
    n_users, n_items = R.shape
    rng = np.random.default_rng(seed)

    P = rng.normal(0, 0.1, size=(n_users, K))
    Q = rng.normal(0, 0.1, size=(n_items, K))
    b_u = np.zeros(n_users)          # biases START at zero, not random
    b_i = np.zeros(n_items)
    mu  = R[R > 0].mean()            # global mean of OBSERVED ratings — computed once

    user_pos, item_pos = R.nonzero()
    samples = list(zip(user_pos, item_pos))

    for epoch in range(epochs):
        rng.shuffle(samples)
        for u, i in samples:
            e = R[u, i] - (mu + b_u[u] + b_i[i] + P[u]@Q[i])

            P_u = deepcopy(P[u])
            P[u] = P[u] + lr * (e * Q[i] - reg * P[u])
            Q[i] = Q[i] + lr * (e * P_u - reg * Q[i])

            b_u[u] = b_u[u] + lr * (e-reg*b_u[u])
            b_i[i] = b_i[i] + lr * (e-reg*b_i[i])

        preds = np.array([mu + b_u[u] + b_i[i] + P[u] @ Q[i] for u, i in samples])
        truth = np.array([R[u, i] for u, i in samples])
        rmse = np.sqrt(np.mean((truth - preds) ** 2))
        print(f"epoch {epoch:2d}  RMSE={rmse:.4f}")

    return P, Q, b_u, b_i, mu
"""

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

"""
R_train, test_set = evaluate(ratings_arr)
P, Q, b_u, b_i, mu = matrix_factorisation_bias(R_train, K=40, epochs=20)
preds = np.clip(
    np.array([mu + b_u[u] + b_i[i] + P[u] @ Q[i] for (u, i, _) in test_set]),
    0.5, 5.0)
truth = np.array([r for (_, _, r) in test_set])
test_rmse = np.sqrt(np.mean((truth - preds) ** 2))

print(movie_df)

"""