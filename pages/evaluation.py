import streamlit as st
import pandas as pd
import numpy as np
from camellia import ratings, movie_df, ratings_arr, sim_matrix
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from copy import deepcopy

st.set_page_config(
    page_title="evaluation",
    page_icon="🌸",
)

@st.cache_data
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

    st.write(f"held out {len(test_set)} ratings | empty users: {empty_users}, empty items: {empty_items}")

    return R_train, test_set

@st.cache_data
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
        st.write(f"epoch {epoch:2d}  RMSE={rmse:.4f}")

    return P, Q, b_u, b_i, mu

st.title('Evaluation')
st.write(
"""
On this page, the model predicts each rating as the sum of four terms:

1. **Global mean** `μ` (or `mu` in the program): the average rating across the entire dataset, serving as the baseline.
2. **User bias** `b_u[u]`: how generously or harshly this particular user tends to rate.
3. **Item bias** `b_i[i]`: how favourably this particular film is received in general.
4. **Interaction term** `P[u] @ Q[i]`: the match between this user's latent taste and this film's latent characteristics.
"""
)

st.subheader("How many epochs?")
n = st.slider("Epcohs is to decide how many training test can go through, the ideal amount is 20", 5, 100, 20)
st.subheader("How many K?")
K_n = st.slider("""
K sets the width of the taste vectors,

a smaller K may capture the broad tendencies but cannot represent nuanced taste

a larger K gives the model more room but will filled with memorised noise as the `reg` is set to 0.05

the ideal K range should be within 30 - 40
""", 5, 100, 20)
if st.button("run"):
    R_train, test_set = evaluate(ratings_arr)
    st.divider()
    st.write("RMSE measures prediction errors, model accuracy, and average differences")
    P, Q, b_u, b_i, mu = matrix_factorisation_bias(R_train, K=K_n, epochs=n)
    preds = np.clip(
        np.array([mu + b_u[u] + b_i[i] + P[u] @ Q[i] for (u, i, _) in test_set]),
        0.5, 5.0)
    truth = np.array([r for (_, _, r) in test_set])
    test_rmse = np.sqrt(np.mean((truth - preds) ** 2))
    st.divider()
    st.write("The average RMSE range on this Matrix factorisation with biases model should be within 0.84 to 0.88")
    st.write(f"total RMSE: {test_rmse}")