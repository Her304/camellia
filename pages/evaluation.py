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

    RELEVANT = 4.0

    mask = R_train > 0
    mu = R_train[mask].mean()
    n_u, n_i = mask.sum(1), mask.sum(0)
    user_mean = np.divide(R_train.sum(1), n_u, out=np.full(R_train.shape[0], mu), where=n_u > 0)
    item_mean = np.divide(R_train.sum(0), n_i, out=np.full(R_train.shape[1], mu), where=n_i > 0)
    b_u, b_i = user_mean - mu, item_mean - mu

    denom = (R_train > 0) @ np.abs(sim_matrix)
    denom[denom == 0] = 1e-10
    cf_scores = (R_train @ sim_matrix) / denom

    _grouped = {}
    for u, i, r in test_set:
        _grouped.setdefault(u, []).append((i, r))
    by_user = {u: (np.array([i for i, _ in v]), np.array([r for _, r in v]))
            for u, v in _grouped.items()}

    def precision_at_k(score_fn, K=5, relevant=RELEVANT): 
        precisions, recalls = [], []
        for u, (items, truth) in by_user.items():
            rel = truth >= relevant
            if len(items) < K or not rel.any():
                continue
            top = np.argsort(-np.asarray(score_fn(u, items), dtype=float))[:K]
            precisions.append(rel[top].sum() / K)
            recalls.append(rel[top].sum() / rel.sum())
        return np.mean(precisions), np.mean(recalls), len(precisions)


    rng = np.random.default_rng(0)
    models = {
        "Random":                 lambda u, items: rng.random(len(items)),
        "Popularity (item mean)": lambda u, items: item_mean[items],
        "User mean + item bias":  lambda u, items: mu + b_u[u] + b_i[items],
        "Item-item cosine":       lambda u, items: cf_scores[u, items],
        "MF (K=40)":              lambda u, items: P[u] @ Q[items].T,
        "MF + bias (K=40)":       lambda u, items: mu + b_u[u] + b_i[items] + P[u] @ Q[items].T,
    }

    def precision_table(K):
        rows = []
        for name, fn in models.items():
            p, r, n = precision_at_k(fn, K)
            rows.append({"Model": name, "precision": p, "recall": r, "users": n})
        return pd.DataFrame(rows)

    st.divider()
    st.write("Precision@K is the fraction the user actually liked of the K movies user recommend")
    st.write(f"Precision@{K_n}")
    st.dataframe(
        precision_table(K_n),
        hide_index=True,
        use_container_width=True,
        column_config={
            "precision": st.column_config.NumberColumn(f"P@{K_n}", format="%.4f"),
            "recall":    st.column_config.NumberColumn(f"R@{K_n}", format="%.4f"),
            "users":     st.column_config.NumberColumn(
                "users scored", help="Users with at least K held-out ratings and at least one rated >= 4"),
        },
    )