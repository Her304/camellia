import streamlit as st
import pandas as pd
import numpy as np
from camellia import ratings, movie_df, ratings_arr, sim_matrix
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode

st.set_page_config(
    page_title="recommendation",
    page_icon="🌸",
)

TITLES = movie_df.set_index("movieId")[["title", "genres"]]

@st.cache_data
def recommender(user, N=10):
    r = ratings_arr[ratings.index.get_loc(user)]
    scores = r @ sim_matrix
    denorm = (r>0) @ np.abs(sim_matrix)
    denorm[denorm == 0] = 1e-10
    scores = scores / denorm
    scores[r > 0] = -np.inf
    ranks = np.argsort(-scores)[:N]
    return TITLES.loc[ratings.columns[ranks]]

@st.cache_data
def top_rated(user, n=5):
    r = ratings.loc[user]
    return TITLES.loc[r[r > 0].sort_values(ascending=False).head(n).index]

@st.cache_data
def recommend_for_picks(picks, N=10):
    pos = ratings.columns.get_indexer(list(picks))
    known = pos != -1
    if not known.any():
        return None
    r = np.zeros(len(ratings.columns))
    r[pos[known]] = np.array(list(picks.values()), dtype=float)[known]
    scores = r@sim_matrix
    denom = (r>0) @ np.abs(sim_matrix)
    denom[denom == 0] =1e-10
    scores = scores/denom
    scores[r>0] = -np.inf
    ranks = np.argsort(-scores)[:N]
    return movie_df.set_index("movieId").loc[ratings.columns[ranks], ["title", "genres"]]

st.title("Movie recommendation")
st.subheader("get recommendations with one user")

user_id = st.selectbox("User", ratings.index)
n = st.slider("How many recommendations?", 5, 25, 10)

st.subheader(f"What user {user_id} already liked")
st.dataframe(top_rated(user_id), use_container_width=True)

st.subheader("Recommended for them")
st.dataframe(recommender(user_id, n), use_container_width=True)

st.divider()

st.subheader("recommend for your movie picks")
display_df = movie_df[["movieId", "title", "year"]]
gb = GridOptionsBuilder.from_dataframe(display_df)

# configure selection
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()
gridOptions = gb.build()

data = AgGrid(display_df,
              gridOptions=gridOptions,
              enable_enterprise_modules=True,
              allow_unsafe_jscode=True,
              update_mode=GridUpdateMode.SELECTION_CHANGED,
              columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

selected_movies = data["selected_rows"]

if selected_movies is not None and len(selected_movies) > 0:
    st.write("Rate each pick, then get recommendations based on them.")

    picks = {}
    for _, row in selected_movies.iterrows():
        picks[row["movieId"]] = st.slider(
            row["title"], 0.5, 5.0, 3.0, 0.5, key=f"rate_{row['movieId']}"
        )

    result = recommend_for_picks(picks, n)
    if result is None:
        st.warning("None of your picks have enough ratings to recommend from. Try others.")
    else:
        st.subheader("Your result")
        st.dataframe(result, use_container_width=True)
