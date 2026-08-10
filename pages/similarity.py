import streamlit as st
import pandas as pd
import numpy as np
from camellia import ratings, movie_df, sim_matrix
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def heat_map(movie_ids):
    pos = ratings.columns.get_indexer(movie_ids)   # -1 for movies not in the matrix
    missing = [m for m, p in zip(movie_ids, pos) if p == -1]
    if missing:
        st.warning(f"Not in the ratings matrix (fewer than 5 ratings): {missing}")
    pos = pos[pos != -1]

    if len(pos) < 2:
        st.info("Pick at least two movies.")
        return

    titles = movie_df.set_index("movieId")["title"]
    labels = [titles.loc[ratings.columns[p]] for p in pos]
    block = sim_matrix[np.ix_(pos, pos)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(block,
                    xticklabels=labels, yticklabels=labels,
                    annot=True, fmt=".2f",        # show the similarity value in each cell
                    cmap="viridis", ax=ax)
    ax.set_title("Item–item cosine similarity")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(
    page_title="similarity",
    page_icon="🌸",
)

st.title("Cosine similarity")
st.write("By selecting the movie below and click 'run' button, the program is checking the similarity of the movie you have selected")
# select the columns you want the users to see
gb = GridOptionsBuilder.from_dataframe(movie_df[["movieId", "title", "year"]])

# configure selection
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()
gridOptions = gb.build()

data = AgGrid(movie_df,
              gridOptions=gridOptions,
              enable_enterprise_modules=True,
              allow_unsafe_jscode=True,
              update_mode=GridUpdateMode.SELECTION_CHANGED,
              columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

selected_rows = data["selected_rows"]

if selected_rows is not None:
    if len(selected_rows) != 0:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Movie ID")
            st.markdown("\n".join(f"- :orange[{m}]" for m in selected_rows["movieId"]))

        with col2:
            st.markdown("##### Movie Title")
            st.markdown("\n".join(f"- :orange[{t}]" for t in selected_rows["title"]))

if selected_rows is not None and len(selected_rows) > 0:
    st.write("click 'run' button and see the similarity of the movie you have selected")
    if st.button("run"):
        heat_map(selected_rows["movieId"].tolist())

