# Movie Recommender System from the MovieLens 100K dataset

goal of this project is to demonstrate the skills of raw data interaction and turn it into a working recommender by understanding the mechanism

## Methodology
- Data acquisition and cleaning (pandas): loading the raw files, checking for duplicate ratings, missing values and inconsistent identifiers and filtering out users and films with very few interactions.
- Interaction matrix: pivoting the cleaned frame into a user × item matrix, then converting to a NumPy array for the numerical work.
- Similarity from first principles: cosine similarity written directly in NumPy.
- Neighbourhood-based collaborative filtering: top-N recommendations generated from the similarity matrices.
- Matrix factorisation: SGD-based, implemented by hand, with and without user and item bias terms.
- Evaluation: a held-out test split, reported as RMSE.
- Visualisation: Streamlit, chosen so that attention could go to the data rather than to tuning CSS by hand.

Live demonstration: https://camellia-ptocznwdkupmsy3dfbkkze.streamlit.app/


## Result
Rating prediction (RMSE on the held-out test set)

|   Model   |   RMSE    |
|-----------|-----------|
|Global mean baseline   |   1.0279  |
|User mean + item bias baseline    |    0.8815  |
|Matrix factorisation (K = 40, 20 epochs)   |   0.8531  |
|Matrix factorisation with bias (K = 40, 20 epochs)     |   0.8456  |

Adding bias terms reduces the error from 0.8531 to 0.8456. The improvement comes from separating out systematic tendencies — some users rate generously, some films are simply well liked — so that the latent factors are free to model genuine taste interactions rather than absorbing these offsets.

## What I learnt
This project consolidated my pandas data-cleaning work and, more usefully, forced me to distinguish between cosine similarity and Pearson correlation by implementing both rather than calling them. Testing notebook output with ipytest was new to me and proved a good discipline. The factorisation and evaluation stages took the most time and patience to understand, but they were also where the mechanics of the algorithm finally became concrete.

## Limitations and future work
The most significant limitation is the treatment of unobserved ratings in the similarity computation. Filling absent entries with zero confused "has not seen this film" with "rated this film zero", which almost certainly inflates the apparent dissimilarity between users with little overlap in viewing history. A more defensible approach would restrict the comparison to co-rated items, or use mean-centred values so that an absent rating carries no signal.

The dataset is also small and static: 610 users and 9,742 films is a fraction of the scale at which recommender systems are usually deployed, and the cold-start problem — how to serve a user or film with no ratings at all — is not addressed here.

Looking ahead, I intend to apply the same techniques to a larger dataset and in a different fields, most likely music recommendation, and to add ranking metrics alongside RMSE, since rating accuracy and the quality of a top-N list are not the same thing.

## Installation

Requires Python 3.12.

1. Clone the project and enter the directory
```
git clone https://github.com/Her304/camellia.git
cd camellia
```

2. (Optional but recommended) create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

3. Install the requirements
```
pip install -r requirements.txt
```

4. Run Streamlit to view the visual results — from the repo root, as the
   data is loaded via relative paths
```
streamlit run Hello.py
```

To run the notebook tests as well: `pip install -r requirements-dev.txt`

## Acknowledgements
Kaggle MovieLens 100k dataset: https://www.kaggle.com/datasets/abhikjha/movielens-100k?resource=download

### Note on the dataset
This project uses ml-latest-small (610 users, 9,742 films, 100,836 ratings), not the classic MovieLens 100K (943 users, 1,682 items, exactly 100,000 ratings). The Kaggle upload used here is labelled "MovieLens 100k", but the file contents are those of ml-latest-small. The distinction matters when comparing RMSE against published baselines, so the correct name is used throughout.