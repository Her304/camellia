# Movie Recommender System from the MovieLens 100K dataset

goal of this project is to demonstrate the skills of raw data interaction and turn it into a working recommender by understanding the mechanism

From this project, I am consolidating my skills in data cleansing with pandas and have also learnt to apply and distinguish between cosine similarity and Pearson correlation. Meanwhile, using ipytest, I can test whether the results match expectations, which is new to me in this project. In addition, though the building process on recommender, matrix factorisation and evaluation cost me time and patience to understand the mechanism, it is still a good chance for me to understand how the recommendation algorithm was built.

For the visualisation parts, I chose Streamlit to help me visualise and demonstrate the result. With the decent preset visual style and the optimisation of dataset visualisation, I can focus on the data demonstration rather than tuning the webpage CSS style line by line. 

## Result
For the matrix factorisation part, I set the K value to 40 and epochs to 20, the test RMSE is 0.8531, and for the matrix factorisation with bias part, the K value and epochs are the same as above, the test RMSE is reduced to 0.8456.

## Limitations and future
For limitations, even though it is for practice purposes, the limitations of this project still exist. For the datasets, because I am using the 100k dataset, only 610 users, 9742 movies and 100836 ratings, it is obvious that the dataset is tiny and frozen in this project. On the other hand, all the null values have been set to 0/1e-10, leading the algorithm to read the ratings from “have not seen the movie” to “the user rates the movie as zero”, it may gain some levels of inaccuracies from the result. 

For the future, I expect myself to apply the recommendation algorithm by myself in various situations, e.g., a song recommendation system, with a larger dataset.

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


