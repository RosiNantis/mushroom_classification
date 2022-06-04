"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np
#from fuzzywuzzy import process
# rapidfuzz package is a faster version of fuzzywuzzy
from rapidfuzz import process
import pickle
from sklearn.impute import SimpleImputer


#movies = pd.read_csv('movies_ratings.csv', index_col=0)
umrT = pd.read_csv('../data/ml-latest-small/ratings.csv', na_values = 'Nan')
mtg = pd.read_csv('../data/ml-latest-small/movies.csv', na_values = 'Nan')
# merge the two frames based on the column movieid
movies = pd.merge(umrT, mtg, on='movieId')

methods_recommendation = ['random','NMF','user_similarity']

# load model
with open('../10.3.collaborative_filtering/model6_8h.pkl', 'rb') as f:
    model= pickle.load(f)


def get_movie_frame(method = methods_recommendation, umrT=umrT, mtg = mtg):

    if method == 'NMF':
        """
        i will get a Data Frame with movieId Title and userId 
        pivoted in a matrix with NaN where user has no rating
        """
        # use pivot to make the matrix of movie rates
        rates =umrT.pivot(index='userId',columns = 'movieId')
        # Split the movie name from movie year and apply it in the matrix
        mtg['title_new'] = mtg.title.astype(str).str[:]
        # Try to zip columns with movie names
        #new_columns = dict(zip(df.movieId,df_movie.title))
        rates.rename(columns=dict(zip(mtg["movieId"], mtg["title_new"])), inplace = True)
        movies = rates.rating
    elif method == 'detailed_table':
        """
        i will get a Data Frame with movieId Title average ratings for per user.
        Final outcome is a frame with columns:# 
        movieID, userID, avg(rate), title, genres, accurate(rate)
        """
        umrT_av_rat = umrT.set_index('movieId').groupby(['movieId']).mean()
        # merge the two frames based on the column movieid
        movies_merge = pd.merge(umrT, mtg, on='movieId')
        # merge the two frames based on the column movieid to find average rating
        movieId_rating = pd.merge(umrT_av_rat, mtg, on='movieId')
        # merge movieID, userID, avg(rate), title, genres, accurate(rate)
        mov = pd.merge(movieId_rating,movies_merge, on='movieId', how = 'left')[["movieId", "rating_x","title_x","userId_y","genres_x","rating_y"]]
        mov.columns = [["movieId", "rating_avg","title","userId","genres","rating_acc"]]
        # reset index
        movies = mov
    elif method == 'user_similarity':
        """
        i will get a Data Frame with movieId Title ratings per user.
        """
        # use pivot to make the matrix of movie rates
        rates =umrT.pivot(index='userId',columns = 'movieId')
        rates.rename(columns=dict(zip(mtg["movieId"], mtg["title"])),inplace = True)
        movies = rates.rating
    return movies

def match_movie_title(input_title, movie_titles):
    """
    Matches inputed movie title to existing one in the list with fuzzywuzzy
    """
    matched_title = process.extractOne(input_title, movie_titles)[0]

    return matched_title

def print_movie_titles(movie_titles):
    """
    Prints list of movie titles in cli app
    """    
    for movie_id in movie_titles:
        print(f'            > {movie_id}')


def create_user_vector(user_rating,movies):
    """
    Convert dict of user_ratings to a user_vector
    """       
    ##### add a new user for the NMF method with movies and ratings and NaN else #######

    # ------------------------------------------------------------ # 
    user = pd.DataFrame(user_rating, index=[0])
    user_t = user.T.reset_index()
    # list of the entry movies
    user_movie_entries = list(user_t["index"])
    # list of the entry movies ratings
    user_rate_entries = list(user_t[0])
    #list of the movie titles of library
    # movies = get_movie_frame(method = 'NMF')
    movie_titles = list(movies.columns)
    # # matches the movies from user with the library
    intended_movies = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    # # create a frame with one user
    user_new = pd.DataFrame(movies.loc[1].copy())
    user_new.columns = [['0']]
    user_new[['0']] = np.nan
    for mov in user_new.index:
        for idx, int_mov in enumerate(intended_movies):
            if mov == int_mov:
                user_new.loc[int_mov] = user_rate_entries[idx]
    new_user = user_new.T
    return new_user


def lookup_movieId(movies, movieId):
    """
    Convert output of recommendation to movie title
    """
    # match movieId to title
    movies = movies.reset_index()
    boolean = movies["movieId"] == movieId
    movie_title = list(movies[boolean]["title"])[0]
    return movie_title

def clean_nan_numbers(data):
    """
    clean a data frame from NaN by using the mean value
    """
    imputer = SimpleImputer(strategy = 'mean') # add the NaN with the average of movie recommendations 
    Rtrue = imputer.fit_transform(data)
    return Rtrue, imputer
