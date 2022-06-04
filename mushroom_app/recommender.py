"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import  match_movie_title, model, create_user_vector, get_movie_frame, clean_nan_numbers
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity


def recommend_random(movies, user_rating, k=5):
    """
    return k random unseen movies for user 
    """
    # makes a frame for the external user ratings of the movies(features)
    user = pd.DataFrame(user_rating, index=[0])
    # rearrange the frame as movies and ratings two columns, movies and ratings
    user_t = user.T.reset_index()
    # list of the entry movies
    user_movie_entries = list(user_t["index"])
    # list of the movie titles of library
    movie_titles = list(movies["title"])
    # matches the movies from user with the library
    intended_movies = [match_movie_title(title, movie_titles) for title in user_movie_entries]
    
    # convert these movies to intended movies and convert them into movie ids
    recommend = movies.copy()
    recommend = recommend.reset_index()
    recommend = recommend.set_index("title")
    recommend.drop(intended_movies, inplace=True)
    random_movies = np.random.choice(list(recommend.index), replace=False, size=k)
    return random_movies  


    
def recommend_with_NMF(movies ,new_user, model=model, k=5):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds
    """
    # cncatenate new user with database
    table = pd.concat([new_user, movies], axis = 0,ignore_index=True) 
    # ------------------------------------------------------------#  
    #  dEal with missing values with Imputer
    packet= clean_nan_numbers(table)
    clean_table=packet[0]
    imputer = packet[1]

    # ------------------------------------------------------------#  
    # take Q and P matrices
    Q = model.components_
    P = model.transform(clean_table)
    # ------------------------------------------------------------#  
    # locate new user and give an array of rates with Imputed values
    user = table.iloc[0,:].values
    user = user.reshape(1, -1)
    # ------------------------------------------------------------#
    # predict user P, R values
    user_clean = imputer.transform(user)
    user_P = model.transform(user_clean) 
    user_R = np.dot(user_P,Q)
    # ------------------------------------------------------------#
    # remove seen movies & give top n recommendations   
    recommendation = pd.DataFrame({'user_input':user[0], 'predicted_ratings':user_R[0]}, index = table.columns)
    recommendation = recommendation[recommendation['user_input'].isna()].sort_values(by = 'predicted_ratings', ascending= False)
    NMF_movies = list(recommendation.iloc[:k].index)
    return NMF_movies



def recommend_with_user_similarity(new_user, movies, k=5):
    # ------------------------------------------------------------#  
    # combine new user with database
    tabl = pd.concat([new_user, movies], axis = 0,ignore_index=True) 
    # ------------------------------------------------------------#  
    # Drop duplicate movies from data frame
    table = tabl.T.groupby(level=0).first().T
    # ------------------------------------------------------------#  
    # fill in NaN values with zeros
    movie_CS_u = table.fillna(0)
    # ------------------------------------------------------------#  
    # We can turn this into a dataframe:
    cos_sim_table = pd.DataFrame(cosine_similarity(movie_CS_u),index=movie_CS_u.index, columns = movie_CS_u.index)
    # ------------------------------------------------------------#  
    # use the transposed version of R
    R_t = movie_CS_u.T.astype(int)
    # ------------------------------------------------------------#  
    # create a list of unseen movies for this user
    unseen_movies = list(R_t.loc[~(R_t!=0).all(axis=1)].index)
    # ------------------------------------------------------------#  
    # Create a list of top 3 similar user (nearest neighbours)
    neighbours = list(cos_sim_table.iloc[0].sort_values(ascending=False).index[1:4])
    # create the recommendation (predicted/rated movie)
    # ------------------------------------------------------------#  
    predicted_ratings_movies = []
    for idx, movie in enumerate(unseen_movies):
        # we check the users who watched the movie

        people_who_have_seen_the_movie = list(R_t.columns[R_t.loc[movie] > 0])
        num = 0
        den = 0
        for user in neighbours:
            # if this person has seen the movie
            if user in people_who_have_seen_the_movie:
            #  we want extract the ratings and similarities
                rating = R_t.loc[movie,user]
                similarity = cos_sim_table.loc[0,user]
                num += rating*similarity
                den += similarity

        if den != 0:
            predicted_ratings = num/den
            predicted_ratings_movies.append([int(round(predicted_ratings,0)),movie])

    def_pred = pd.DataFrame(predicted_ratings_movies,columns= ['rating', 'movie']).sort_values("rating", ascending=False)
    CS_movies = list(def_pred.iloc[:k].movie)
    return CS_movies
