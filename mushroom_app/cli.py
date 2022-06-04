from recommender import recommend_random
from utils import movies
# example input of web application
user_rating = {
    'the lion king': 5,
    'terminator': 5,
    'star wars': 2
}


# Please make sure that you output the ids and then modify the lookupmovieId to give the user the titles

### Terminal recommender:

print('>>>> Here are some movie recommendations for you<<<<')
print('')
print('Random movies')
movie_ids = recommend_random(movies, user_rating)
print(movie_ids)


