import re
#from recommender import recommend_random, recommend_with_NMF, recommend_with_user_similarity
from flask import Flask,render_template,request
#from utils import movies, methods_recommendation, get_movie_frame, create_user_vector

# # example input of web application
# user_rating = {
#     'the lion king': 5,
#     'terminator': 5,
#     'star wars': 2
# }



# construct our flask instance, pass name of module
app = Flask(__name__)


# route decorator for mapping urls to functions
@app.route('/')
def welcome():
    # renders the html page as the output of this function

    return render_template(
        'first_page.html',
        name="Welcome to the Mushroom Classifier Application ",
        # movie=movies['title'].unique().tolist(),
        # recommended_method = methods_recommendation,
        )
    # 'movies' variable is passed from python file to the html file for accessing it inside the html file


@app.route('/camera')
def camera():
    return  render_template('camera.html') 



@app.route('/recommender')
def recommend():
    #read user input from url/webpage
    print(request.args)
    mushroom = request.args.getlist('') # taking lists of titles only from user input
    return  render_template('recommender.html') 
    # # 'movie_ids' variable is passed from python file to the html file for accessing it inside the html file
    pass
# Runs the app (main module)
if __name__=='__main__':
    app.run(debug=True,port=5000)




