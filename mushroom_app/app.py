import re
from NLP_wiki_abstraction import give_abstract_text
from wikipedia_info import fetch_wiki_text, fetch_wiki_url, fetch_wiki_image
from recommender import mushroom_classification, mushroom_depict
from flask import Flask,render_template,request

# construct our flask instance, pass name of module
app = Flask(__name__)


# route decorator for mapping urls to functions
@app.route('/')
def welcome():
    # renders the html page as the output of this function
    return render_template(
        'firstpage.html',
        name="Welcome to the Mushroom Classifier Application "
        )
    # 'movies' variable is passed from python file to the html file for accessing it inside the html file


@app.route('/camera')
def camera():
    return  render_template('camera.html') 



@app.route('/recommender')
def recommend():
    #read user input from url/webpage
    #model_ann, model_pegasus = initiate_models()
    test = request.args.getlist('test') # taking lists of titles only from user input
    image_directory = '../static/data/' + test[0]
    mush_classified,  prediction, kind = mushroom_classification(test[0])
    #test_image= mushroom_depict(test[0])
    wiki_image_url= fetch_wiki_image(prediction[0])
    wiki_text = fetch_wiki_text(prediction[0])
    nlp_abstract = give_abstract_text(wiki_text)
    wiki_url = fetch_wiki_url(prediction[0])
    return  render_template(
        'recommender.html', mush_classified = mush_classified, wiki_text=wiki_text,
        wiki_url = wiki_url, prediction = prediction, wiki_image_url=wiki_image_url, 
        image_directory = image_directory, kind = kind,nlp_abstract =nlp_abstract)

# Runs the app (main module)
if __name__=='__main__':
    #app.run(host='0.0.0.0', port=5000)
    app.run(debug=True,port=5400)
