#import wikipediaapi
import wikipedia
import re
# wiki = wikipediaapi.Wikipedia('en',extract_format=wikipediaapi.ExtractFormat.WIKI)
# def fetch_wiki_text(prediction):
#     """This function gives the abstract text of the wiki page search of item
#     """
#     text = wiki.page(prediction).summary
    
#     return text    

# def fetch_wiki_url(prediction):
#     """This function gives the url of the wiki page search of item
#     """
#     url_item = wiki.page(prediction)
    
#     return url_item.fullurl

mentions_regex= '@[A-Za-z0-9]+'
url_regex='https?:\/\/\S+' #this will not catch all possible URLs     ###add this to etl script
hashtag_regex= '#'
rt_regex= 'RT\s'
apostroph = "(?i)\\b(?<!')(?![AOI])\\\b"

def clean_text(text):
    text = re.sub(mentions_regex, '', text)  #removes @mentions
    text = re.sub(hashtag_regex, '', text) #removes hashtag symbol
    text = re.sub(rt_regex, '', text) #removes RT to announce retweet
    text = re.sub(url_regex, '', text) #removes most URLs
    text = re.sub(apostroph, '', text) #removes most URLs
    text = re.sub(r"'", " ", text)
    return text


def fetch_wiki_image(prediction):
    """This function gives the first image of the wiki page search of item
    """
    # accurate_name = '_'+prediction+'_'
    df = wikipedia.page(prediction).images
    item_image_wiki_url = list(filter(lambda x: prediction in x, df))[0]
    #item_image_wiki_url = list(filter(lambda x: accurate_name in x, df))[0]
    print(item_image_wiki_url)
    return item_image_wiki_url

#fetch_wiki_image('Trametes_versicolor')
def fetch_wiki_text_alternative(prediction):
    prediction = clean_text(prediction)
    """This function gives the abstract text of the wiki page search of item

    we use this function as alternative of the function fetch_wiki_text() in case the wikepediaapi crashes
    """
    text = wikipedia.page(prediction).summary
    
    return text   

def fetch_wiki_url_alternative(prediction):
    """This function gives the url of the wiki page search of item
    we use this function as alternative of the function fetch_wiki_url() in case the wikepediaapi crashes
    """

    url_item = wikipedia.page(prediction)
    
    return url_item.url
