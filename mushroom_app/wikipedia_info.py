#import wikipediaapi
import wikipedia

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