# Importing dependencies from transformers
from transformers import PegasusForConditionalGeneration, AutoTokenizer
import re



# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
# Load model 
model_pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")


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


def give_abstract_text(text):
    text = clean_text(text)
    # Create tokens - number representation of our text
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    # Summarize 
    summary = model_pegasus.generate(**tokens)
    # Decode summary
    text_me = tokenizer.decode(summary[0])
    return text_me



