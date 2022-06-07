# Importing dependencies from transformers
from transformers import PegasusForConditionalGeneration, AutoTokenizer



# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
# Load model 
model_pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")




def give_abstract_text(text ):
    # Create tokens - number representation of our text
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    # Summarize 
    summary = model_pegasus.generate(**tokens)
    # Decode summary
    text_me = tokenizer.decode(summary[0])
    return text_me


