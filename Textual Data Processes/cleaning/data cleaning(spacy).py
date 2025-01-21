import spacy
import re
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Process text
text = "Apple is looking at buying U.K. startup for $1 billion *^*&^)*"

#Remvoing punctuations or special characters
text = re.sub(r'[^a-zA-z0-9\s]',' ',text) 
doc = nlp(text)

print(text)
# Tokenization
print("Tokens:")
tokens = [token.text for token in doc]
print(tokens)
tokens= []
for token in doc:
    if not (token.is_space or token.is_punct or token.is_stop):
        tokens.append(token.text)
    


print(tokens)

print(tokens)