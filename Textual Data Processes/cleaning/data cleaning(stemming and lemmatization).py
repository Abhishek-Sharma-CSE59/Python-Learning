import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import spacy

# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    text = text.lower()
    text = re.sub(r',.*?>','',text)
    text = re.sub(r'&amp;','&',text)
    text = re.sub(r'[^\x00-\x7F]+',"",text)
    text = text.replace('\n',' ').replace('\t',' ')
    text = re.sub(r'\s+',' ',text)
    tokens = []
    for token in doc:
        if not (token.is_punct or token.is_space or token.is_stop):
            tokens.append(str(token))
    return tokens


filtered_tokens = preprocess_text("hello this is a sample programming test running")
filtered_tokens_for_spaCy = preprocess_text("Hello this is a sample text for testing spacy and doing the programming and doing some learning &*^%%$")
print(filtered_tokens_for_spaCy)

print(f"Filtered Words : {filtered_tokens}")
#stemming
print("Stemming is started :")
ps = PorterStemmer()
stemmed_words = []
for w in filtered_tokens:
  if isinstance(w, str):
    stemmed_words.append(ps.stem(w))
  else:
    print(f"Skipping integer: {w}")
print("stemmed Words:",stemmed_words)

print("Stemming ended:")

# Stemming in nltk not working properly and slow for some reasoms

# Lemmatization
print("Lemmatization started :")
lemmatizer = WordNetLemmatizer()
lemmatized_word = lemmatizer.lemmatize("cars")
print(lemmatized_word)

lemmatized_list = []
for w in filtered_tokens:
   lemmatized_list.append(lemmatizer.lemmatize(str(w)))
# lemmatized_word = lemmatizer.lemmatize(w for str(w) in filtered_tokens)

print(lemmatized_list)



# lemmatization in spaCy

def lemmatization(tokens):
   text = " ".join(tokens)
   doc = nlp(text)
   print(text)
   lemmatized_list = [token.lemma_ for token in doc]
   print(lemmatized_list)
#    return lemmatized_tokens


lemmatization(filtered_tokens_for_spaCy)


doc = nlp('The quicking brown fox jumping over the lazy dog')
for token in doc:
    print(token.text, '-->', token.lemma_)


#Lemmatization with spacy 

import spacy

nlp = spacy.load("en_core_web_sm")

def lemmatize_tokens_with_spacy(tokens):
    """Lemmatizes a list of tokens using spaCy."""
    text = " ".join(tokens) #Very important step
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

def lemmatize_tokens_with_spacy_filtered(tokens):
    """Lemmatizes a list of tokens using spaCy and filters out spaces and punctuation."""
    text = " ".join(tokens) #Very important step
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_space and not token.is_punct]
    return lemmatized_tokens

# Example usage:
tokens = ["running", "runs", "ran", "better", "best", ".", ",", " "]

lemmatized = lemmatize_tokens_with_spacy(tokens)
print("Lemmatized tokens:", lemmatized)
# Output: Lemmatized tokens: ['run', 'run', 'run', 'well', 'good', '.', ',', ' ']

lemmatized_filtered = lemmatize_tokens_with_spacy_filtered(tokens)
print("Lemmatized tokens filtered:", lemmatized_filtered)
# Output: Lemmatized tokens filtered: ['run', 'run', 'run', 'well', 'good']

tokens = ["This", "is", "a", "sentence", "with", "multiple", "words", "."]
lemmatized = lemmatize_tokens_with_spacy(tokens)
print("Lemmatized tokens:", lemmatized)
# Output: Lemmatized tokens: ['this', 'be', 'a', 'sentence', 'with', 'multiple', 'word', '.']

lemmatized_filtered = lemmatize_tokens_with_spacy_filtered(tokens)
print("Lemmatized tokens filtered:", lemmatized_filtered)
# Output: Lemmatized tokens filtered: ['this', 'be', 'a', 'sentence', 'with', 'multiple', 'word']