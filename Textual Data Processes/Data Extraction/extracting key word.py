#Basic Frequency-Based Approach

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import re
import string

# nltk.download('punkt_tab')
# nltk.download('stopwords')

def  extracting_keywords_frequency(text, top_n=10):
    text = re.sub(r"<.*?>", "", text)        # Remove HTML tags
    text = re.sub(r"&amp;", "&", text)       # Decode HTML entities
    text = re.sub(r"[^\x00-\x7F]+", "", text) # Remove non-ASCII characters
    text = text.replace('\n', ' ').replace('\t', ' ') # Remove new lines and tabs
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    punctuation = str.maketrans("", "", string.punctuation)
    text = text.translate(punctuation)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_tokens = [w.lower() for w in word_tokens if not w.lower() in stop_words]

    word_counts = Counter(filtered_tokens)
    keywords= word_counts.most_common(top_n)
    return keywords

text = "this is an example text. this text is about keyword extraction. Keyword , Keywords extraction"
keywords = extracting_keywords_frequency(text)
print(keywords)

# with spaCy (with pos tag filtering)
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords_spacy(text,top_n=10):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN","ADJ"):
            keywords.append(token.lemma_.lower())
        
    word_counts = Counter(keywords)
    return word_counts.most_common(top_n)

text = "this is an example text. this text is about keyword extraction. Keyword , Keywords extraction "
keywords = extract_keywords_spacy(text)
print("After spaCy : ", keywords)



#Rake (Rapid automatic keyword Extraction)
from rake_nltk import Rake

def extract_keyword_rake(text,top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:top_n]
    return keywords

text = "this is an example text. this text is about keyword extraction. Keyword , Keywords extraction"
keywords = extract_keyword_rake(text)
print(f"After rake : {keywords}")



#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(text,top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_array = vectorizer.transform([text]).toarray()
    df = pd.DataFrame(tfidf_array, columns=feature_array)
    keywords = df.T.nlargest(top_n,0)
    return keywords


text = "this is an example text. this text is about keyword extraction. Keyword , Keywords extraction"
keywords = extract_keywords_tfidf(text)
print("Keywords (TF-IDF):", keywords)





import spacy
import re
import string

nlp = spacy.load("en_core_web_sm",disable = ["parser","near"]) # it will disable parser and ner for speed if not needed

def preprocess_batch(texts):
    cleaned_texts = []
    for text in texts:
        text = re.sub(r"<.*?>"," ",text)
        text = re.sub(r"&amp;","&",text)
        text = re.sub(r"[^\x00-\x7F]+"," ",text)
        text = text.replace('\n',' ').replace('\t',' ')
        text = re.sub(r'\s+',' ',text).strip()
        punctuation = str.maketrans('','',string.punctuation)
        text = text.translate(punctuation)
        cleaned_texts.append(text)
    docs = list(nlp.pipe(cleaned_texts))
    preprocessed_text = []
    for doc in docs:
        filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        preprocessed_text.append(filtered_tokens)

    return preprocessed_text


text = "this is a sample text for testing the sample and sample."


preprocessed_text = preprocess_batch(text)
for i, text in enumerate(preprocessed_text):
    print(f"Preprocessed text {i+1}: {text}")



# using spacy wit pos tag filtering

from collections import Counter

def extract_keywords_from_preprocessed(preprocessed_texts, top_n=10):
    all_keywords = []
    for text in preprocessed_texts:
        word_counts = Counter(text)
        keywords = word_counts.most_common(top_n)
        all_keywords.append(keywords)
    return all_keywords

keywords = extract_keywords_from_preprocessed(preprocessed_text)
for i, text in enumerate(keywords):
    print(f"Keywords for text {i+1}: {text}")