import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocesses text by removing HTML tags, special characters, etc."""
    # Add your preprocessing steps here (e.g., removing HTML, special chars, etc.)
    # For this example, I'll just lowercase and split for simplicity.
    text = text.lower()
    return text.split()

def extract_keywords_spacy(preprocessed_text, top_n=10):
    """Extracts keywords from preprocessed text using spaCy."""
    text = " ".join(preprocessed_text)
    doc = nlp(text)
    keywords = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN", "ADJ"):
            keywords.append(token.lemma_.lower())

    from collections import Counter
    word_counts = Counter(keywords)
    return word_counts.most_common(top_n)

def summarize_keywords(keywords):
    """Summarizes a list of keywords into a comma-separated string."""
    keyword_strings = [keyword for keyword, count in keywords]
    if not keyword_strings:
        return ""
    elif len(keyword_strings) == 1:
        return keyword_strings[0]
    else:
        return ", ".join(keyword_strings[:-1]) + ", " + keyword_strings[-1]

# Example usage:
text = "This provides an in-depth introduction to Python programming, focusing on data analysis, machine learning, and automation."
preprocessed_text = preprocess_text(text)
print("After Preprocessing:", preprocessed_text)

keywords = extract_keywords_spacy(preprocessed_text)
print("Extracted Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)

text = "This is another topic about Advanced Machine Learning and Deep Learning, with focus on Neural Networks and AI."
preprocessed_text = preprocess_text(text)
print("\nAfter Preprocessing:", preprocessed_text)

keywords = extract_keywords_spacy(preprocessed_text)
print("Extracted Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)

text = "This is a simple text."
preprocessed_text = preprocess_text(text)
print("\nAfter Preprocessing:", preprocessed_text)

keywords = extract_keywords_spacy(preprocessed_text)
print("Extracted Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)





#Using Rake 

print("Rake processing ...")

# !pip install rake-nltk
from rake_nltk import Rake
import re

def preprocess_text(text):
    """Preprocesses text by removing HTML tags, special characters, etc."""
    text = re.sub(r"<.*?>", "", text)        # Remove HTML tags
    text = re.sub(r"&amp;", "&", text)       # Decode HTML entities
    text = re.sub(r"[^\x00-\x7F]+", "", text) # Remove non-ASCII characters
    text = text.replace('\n', ' ').replace('\t', ' ') # Remove new lines and tabs
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def extract_keywords_rake(text, top_n=10):
    """Extracts keywords using RAKE."""
    r = Rake()  # You can customize Rake's parameters here if needed
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases()[:top_n]
    return ranked_phrases

def summarize_keywords(keywords):
    """Summarizes a list of keywords into a comma-separated string."""
    if not keywords:
        return ""
    elif len(keywords) == 1:
        return keywords[0]
    else:
        return ", ".join(keywords[:-1]) + ", " + keywords[-1]

# Example usage:
text = "This provides an in-depth introduction to Python programming, focusing on data analysis, machine learning, and automation."
preprocessed_text = preprocess_text(text)
print("After Preprocessing:", preprocessed_text)

keywords = extract_keywords_rake(preprocessed_text)
print("RAKE Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)

text = "This is another topic about Advanced Machine Learning and Deep Learning, with focus on Neural Networks and AI."
preprocessed_text = preprocess_text(text)
print("\nAfter Preprocessing:", preprocessed_text)

keywords = extract_keywords_rake(preprocessed_text)
print("RAKE Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)

text = "This is a simple text."
preprocessed_text = preprocess_text(text)
print("\nAfter Preprocessing:", preprocessed_text)

keywords = extract_keywords_rake(preprocessed_text)
print("RAKE Keywords:", keywords)

summary = summarize_keywords(keywords)
print("Summary:", summary)