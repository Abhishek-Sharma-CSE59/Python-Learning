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


course_descriptions = [
    "<p>This is a <strong>very</strong> interesting course about <em>Python</em> programming.</p> Learn the basics &amp; advanced topics!",
    "Introduction to Data Science. Learn about Machine Learning & AI.",
    "A course on Web Development using HTML, CSS & JavaScript.",
    "This course covers advanced topics in <b>Deep Learning</b> and Neural Networks.",
    "Another course about Python.\nNew line\tTab"
]

preprocessed_descriptions = preprocess_batch(course_descriptions)
for i, description in enumerate(preprocessed_descriptions):
    print(f"Preprocessed description {i+1}: {description}")



# Clustering with usning spacy and k-means

import spacy
from sklearn.cluster import KMeans
import numpy as np

nlp = spacy.load("en_core_web_lg")

def cluster_keywords(Keywords, n_clusters=5):
    keyword_vectors = [nlp(keyword).vector for keyword in keywords]
    keyword_vectors = np.array(keyword_vectors)

    kmeans = KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
    kmeans.fit(keyword_vectors)

    clusters = {}
    for i,label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(keywords[i])
    return clusters

keywords = ["python", "programming", "coding", "data", "science", "analysis", "machine", "learning", "algorithm", "network", "web", "html", "css", "javascript"]

clusters = cluster_keywords(keywords)
for label, cluster_keywords in clusters.items():
    print(f"Cluster {label}: {cluster_keywords}")



#topic Modeling

import nltk
import spacy
import gensim
from gensim import corpora
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt

# Download NLTK resources (do this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Load spaCy
nlp = spacy.load("en_core_web_lg")  # Use a larger model for word vectors

# Sample texts (replace with your actual data)
texts = [
    "Python programming is fun and useful for data science.",
    "Machine learning algorithms are used in many applications.",
    "Web development involves HTML, CSS, and JavaScript.",
    "Data analysis is important for business decisions.",
    "Deep learning is a subfield of machine learning.",
    "Natural language processing is a part of artificial intelligence.",
    "Software development is a broad field with many specializations.",
    "Cloud computing is changing the way we store and access data.",
    "Cybersecurity is crucial for protecting sensitive information.",
    "Databases are essential for managing large amounts of data.",
]

# 1. LDA (Latent Dirichlet Allocation)
def topic_modeling_lda(texts, num_topics=3):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    return lda_model, dictionary

lda_model, dictionary = topic_modeling_lda(texts)
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# 2. LSI/LSA (Latent Semantic Indexing/Analysis)
def topic_modeling_lsi(texts, num_topics=3):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi_model = gensim.models.LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)
    return lsi_model, dictionary

lsi_model, dictionary = topic_modeling_lsi(texts)
print("\nLSI Topics:")
for idx, topic in lsi_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# 3. NMF (Non-negative Matrix Factorization)
def topic_modeling_nmf(texts, num_topics=3):
    vectorizer = TfidfVectorizer(stop_words='english') #NMF needs numeric input
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    print("\nNMF Topics:")
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[:-11:-1] #get the top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx}: {' '.join(top_words)}")
    return nmf_model, vectorizer

nmf_model, vectorizer = topic_modeling_nmf(texts)

# 4. BERTopic
# !pip install bertopic
def topic_modeling_bertopic(texts):
    # Initialize and fit BERTopic
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(texts)
    
    # Display topics
    print("\nBERTopic Topics:")
    print(topic_model.get_topic_info())  # Display the topic summary table
    
    # Example: Find topics related to a specific keyword
    keyword = "example"  # Replace with your search keyword
    similar_topics, similarity_scores = topic_model.find_topics(keyword, top_n=5)
    print(f"\nTopics related to '{keyword}':")
    for topic, score in zip(similar_topics, similarity_scores):
        print(f"Topic {topic}: Score {score}")
    
    return topic_model

bertopic_model = topic_modeling_bertopic(texts)

# 5. Clustering (K-Means with spaCy embeddings)
def cluster_keywords(keywords, n_clusters=3):
    keyword_vectors = [nlp(keyword).vector for keyword in keywords]
    keyword_vectors = np.array(keyword_vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(keyword_vectors)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(keywords[i])
    return clusters

keywords = ["python", "programming", "coding", "data", "science", "analysis", "machine", "learning", "algorithm", "network", "web", "html", "css", "javascript", "cloud", "computing", "cybersecurity", "databases"]
clusters = cluster_keywords(keywords)
print("\nK-Means Clusters:")
for label, cluster_keywords in clusters.items():
    print(f"Cluster {label}: {cluster_keywords}")