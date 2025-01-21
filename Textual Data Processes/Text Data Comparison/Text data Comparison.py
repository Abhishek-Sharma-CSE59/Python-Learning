import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import jaccard
from gensim.models import KeyedVectors
# from gensim.similarity import soft_cos_sim
from gensim.corpora import Dictionary

# Example dataset
Samples = [
    "Advanced Machine Learning and Deep Learning, with focus on Neural Networks and AI.",
    "Introduction to Programming in Python and Basics of AI.",
    "Data Science and Machine Learning using Python and AI techniques.",
]

# Preprocessing: Lowercase and simple tokenization
cleaned_samples = [desc.lower().replace(",", "").replace(".", "") for desc in Samples]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_samples)

# Method 1: Cosine Similarity
print("Cosine Similarity:")
cos_sim = cosine_similarity(tfidf_matrix)
print(cos_sim)

# Method 2: Euclidean Distance
print("\nEuclidean Distance:")
euclidean_dist = euclidean_distances(tfidf_matrix)
print(euclidean_dist)

# Method 3: Jaccard Similarity
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    return len(set1 & set2) / len(set1 | set2)

print("\nJaccard Similarity:")
jaccard_sim = np.zeros((len(cleaned_samples), len(cleaned_samples)))
for i in range(len(cleaned_samples)):
    for j in range(len(cleaned_samples)):
        jaccard_sim[i, j] = jaccard_similarity(cleaned_samples[i], cleaned_samples[j])
print(jaccard_sim)









# # Method 4: Soft Cosine Similarity
# print("\nSoft Cosine Similarity:")
# # Load pre-trained word vectors (replace with actual file path to word embeddings like GloVe or Word2Vec)
# # word_vectors = KeyedVectors.load_word2vec_format("path/to/word2vec/file", binary=True)
# # For demonstration, we create a dummy word embedding dictionary
# dummy_word_vectors = {
#     "advanced": np.random.rand(100),
#     "machine": np.random.rand(100),
#     "learning": np.random.rand(100),
#     "python": np.random.rand(100),
#     "ai": np.random.rand(100),
#     "data": np.random.rand(100),
#     "science": np.random.rand(100),
# }

# # Create a Gensim dictionary and similarity matrix
# dictionary = Dictionary([desc.split() for desc in cleaned_samples])
# similarity_matrix = np.zeros((len(dictionary), len(dictionary)))

# # Fill the similarity matrix using dummy vectors
# for i, word1 in enumerate(dictionary.token2id.keys()):
#     for j, word2 in enumerate(dictionary.token2id.keys()):
#         vec1, vec2 = dummy_word_vectors.get(word1, np.zeros(100)), dummy_word_vectors.get(word2, np.zeros(100))
#         similarity_matrix[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

# # Soft Cosine Similarity
# soft_cos_sim = np.zeros((len(cleaned_samples), len(cleaned_samples)))
# for i, desc1 in enumerate(cleaned_samples):
#     bow1 = dictionary.doc2bow(desc1.split())
#     for j, desc2 in enumerate(cleaned_samples):
#         bow2 = dictionary.doc2bow(desc2.split())
#         soft_cos_sim[i, j] = soft_cos_sim(bow1, bow2, similarity_matrix)

# print(soft_cos_sim)

# # Method 5: Pre-trained Embeddings with Cosine Similarity
# print("\nPre-trained Embeddings with Cosine Similarity:")
# # Generate dummy sentence embeddings by averaging word vectors
# sentence_embeddings = []
# for desc in cleaned_samples:
#     vectors = [dummy_word_vectors.get(word, np.zeros(100)) for word in desc.split()]
#     sentence_embeddings.append(np.mean(vectors, axis=0))

# # Compute cosine similarity for sentence embeddings
# pretrained_cos_sim = cosine_similarity(sentence_embeddings)
# print(pretrained_cos_sim)
