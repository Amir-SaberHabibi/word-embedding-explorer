import time
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def type_text(text, delay=0.05, placeholder=None):
    typed_text = ""
    for char in text:
        typed_text += char
        if placeholder:
            placeholder.markdown(f"<div style='font-family: monospace;'>{typed_text}</div>", unsafe_allow_html=True)
        time.sleep(delay)
    return typed_text

def load_model(model_name):
    model = api.load(model_name)
    return model

def compute_embeddings_and_pca(model, words, n_components=3):
    embeddings = [model[word] for word in words if word in model]
    if len(embeddings) < 2:
        return None
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def get_nearest_neighbors(model, word, topn=5):
    return model.most_similar(word, topn=topn) if word in model else []

def word_similarity(model, word1, word2):
    return model.similarity(word1, word2) if word1 in model and word2 in model else None

def word_analogy(model, positive, negative):
    return model.most_similar(positive=positive, negative=negative, topn=1) if all(w in model for w in positive + negative) else None

def vector_search(model, query_word, topn=5):
    query_vector = model[query_word] if query_word in model else None
    if query_vector is not None:
        similarities = {}
        for word in model.key_to_index:
            if word != query_word:
                similarities[word] = cosine_similarity([query_vector], [model[word]])[0][0]
        sorted_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:topn]
        return sorted_words
    return []

def cluster_words(model, words, num_clusters=3):
    embeddings = [model[word] for word in words if word in model]
    if len(embeddings) < num_clusters:
        return None, None
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_
    return clusters, embeddings
