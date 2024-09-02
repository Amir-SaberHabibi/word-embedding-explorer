import streamlit as st
import time
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Function to create a typing effect
def type_text(text, delay=0.05):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(f"<div style='font-family: monospace;'>{typed_text}</div>", unsafe_allow_html=True)
        time.sleep(delay)
    return typed_text

@st.cache_resource
def load_model(model_name):
    model = api.load(model_name)
    return model

def compute_embeddings_and_pca(model, words, n_components=3):
    embeddings = [model[word] for word in words if word in model]
    if len(embeddings) < 2:
        return None, None
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

def main():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Home", "Word Analogies", "Nearest Neighbors", 
                                                               "Word Similarity", "Interactive Exploration", 
                                                               "Word Embeddings"])

    with tab1:
        st.markdown("<h3>Welcome to the advanced word embeddings vectorization tool!</h3>", unsafe_allow_html=True)
        type_text("Explore various features using the tabs above.", delay=0.05)

    with tab2:
        st.subheader("Word Analogies")
        st.markdown("Explore word analogies like `king - man + woman = queen`.")
        positive_words = st.text_input("Enter positive words (comma-separated):", value="king, woman")
        negative_words = st.text_input("Enter negative words (comma-separated):", value="man")

        if st.button("Compute Analogy"):
            model = load_model('glove-wiki-gigaword-100')
            positive_list = [w.strip() for w in positive_words.split(",")]
            negative_list = [w.strip() for w in negative_words.split(",")]
            result = word_analogy(model, positive=positive_list, negative=negative_list)
            if result:
                st.write(f"Result: **{result[0][0]}** with similarity score of **{result[0][1]:.4f}**")
            else:
                st.write("Could not compute analogy with the provided words.")

    with tab3:
        st.subheader("Nearest Neighbors")
        st.markdown("Find the closest words to a given word in the vector space.")
        word = st.text_input("Enter a word:")

        with st.expander("Select Number of Neighbors"):
            num_neighbors = st.slider("Number of neighbors:", 3, 10, 5)
            st.markdown("""
                <style>
                .css-1d391kg {
                    border: 2px solid lightgreen;
                }
                </style>
                """, unsafe_allow_html=True)
                
        if st.button("Find Nearest Neighbors"):
            model = load_model('glove-wiki-gigaword-100')
            neighbors = get_nearest_neighbors(model, word, topn=num_neighbors)
            if neighbors:
                st.write(f"Nearest neighbors to **{word}**:")
                df_neighbors = pd.DataFrame(neighbors, columns=["Neighbor", "Similarity Score"])
                st.dataframe(df_neighbors)
            else:
                st.write("Word not found in the model.")

    with tab4:
        st.subheader("Word Similarity")
        st.markdown("Compute and display the similarity score between two words.")
        word1 = st.text_input("Enter the first word:")
        word2 = st.text_input("Enter the second word:")

        if st.button("Compute Similarity"):
            model = load_model('glove-wiki-gigaword-100')
            similarity = word_similarity(model, word1, word2)
            if similarity is not None:
                st.write(f"Similarity between **{word1}** and **{word2}**: **{similarity:.4f}**")
            else:
                st.write("One or both words not found in the model.")

    with tab5:
        st.subheader("Interactive Exploration")
        st.markdown("Interactively explore the vector space by selecting or hovering over words.")
        words_input = st.text_input("Enter words (comma-separated):", value="king, queen, man, woman")

        with st.expander("Select Number of Words to Display"):
            num_words = st.slider("Number of words to display:", 1, 20, 5)
            st.markdown("""
                <style>
                .css-1d391kg {
                    border: 2px solid lightgreen;
                }
                </style>
                """, unsafe_allow_html=True)

        if words_input:
            model = load_model('glove-wiki-gigaword-100')
            words = [w.strip() for w in words_input.split(",") if w.strip() in model]
            words = words[:num_words]

            if words:
                reduced_embeddings = compute_embeddings_and_pca(model, words)
                if reduced_embeddings is not None:
                    fig = go.Figure(data=[go.Scatter3d(
                        x=reduced_embeddings[:, 0],
                        y=reduced_embeddings[:, 1],
                        z=reduced_embeddings[:, 2],
                        text=words,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=np.arange(len(words)),
                            colorscale='Viridis',
                            opacity=0.8,
                        ),
                    )])

                    fig.update_layout(
                        title='Interactive 3D Word Embeddings',
                        scene=dict(
                            xaxis_title='PCA Component 1',
                            yaxis_title='PCA Component 2',
                            zaxis_title='PCA Component 3',
                            xaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                            yaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                            zaxis=dict(showspikes=False, backgroundcolor='black', gridcolor='gray'),
                        ),
                        margin=dict(l=0, r=0, b=0, t=50),
                        hovermode='closest',
                        paper_bgcolor='black',
                        plot_bgcolor='black',
                        font_color="white",
                        scene_camera=dict(
                            eye=dict(x=1.25, y=1.25, z=1.25)
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Not enough words to compute PCA.")
            else:
                st.write("No valid words found in the model.")


    with tab6:
        st.subheader("Word Embeddings")
        st.markdown("Calculate and display the embeddings for a word or a sequence of words.")

        # Add model selection
        model_options = ['glove-wiki-gigaword-100', 'word2vec-google-news-300', 'fasttext-wiki-news-subwords-300']
        selected_model = st.selectbox("Choose an embedding model:", model_options)

        words_input = st.text_input("Enter a word or sequence of words:")
        if st.button("Get Embedding"):
            if words_input:
                model = load_model(selected_model)
                words = [w.strip() for w in words_input.split() if w.strip() in model]
                if words:
                    embeddings = {word: model[word] for word in words}
                    for word, embedding in embeddings.items():
                        st.code(f"{word}: {embedding.tolist()}", language="python")
                else:
                    st.write("No valid words found in the model.")
            else:
                st.write("Please enter a word or sequence of words.")

if __name__ == "__main__":
    main()
