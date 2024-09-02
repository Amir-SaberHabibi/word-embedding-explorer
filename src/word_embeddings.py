import streamlit as st
from components import load_model

def show_word_embeddings():
    st.subheader("Word Embeddings")
    st.markdown("Calculate and display the embeddings for a word or a sequence of words.")

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